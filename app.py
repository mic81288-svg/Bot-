from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,DOGEUSDT").split(",")]
CONFIG = {
    "initial_capital": float(os.getenv("INITIAL_CAPITAL", "800")),
    "risk_per_trade": float(os.getenv("RISK_PER_TRADE", "0.10")),
    "risk_reward": float(os.getenv("RISK_REWARD_RATIO", "3")),
    "max_daily_loss": float(os.getenv("MAX_DAILY_LOSS", "0.05")),
    "max_positions": int(os.getenv("MAX_OPEN_POSITIONS", "1")),
    "interval": os.getenv("TIMEFRAME", "15m"),
    "min_score": int(os.getenv("MIN_SIGNAL_SCORE", "75")),
    "poll_seconds": int(os.getenv("POLL_SECONDS", "30")),
}
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
ENABLE_ORDERS = os.getenv("ENABLE_ORDERS", "false").lower() == "true"
BASE_URL = os.getenv("BINANCE_BASE_URL", "https://testnet.binance.vision/api" if TESTNET else "https://api.binance.com/api")

state: dict[str, Any] = {
    "balance": CONFIG["initial_capital"], "initial_balance": CONFIG["initial_capital"],
    "is_trading": False, "total_trades": 0, "winning_trades": 0,
    "losing_trades": 0, "total_profit": 0.0, "daily_loss": 0.0,
    "open_positions": [], "closed_trades": [], "last_update": datetime.now(timezone.utc).isoformat(),
    "last_error": None, "mode": "TESTNET" if TESTNET else "LIVE", "orders_enabled": ENABLE_ORDERS,
}
lock = threading.RLock()
session = requests.Session()
exchange_info_cache: dict[str, dict[str, Any]] = {}
last_candle: dict[str, int] = {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def api_request(method: str, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> Any:
    params = dict(params or {})
    if signed:
        if not API_KEY or not API_SECRET:
            raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET are required for orders")
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urlencode(params)
        params["signature"] = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
    response = session.request(method, BASE_URL + path, params=params, headers=headers, timeout=12)
    response.raise_for_status()
    return response.json()


def klines(symbol: str) -> np.ndarray:
    raw = api_request("GET", "/v3/klines", {"symbol": symbol, "interval": CONFIG["interval"], "limit": 250})
    # Ignore the currently-forming candle to avoid repainting signals.
    rows = raw[:-1] if len(raw) > 2 else raw
    return np.array([[float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), int(x[0])] for x in rows])


def rsi(closes: np.ndarray, period: int = 14) -> float:
    delta = np.diff(closes)
    gains = np.maximum(delta, 0)
    losses = np.maximum(-delta, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    return float(100 - (100 / (1 + avg_gain / avg_loss)))


def candle_pattern(c: np.ndarray) -> dict[str, Any]:
    o, h, l, close = c[-1, :4]
    prev_o, prev_h, prev_l, prev_c = c[-2, :4]
    body = abs(close - o)
    rng = max(h - l, 1e-12)
    upper = h - max(o, close)
    lower = min(o, close) - l
    bullish_engulf = prev_c < prev_o and close > o and close >= prev_o and o <= prev_c
    bearish_engulf = prev_c > prev_o and close < o and o >= prev_c and close <= prev_o
    bullish_pin = lower >= body * 2 and upper <= body and close > o
    bearish_pin = upper >= body * 2 and lower <= body and close < o
    return {"bullish": bool(bullish_engulf or bullish_pin), "bearish": bool(bearish_engulf or bearish_pin),
            "pattern": "bullish_engulfing" if bullish_engulf else "bearish_engulfing" if bearish_engulf else "bullish_pinbar" if bullish_pin else "bearish_pinbar" if bearish_pin else "none",
            "body_ratio": round(body / rng, 3)}


def structure(c: np.ndarray) -> dict[str, Any]:
    highs, lows, closes = c[:, 1], c[:, 2], c[:, 3]
    swing_high = float(np.max(highs[-30:-3]))
    swing_low = float(np.min(lows[-30:-3]))
    price = float(closes[-1])
    bos_up = price > swing_high
    bos_down = price < swing_low
    # A liquidity sweep is a wick through a recent level followed by a close back inside.
    sweep_low = bool(lows[-1] < swing_low and price > swing_low)
    sweep_high = bool(highs[-1] > swing_high and price < swing_high)
    return {"swing_high": swing_high, "swing_low": swing_low, "bos": "BULLISH" if bos_up else "BEARISH" if bos_down else "NONE", "liquidity_sweep": "LOW" if sweep_low else "HIGH" if sweep_high else "NONE"}


def analyze(symbol: str) -> dict[str, Any]:
    c = klines(symbol)
    close = c[:, 3]
    current = float(close[-1])
    ema20 = float(__import__("pandas").Series(close).ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(__import__("pandas").Series(close).ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(__import__("pandas").Series(close).ewm(span=200, adjust=False).mean().iloc[-1])
    rr = structure(c)
    cp = candle_pattern(c)
    score = 0
    reasons: list[str] = []
    if ema20 > ema50 > ema200: score += 25; reasons.append("EMA trend bullish")
    if current > ema20: score += 10; reasons.append("price above EMA20")
    if rr["bos"] == "BULLISH": score += 25; reasons.append("bullish BOS")
    if rr["liquidity_sweep"] == "LOW": score += 20; reasons.append("sell-side liquidity sweep")
    if cp["bullish"]: score += 20; reasons.append(cp["pattern"])
    if 50 <= rsi(close) <= 70: score += 10; reasons.append("RSI confirmation")
    signal = "BUY" if score >= CONFIG["min_score"] else "NEUTRAL"
    atr = float(np.mean(np.maximum(c[-15:, 1] - c[-15:, 2], current * 0.001)))
    stop = min(current - atr * 1.5, rr["swing_low"] * 0.999) if signal == "BUY" else current - atr * 1.5
    risk_distance = max(current - stop, current * 0.001)
    target = current + risk_distance * CONFIG["risk_reward"]
    return {"symbol": symbol, "price": round(current, 8), "rsi": round(rsi(close), 2), "ema20": round(ema20, 8), "ema50": round(ema50, 8), "ema200": round(ema200, 8), "signal": signal, "strength": min(score, 100), "score": score, "reasons": reasons, "pattern": cp["pattern"], "market_structure": rr, "stop_loss": round(stop, 8), "take_profit": round(target, 8), "risk_reward": CONFIG["risk_reward"], "candle_time": int(c[-1, 5])}


def symbol_rules(symbol: str) -> dict[str, float]:
    if symbol not in exchange_info_cache:
        info = api_request("GET", "/v3/exchangeInfo", {"symbol": symbol})["symbols"][0]
        filters = {x["filterType"]: x for x in info["filters"]}
        exchange_info_cache[symbol] = {"step": float(filters.get("LOT_SIZE", {}).get("stepSize", "0.000001")), "min_qty": float(filters.get("LOT_SIZE", {}).get("minQty", "0.000001")), "min_notional": float(filters.get("NOTIONAL", filters.get("MIN_NOTIONAL", {})).get("minNotional", "5"))}
    return exchange_info_cache[symbol]


def floor_step(value: float, step: float) -> float:
    return float(np.floor(value / step) * step)


def place_testnet_order(a: dict[str, Any]) -> dict[str, Any] | None:
    if not ENABLE_ORDERS:
        return {"paper": True, "status": "SIGNAL_ONLY"}
    rules = symbol_rules(a["symbol"])
    risk_money = state["balance"] * CONFIG["risk_per_trade"]
    quantity = floor_step(risk_money / max(a["price"] - a["stop_loss"], a["price"] * .001), rules["step"])
    quantity = max(quantity, rules["min_qty"])
    if quantity * a["price"] < rules["min_notional"]:
        raise RuntimeError(f"Order below minimum notional for {a['symbol']}")
    return api_request("POST", "/v3/order", {"symbol": a["symbol"], "side": "BUY", "type": "MARKET", "quantity": f"{quantity:.8f}", "newOrderRespType": "RESULT"}, signed=True)


def open_position(a: dict[str, Any]) -> None:
    with lock:
        if a["signal"] != "BUY" or len(state["open_positions"]) >= CONFIG["max_positions"] or any(x["symbol"] == a["symbol"] for x in state["open_positions"]): return
        order = place_testnet_order(a)
        risk_money = state["balance"] * CONFIG["risk_per_trade"]
        quantity = risk_money / max(a["price"] - a["stop_loss"], a["price"] * .001)
        state["open_positions"].append({"symbol": a["symbol"], "side": "BUY", "entry_price": a["price"], "quantity": quantity, "stop_loss": a["stop_loss"], "take_profit": a["take_profit"], "order": order, "opened_at": now()})


def resolve_positions() -> None:
    with lock:
        remaining = []
        for p in state["open_positions"]:
            try: current = float(api_request("GET", "/v3/ticker/price", {"symbol": p["symbol"]})["price"])
            except Exception: remaining.append(p); continue
            hit_tp, hit_sl = current >= p["take_profit"], current <= p["stop_loss"]
            if not (hit_tp or hit_sl): remaining.append(p); continue
            exit_price = p["take_profit"] if hit_tp else p["stop_loss"]
            pnl = (exit_price - p["entry_price"]) * p["quantity"]
            state["balance"] += pnl; state["total_profit"] += pnl; state["total_trades"] += 1
            if pnl >= 0: state["winning_trades"] += 1
            else: state["losing_trades"] += 1; state["daily_loss"] += abs(pnl)
            state["closed_trades"].append({"symbol": p["symbol"], "pnl": round(pnl, 4), "status": "WIN" if pnl >= 0 else "LOSS", "exit_price": exit_price, "time": now()})
        state["open_positions"] = remaining


def loop() -> None:
    while True:
        try:
            if state["is_trading"] and state["daily_loss"] < CONFIG["initial_capital"] * CONFIG["max_daily_loss"]:
                for symbol in SYMBOLS:
                    a = analyze(symbol)
                    if last_candle.get(symbol) != a["candle_time"]:
                        last_candle[symbol] = a["candle_time"]
                        open_position(a)
                resolve_positions(); state["last_update"] = now(); state["last_error"] = None
            time.sleep(CONFIG["poll_seconds"])
        except Exception as exc:
            state["last_error"] = str(exc); print("trading loop error:", exc); time.sleep(CONFIG["poll_seconds"])

threading.Thread(target=loop, daemon=True).start()

@app.get("/")
def index(): return render_template("index.html")

@app.post("/api/trading/start")
def start(): state["is_trading"] = True; return jsonify({"is_trading": True, "status": "SMC trading started", "mode": state["mode"], "orders_enabled": ENABLE_ORDERS})

@app.post("/api/trading/stop")
def stop(): state["is_trading"] = False; return jsonify({"is_trading": False, "status": "Trading stopped"})

@app.get("/api/stats")
def stats():
    with lock:
        total = state["total_trades"]
        return jsonify({**state, "open_positions": len(state["open_positions"]), "win_rate": round(state["winning_trades"] / total * 100, 2) if total else 0, "profit_percent": round((state["balance"] / state["initial_balance"] - 1) * 100, 2)})

@app.get("/api/positions")
def positions(): return jsonify({"open_positions": state["open_positions"], "closed_trades": state["closed_trades"][-50:]})

@app.get("/api/analyze/<symbol>")
def analyze_route(symbol):
    try: return jsonify(analyze(symbol.upper()))
    except Exception as exc: return jsonify({"error": str(exc), "symbol": symbol.upper()}), 502

@app.get("/api/analyze-all")
def analyze_all():
    result = {}
    for symbol in SYMBOLS:
        try: result[symbol] = analyze(symbol)
        except Exception as exc: result[symbol] = {"symbol": symbol, "error": str(exc)}
    return jsonify(result)

if __name__ == "__main__": app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
