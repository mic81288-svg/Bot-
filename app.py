from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
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

SYMBOLS = [x.strip().upper() for x in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,DOGEUSDT").split(",") if x.strip()]
SUPPORTED_INTERVALS = {"1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h"}
TIMEFRAMES = [x.strip() for x in os.getenv("TIMEFRAMES", "1s,1m,3m,5m,15m,30m,1h,2h,4h").split(",") if x.strip() in SUPPORTED_INTERVALS]
TRADE_TIMEFRAME = os.getenv("TRADE_TIMEFRAME", "15m") if os.getenv("TRADE_TIMEFRAME", "15m") in SUPPORTED_INTERVALS else "15m"
CONFIG = {
    "initial_capital": float(os.getenv("INITIAL_CAPITAL", "800")),
    "risk_per_trade": float(os.getenv("RISK_PER_TRADE", "0.10")),
    "risk_reward": float(os.getenv("RISK_REWARD_RATIO", "3")),
    "max_daily_loss": float(os.getenv("MAX_DAILY_LOSS", "0.05")),
    "max_positions": int(os.getenv("MAX_OPEN_POSITIONS", "1")),
    "min_score": int(os.getenv("MIN_SIGNAL_SCORE", "75")),
    "poll_seconds": max(1, int(os.getenv("POLL_SECONDS", "15"))),
    "candle_limit": min(1000, max(210, int(os.getenv("CANDLE_LIMIT", "250")))),
}
API_KEY, API_SECRET = os.getenv("BINANCE_API_KEY", ""), os.getenv("BINANCE_API_SECRET", "")
TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
ENABLE_ORDERS = os.getenv("ENABLE_ORDERS", "false").lower() == "true"
BASE_URL = os.getenv("BINANCE_BASE_URL", "https://testnet.binance.vision/api" if TESTNET else "https://api.binance.com/api")

state: dict[str, Any] = {
    "balance": CONFIG["initial_capital"], "initial_balance": CONFIG["initial_capital"],
    "is_trading": False, "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
    "total_profit": 0.0, "daily_loss": 0.0, "open_positions": [], "closed_trades": [],
    "last_update": datetime.now(timezone.utc).isoformat(), "last_error": None,
    "mode": "TESTNET" if TESTNET else "LIVE", "orders_enabled": ENABLE_ORDERS,
}
lock = threading.RLock()
session = requests.Session()
exchange_cache: dict[str, dict[str, float]] = {}
last_processed: dict[tuple[str, str], int] = {}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_binance(method: str, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> Any:
    params = dict(params or {})
    if signed:
        if not API_KEY or not API_SECRET:
            raise RuntimeError("Binance API credentials are missing")
        params.update(timestamp=int(time.time() * 1000), recvWindow=5000)
        query = urlencode(params)
        params["signature"] = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    response = session.request(method, BASE_URL + path, params=params, headers={"X-MBX-APIKEY": API_KEY} if API_KEY else {}, timeout=15)
    if not response.ok:
        raise RuntimeError(f"Binance {response.status_code}: {response.text[:300]}")
    return response.json()


def candles(symbol: str, interval: str) -> np.ndarray:
    raw = request_binance("GET", "/v3/klines", {"symbol": symbol, "interval": interval, "limit": CONFIG["candle_limit"]})
    if len(raw) < 205:
        raise RuntimeError(f"Not enough {interval} candles for {symbol}")
    # Do not trade from an unfinished candle.
    rows = raw[:-1]
    return np.array([[float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), int(k[0])] for k in rows], dtype=float)


def ema(values: np.ndarray, period: int) -> float:
    alpha, result = 2 / (period + 1), float(values[0])
    for value in values[1:]: result = alpha * float(value) + (1 - alpha) * result
    return result


def rsi(values: np.ndarray, period: int = 14) -> float:
    changes = np.diff(values)[-period:]
    gain, loss = np.mean(np.maximum(changes, 0)), np.mean(np.maximum(-changes, 0))
    return 100.0 if loss == 0 else float(100 - 100 / (1 + gain / loss))


def atr(c: np.ndarray, period: int = 14) -> float:
    ranges = np.maximum(c[-period:, 1] - c[-period:, 2], np.maximum(abs(c[-period:, 1] - c[-period-1:-1, 3]), abs(c[-period:, 2] - c[-period-1:-1, 3])))
    return float(np.mean(ranges))


def candle_reading(c: np.ndarray) -> dict[str, Any]:
    o, h, l, close = c[-1, :4]; po, ph, pl, pc = c[-2, :4]
    body, full = abs(close - o), max(h - l, 1e-12)
    upper, lower = h - max(o, close), min(o, close) - l
    bull_engulf = pc < po and close > o and close >= po and o <= pc
    bear_engulf = pc > po and close < o and o >= pc and close <= po
    bull_pin, bear_pin = lower >= max(body, full * .01) * 2 and upper <= body, upper >= max(body, full * .01) * 2 and lower <= body
    name = "bullish_engulfing" if bull_engulf else "bearish_engulfing" if bear_engulf else "bullish_pinbar" if bull_pin and close > o else "bearish_pinbar" if bear_pin and close < o else "none"
    return {"pattern": name, "bullish": bool(bull_engulf or (bull_pin and close > o)), "bearish": bool(bear_engulf or (bear_pin and close < o)), "body_ratio": round(body / full, 3)}


def smc(c: np.ndarray) -> dict[str, Any]:
    high, low, close = c[:, 1], c[:, 2], c[:, 3]
    swing_high, swing_low, price = float(np.max(high[-30:-3])), float(np.min(low[-30:-3])), float(close[-1])
    return {"swing_high": swing_high, "swing_low": swing_low, "bos": "BULLISH" if price > swing_high else "BEARISH" if price < swing_low else "NONE", "liquidity_sweep": "LOW" if low[-1] < swing_low and price > swing_low else "HIGH" if high[-1] > swing_high and price < swing_high else "NONE"}


def analyze(symbol: str, interval: str = TRADE_TIMEFRAME) -> dict[str, Any]:
    c = candles(symbol, interval); close = c[:, 3]; price = float(close[-1]); structure = smc(c); pattern = candle_reading(c)
    e20, e50, e200, current_rsi = ema(close, 20), ema(close, 50), ema(close, 200), rsi(close)
    score, reasons = 0, []
    if e20 > e50 > e200: score += 25; reasons.append("EMA trend bullish")
    if price > e20: score += 10; reasons.append("price above EMA20")
    if structure["bos"] == "BULLISH": score += 25; reasons.append("bullish BOS")
    if structure["liquidity_sweep"] == "LOW": score += 20; reasons.append("sell-side liquidity sweep")
    if pattern["bullish"]: score += 20; reasons.append(pattern["pattern"])
    if 50 <= current_rsi <= 70: score += 10; reasons.append("RSI confirmation")
    signal = "BUY" if score >= CONFIG["min_score"] else "NEUTRAL"
    distance = max(atr(c) * 1.5, price * 0.001)
    stop = min(price - distance, structure["swing_low"] * 0.999) if signal == "BUY" else price - distance
    risk_distance = max(price - stop, price * 0.001)
    return {"symbol": symbol, "timeframe": interval, "price": round(price, 8), "rsi": round(current_rsi, 2), "ema20": round(e20, 8), "ema50": round(e50, 8), "ema200": round(e200, 8), "signal": signal, "strength": min(score, 100), "reasons": reasons, "candle": pattern, "market_structure": structure, "stop_loss": round(stop, 8), "take_profit": round(price + risk_distance * CONFIG["risk_reward"], 8), "risk_reward": CONFIG["risk_reward"], "candle_time": int(c[-1, 5])}


def rules(symbol: str) -> dict[str, float]:
    if symbol not in exchange_cache:
        info = request_binance("GET", "/v3/exchangeInfo", {"symbol": symbol})["symbols"][0]
        filters = {f["filterType"]: f for f in info["filters"]}
        lot = filters.get("LOT_SIZE", {})
        notion = filters.get("NOTIONAL", filters.get("MIN_NOTIONAL", {}))
        exchange_cache[symbol] = {"step": float(lot.get("stepSize", "0.000001")), "min_qty": float(lot.get("minQty", "0.000001")), "min_notional": float(notion.get("minNotional", "5"))}
    return exchange_cache[symbol]


def round_step(value: float, step: float) -> float:
    d, s = Decimal(str(value)), Decimal(str(step))
    return float((d / s).to_integral_value(rounding=ROUND_DOWN) * s)


def order_quantity(a: dict[str, Any]) -> float:
    r = rules(a["symbol"]); risk_cash = state["balance"] * CONFIG["risk_per_trade"]
    quantity = round_step(risk_cash / max(a["price"] - a["stop_loss"], a["price"] * 0.001), r["step"])
    quantity = max(quantity, r["min_qty"])
    if quantity * a["price"] < r["min_notional"]: raise RuntimeError(f"Order is below {r['min_notional']} USDT minimum")
    return quantity


def market_order(symbol: str, side: str, quantity: float) -> dict[str, Any]:
    return request_binance("POST", "/v3/order", {"symbol": symbol, "side": side, "type": "MARKET", "quantity": f"{quantity:.8f}", "newOrderRespType": "RESULT"}, signed=True)


def open_position(a: dict[str, Any]) -> None:
    with lock:
        if a["signal"] != "BUY" or len(state["open_positions"]) >= CONFIG["max_positions"] or any(x["symbol"] == a["symbol"] for x in state["open_positions"]): return
        quantity = order_quantity(a) if ENABLE_ORDERS else state["balance"] * CONFIG["risk_per_trade"] / max(a["price"] - a["stop_loss"], a["price"] * .001)
        exchange_order = market_order(a["symbol"], "BUY", quantity) if ENABLE_ORDERS else {"status": "SIGNAL_ONLY"}
        state["open_positions"].append({"symbol": a["symbol"], "side": "BUY", "entry_price": a["price"], "quantity": quantity, "stop_loss": a["stop_loss"], "take_profit": a["take_profit"], "order": exchange_order, "opened_at": utcnow()})


def close_position(position: dict[str, Any], exit_price: float, reason: str) -> None:
    quantity = position["quantity"]
    if ENABLE_ORDERS:
        # Sell only the quantity bought by this bot. Testnet Spot has no native short position here.
        market_order(position["symbol"], "SELL", round_step(quantity, rules(position["symbol"])["step"]))
    pnl = (exit_price - position["entry_price"]) * quantity
    state["balance"] += pnl; state["total_profit"] += pnl; state["total_trades"] += 1
    if pnl >= 0: state["winning_trades"] += 1
    else: state["losing_trades"] += 1; state["daily_loss"] += abs(pnl)
    state["closed_trades"].append({"symbol": position["symbol"], "pnl": round(pnl, 4), "status": "WIN" if pnl >= 0 else "LOSS", "reason": reason, "time": utcnow()})


def resolve_positions() -> None:
    with lock:
        remaining = []
        for position in state["open_positions"]:
            try: price = float(request_binance("GET", "/v3/ticker/price", {"symbol": position["symbol"]})["price"])
            except Exception: remaining.append(position); continue
            if price >= position["take_profit"]: close_position(position, position["take_profit"], "TAKE_PROFIT")
            elif price <= position["stop_loss"]: close_position(position, position["stop_loss"], "STOP_LOSS")
            else: remaining.append(position)
        state["open_positions"] = remaining


def loop() -> None:
    while True:
        try:
            if state["is_trading"] and state["daily_loss"] < CONFIG["initial_capital"] * CONFIG["max_daily_loss"]:
                for symbol in SYMBOLS:
                    a = analyze(symbol)
                    key = (symbol, TRADE_TIMEFRAME)
                    if last_processed.get(key) != a["candle_time"]:
                        last_processed[key] = a["candle_time"]; open_position(a)
                resolve_positions(); state["last_error"] = None; state["last_update"] = utcnow()
            time.sleep(CONFIG["poll_seconds"])
        except Exception as exc:
            state["last_error"] = str(exc); print("trading loop error:", exc); time.sleep(CONFIG["poll_seconds"])

threading.Thread(target=loop, daemon=True).start()

@app.get("/")
def index(): return render_template("index.html")

@app.get("/health")
def health(): return jsonify({"ok": True, "mode": state["mode"], "timeframes": TIMEFRAMES, "trade_timeframe": TRADE_TIMEFRAME})

@app.post("/api/trading/start")
def start(): state["is_trading"] = True; return jsonify({"is_trading": True, "mode": state["mode"], "orders_enabled": ENABLE_ORDERS})

@app.post("/api/trading/stop")
def stop(): state["is_trading"] = False; return jsonify({"is_trading": False})

@app.get("/api/stats")
def stats():
    with lock:
        total = state["total_trades"]
        return jsonify({**state, "open_positions": len(state["open_positions"]), "win_rate": round(state["winning_trades"] / total * 100, 2) if total else 0, "profit_percent": round((state["balance"] / state["initial_balance"] - 1) * 100, 2), "risk_per_trade": CONFIG["risk_per_trade"], "risk_reward": CONFIG["risk_reward"]})

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

@app.get("/api/timeframes")
def timeframes():
    result: dict[str, Any] = {"trade_timeframe": TRADE_TIMEFRAME, "supported": TIMEFRAMES, "data": {}}
    for interval in TIMEFRAMES:
        try: result["data"][interval] = analyze(SYMBOLS[0], interval)
        except Exception as exc: result["data"][interval] = {"error": str(exc)}
    return jsonify(result)

if __name__ == "__main__": app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
