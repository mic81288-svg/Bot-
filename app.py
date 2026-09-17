from datetime import datetime
import os, random, threading, time

import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template
from flask_cors import CORS

load_dotenv()
app = Flask(__name__)
CORS(app)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT"]
COINS = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "BNBUSDT": "binancecoin", "ADAUSDT": "cardano", "DOGEUSDT": "dogecoin"}
CONFIG = {
    "initial_capital": float(os.getenv("INITIAL_CAPITAL", "800")),
    "risk_per_trade": float(os.getenv("RISK_PER_TRADE", "0.02")),
    "risk_reward": float(os.getenv("RISK_REWARD_RATIO", "3")),
    "max_positions": int(os.getenv("MAX_OPEN_POSITIONS", "5")),
}
state = {"balance": CONFIG["initial_capital"], "is_trading": False, "total_trades": 0, "winning_trades": 0, "losing_trades": 0, "total_profit": 0.0, "daily_loss": 0.0, "open_positions": [], "closed_trades": [], "last_update": datetime.utcnow().isoformat()}
prices = {}
lock = threading.Lock()

def price(symbol):
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price", params={"ids": COINS[symbol], "vs_currencies": "usd"}, timeout=8)
        r.raise_for_status()
        value = float(r.json()[COINS[symbol]]["usd"])
        prices[symbol] = value
        return value
    except Exception:
        return prices.get(symbol, {"BTCUSDT": 50000, "ETHUSDT": 2500, "BNBUSDT": 500, "ADAUSDT": .5, "DOGEUSDT": .1}[symbol])

def analysis(symbol):
    current = price(symbol)
    values = [current * .98]
    for _ in range(299):
        values.append(values[-1] * (1 + random.uniform(-.01, .01)))
    p = np.array(values)
    ma20, ma50, ma200 = p[-20:].mean(), p[-50:].mean(), p[-200:].mean()
    d = np.diff(p[-15:]); gains = d[d > 0].sum() / 14; losses = -d[d < 0].sum() / 14
    rsi = 100.0 if losses == 0 else 100 - 100 / (1 + gains / losses)
    middle, deviation = p[-20:].mean(), p[-20:].std()
    signal, strength = "NEUTRAL", 0
    if ma20 > ma50 and rsi > 45 and current > middle:
        signal, strength = "BUY", min(100, int(rsi - 40 + 30))
    elif ma20 < ma50 and rsi < 55 and current < middle:
        signal, strength = "SELL", min(100, int(60 - rsi + 30))
    elif rsi > 75:
        signal, strength = "SELL", 75
    elif rsi < 25:
        signal, strength = "BUY", 75
    return {"symbol": symbol, "price": round(current, 8), "rsi": round(rsi, 1), "signal": signal, "strength": strength, "upper": middle + 2 * deviation, "middle": middle, "lower": middle - 2 * deviation}

def open_paper_position(a):
    if a["strength"] < 50 or len(state["open_positions"]) >= CONFIG["max_positions"]:
        return
    if any(p["symbol"] == a["symbol"] for p in state["open_positions"]):
        return
    entry = a["price"]; distance = entry * .005
    risk_money = state["balance"] * CONFIG["risk_per_trade"]
    quantity = risk_money / distance
    buy = a["signal"] == "BUY"
    state["open_positions"].append({"symbol": a["symbol"], "side": a["signal"], "entry_price": entry, "quantity": quantity, "stop_loss": entry - distance if buy else entry + distance, "take_profit": entry + distance * CONFIG["risk_reward"] if buy else entry - distance * CONFIG["risk_reward"], "opened_at": datetime.utcnow().isoformat()})

def resolve_positions():
    remaining = []
    for p in state["open_positions"]:
        now = price(p["symbol"]); buy = p["side"] == "BUY"
        hit_tp = now >= p["take_profit"] if buy else now <= p["take_profit"]
        hit_sl = now <= p["stop_loss"] if buy else now >= p["stop_loss"]
        if not (hit_tp or hit_sl):
            remaining.append(p); continue
        exit_price = p["take_profit"] if hit_tp else p["stop_loss"]
        pnl = (exit_price - p["entry_price"]) * p["quantity"] if buy else (p["entry_price"] - exit_price) * p["quantity"]
        state["balance"] += pnl; state["total_profit"] += pnl; state["total_trades"] += 1
        if pnl >= 0: state["winning_trades"] += 1
        else: state["losing_trades"] += 1; state["daily_loss"] += abs(pnl)
        state["closed_trades"].append({"symbol": p["symbol"], "side": p["side"], "pnl": round(pnl, 2), "status": "WIN" if pnl >= 0 else "LOSS", "time": datetime.utcnow().isoformat()})
    state["open_positions"] = remaining

def loop():
    while True:
        try:
            if state["is_trading"]:
                for symbol in SYMBOLS:
                    a = analysis(symbol)
                    if a["signal"] != "NEUTRAL": open_paper_position(a)
                resolve_positions()
                state["last_update"] = datetime.utcnow().isoformat()
            time.sleep(5)
        except Exception as exc:
            print("paper loop error:", exc)
            time.sleep(5)

threading.Thread(target=loop, daemon=True).start()

@app.get("/")
def index(): return render_template("index.html")
@app.post("/api/trading/start")
def start(): state["is_trading"] = True; return jsonify({"is_trading": True, "status": "Paper trading started"})
@app.post("/api/trading/stop")
def stop(): state["is_trading"] = False; return jsonify({"is_trading": False, "status": "Paper trading stopped"})
@app.get("/api/stats")
def stats():
    with lock:
        total = state["total_trades"]
        return jsonify({**state, "open_positions": len(state["open_positions"]), "win_rate": round(state["winning_trades"] / total * 100, 2) if total else 0, "initial_balance": CONFIG["initial_capital"], "risk_reward": CONFIG["risk_reward"]})
@app.get("/api/positions")
def positions(): return jsonify({"open_positions": state["open_positions"], "closed_trades": state["closed_trades"][-50:]})
@app.get("/api/analyze-all")
def analyze_all(): return jsonify({s: analysis(s) for s in SYMBOLS})

if __name__ == "__main__": app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
