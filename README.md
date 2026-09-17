# Paper Trading Bot

Safe paper-trading dashboard. It uses CoinGecko prices and simulated history and does **not** connect to Binance or place orders.

Railway start command: `gunicorn app:app`

Variables: `INITIAL_CAPITAL=800`, `RISK_PER_TRADE=0.02`, `RISK_REWARD_RATIO=3`, `MAX_OPEN_POSITIONS=5`.
