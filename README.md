# Binance Testnet SMC bot

The bot now uses Binance candlesticks instead of CoinGecko/random prices. It includes EMA trend filtering, candle patterns, market structure/BOS, liquidity-sweep confirmation, position sizing with 10% configured risk, and a fixed 3:1 reward-to-risk target.

## Railway variables

Copy `.env.example` into Railway Variables and replace the Binance Testnet credentials. Never commit keys or send them in chat.

Start with `ENABLE_ORDERS=false`. This runs signal-only mode. After confirming the dashboard and Testnet account are correct, set `ENABLE_ORDERS=true` to enable authenticated Binance Spot Testnet market buys. The bot is long-only on Spot; it does not short-sell.

## Deploy

Railway can deploy this repository with the included `Procfile`. Use one worker because the trading loop is an in-process background thread.

A 10% risk setting is aggressive and is configurable through `RISK_PER_TRADE`; it does not guarantee a 10% loss or any win rate. Testnet results are not a promise of live performance.
