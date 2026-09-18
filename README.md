# Binance Testnet SMC dashboard

This version is Railway-ready and uses Binance candles, EMA trend, RSI, candle reading, BOS/liquidity-sweep SMC confirmation, 10% configurable risk per trade, and a fixed 3:1 target. It supports configured analysis timeframes from 1 second through 4 hours; trading uses `TRADE_TIMEFRAME` (15m by default) so the bot does not overtrade on every timeframe.

## Deploy safely

1. Deploy this repository on Railway.
2. Add `BINANCE_API_KEY` and `BINANCE_API_SECRET` as Railway Variables only.
3. Keep `ENABLE_ORDERS=false` for the first test. This creates signal-only dashboard positions without exchange orders.
4. Verify `/health`, `/api/analyze-all`, and `/api/timeframes`.
5. Only on Binance Spot Testnet, after verification, set `ENABLE_ORDERS=true`.

This is Spot and long-only: it buys and later sells the same asset at the calculated stop or target. It does not short. Railway restarts reset in-memory dashboard state, so production persistence is still needed for a long-running account. A 10% risk setting is aggressive and no win rate or profit is guaranteed.
