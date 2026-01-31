# 📊 Crypto Market Dashboard (Data → Report → Charts)

A simple 4-step pipeline that pulls crypto market data from Binance, computes regime + risk stats, writes an expert-style report, and generates charts.

---

## ✅ What it does

### Data script

* Fetches Binance OHLCV candles (daily by default)
* Computes:

  * Performance & risk: return, CAGR, vol, max drawdown, Sharpe/Sortino, Calmar, VaR/CVaR
  * Cycle snapshot: price vs SMA200, SMA50 vs SMA200, RSI, vol regime (percentile)
  * BTC relationships for alts: rolling corr/beta vs BTC (USDT pairs)

### Analyzer script

* Reads the saved CSVs (no API calls)
* Generates an expert-style Markdown report:

  * Executive summary
  * Long-term investor lens
  * Swing trader lens
  * Short-term / risk lens
  * Rankings (best trends, hottest risk, worst drawdowns)
  * Alt/BTC ratio dashboard (if you include ETHBTC/SOLBTC/etc.)

### Charts script

* Reads saved OHLCV CSVs
* Generates per-symbol charts:

  * Price + SMA50 + SMA200
  * Drawdown (underwater)
  * Rolling 30D annualized volatility (+ vol percentile note)

### Breadth + Alt Index script (cycle upgrade)

* Market breadth:

  * % of alts above SMA200 / SMA50
  * % positive 30D return
  * median 30D return
  * 90D new highs vs new lows
* Alt Index vs BTC (risk-on gauge):

  * equal-weight alt index vs BTC index
  * Alt/BTC relative strength chart

---

## 📦 Install

pip install requests pandas numpy matplotlib tabulate

OR

python -m pip install requests pandas numpy matplotlib tabulate

---

## 🏃 Run (terminal)

Run these from your project folder:

1. Build / refresh data
   python data_script.py

2. Generate the market report
   python analyze_market.py

3. Generate charts
   python plot_market.py

4. Breadth + Alt Index
   python breadth_altindex.py

If your system uses python3:

python3 data_script.py
python3 analyze_market.py
python3 plot_market.py
python3 breadth_altindex.py

---

## ⭐ Recommended SYMBOLS

Paste this into your data script:

SYMBOLS = [
"BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","TRXUSDT",
"LINKUSDT","LTCUSDT","BCHUSDT","XLMUSDT","XMRUSDT",
"AVAXUSDT","DOTUSDT","ATOMUSDT","NEARUSDT","ICPUSDT","FILUSDT","APTUSDT","SUIUSDT",
"ARBUSDT","OPUSDT",
"UNIUSDT","AAVEUSDT","MKRUSDT","LDOUSDT",
"INJUSDT","TIAUSDT","SEIUSDT",
"ETHBTC","SOLBTC","BNBBTC","XRPBTC","ADABTC","LINKBTC"
]

Tip: If Binance rejects a symbol in your region, remove it and rerun.

---

## 👀 How to read it (fast)

### 1) Start with BTCUSDT regime

* uptrend + low/mid vol = healthiest environment
* uptrend + high vol = strong but fragile (size down)
* transition/chop = whipsaw risk
* downtrend = defensive posture

### 2) Check ETHBTC / SOLBTC

* uptrend = alts gaining vs BTC (risk-on improving)
* downtrend = BTC dominance (risk-off)

### 3) Check breadth

* rising % above SMA200 + rising % positive 30D = broad, healthy bull participation
* falling breadth while BTC holds = narrow leadership (often fragile)

### 4) Check drawdown + CVaR

* drawdown charts show “pain + time underwater”
* CVaR 5% tells you how ugly bad days are on average

---

## ⚠️ Notes

* Not financial advice.
* Accurate for the metrics it computes (based on Binance OHLCV).
* “Cycle” here is price/regime-based; add breadth + alt index to make it much stronger.

---
