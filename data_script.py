import time
import math
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# =========================
# Configuration
# =========================
BASES = [
    "https://api.binance.com",          # primary
    "https://data-api.binance.vision"   # market-data-only domain
]

SYMBOLS = [
    # Core majors (market direction)
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT",

    # Large caps / market structure
    "LINKUSDT","LTCUSDT","BCHUSDT","XLMUSDT","XMRUSDT",

    # L1 / smart contract basket (broad alt health)
    "AVAXUSDT","DOTUSDT","ATOMUSDT","NEARUSDT","ICPUSDT","FILUSDT","APTUSDT","SUIUSDT",

    # L2 / scaling (alt-beta + cycle)
    "ARBUSDT","OPUSDT",

    # BTC ratio pairs (risk-on vs risk-off confirmation)
    "ETHBTC","SOLBTC","BNBBTC","XRPBTC","ADABTC","LINKBTC"
]
INTERVAL = "1d"

# Use either:
# 1) explicit date range, or
# 2) rolling_days window (recommended for "where are we now?" dashboards)
USE_ROLLING_WINDOW = True
ROLLING_DAYS = 365  # e.g., 120/180/365

END = datetime.now(timezone.utc) # datetime(2026, 1, 31, 23, 59, 59, tzinfo=timezone.utc)
START = datetime(2023, 10, 1, tzinfo=timezone.utc)  # only used if USE_ROLLING_WINDOW=False

# If you want "today" automatically:
# END = datetime.now(timezone.utc)

# =========================
# Helpers
# =========================

@dataclass
class FetchConfig:
    timeout: int = 30
    max_retries: int = 5
    backoff_base: float = 0.8  # exponential backoff base seconds
    limit: int = 1000          # Binance max per request


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def interval_ms(interval: str) -> int:
    """
    Convert Binance interval string to milliseconds.
    We only need 1d here, but this makes pagination safe if you change it later.
    """
    mapping = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
        "1w": 604_800_000
    }
    if interval not in mapping:
        raise ValueError(f"Unsupported interval: {interval}")
    return mapping[interval]


# =========================
# Data Fetching (Improved)
# =========================

def fetch_klines_paginated(symbol: str, interval: str, start: datetime, end: datetime,
                          session: requests.Session, cfg: FetchConfig) -> pd.DataFrame:
    """
    Binance /api/v3/klines returns up to 1000 rows per call.
    Pagination logic:
      - request [startTime, endTime]
      - if you get 1000 rows, advance startTime to last_open_time + interval_ms
      - repeat until fewer than 1000 rows or last time >= end
    """
    start_ms = to_ms(start)
    end_ms = to_ms(end)
    step = interval_ms(interval)

    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ]

    all_rows = []
    last_err = None

    for base in BASES:
        url = f"{base}/api/v3/klines"
        cur = start_ms

        try:
            while cur <= end_ms:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cur,
                    "endTime": end_ms,
                    "limit": cfg.limit
                }

                # Retry w/ exponential backoff on transient failures
                for attempt in range(cfg.max_retries):
                    try:
                        r = session.get(url, params=params, timeout=cfg.timeout)
                        if r.status_code in (418, 429):
                            # 418/429: rate limit / IP ban risk. Back off hard.
                            sleep_s = cfg.backoff_base * (2 ** attempt) + np.random.rand() * 0.2
                            time.sleep(sleep_s)
                            continue

                        r.raise_for_status()
                        data = r.json()
                        break
                    except Exception as e:
                        last_err = e
                        sleep_s = cfg.backoff_base * (2 ** attempt) + np.random.rand() * 0.2
                        time.sleep(sleep_s)
                else:
                    raise RuntimeError(f"Retries exceeded for {symbol} @ {base}: {last_err}")

                if not data:
                    break

                all_rows.extend(data)

                # Pagination advance
                last_open = data[-1][0]
                if len(data) < cfg.limit:
                    break
                cur = last_open + step

            # Parse into DF
            df = pd.DataFrame(all_rows, columns=cols)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

            for c in ["open","high","low","close","volume"]:
                df[c] = df[c].astype(float)

            df = df.sort_values("open_time").drop_duplicates("open_time").set_index("open_time")
            return df[["open","high","low","close","volume"]]

        except Exception as e:
            last_err = e
            all_rows = []  # reset if base fails and we try next base

    raise RuntimeError(f"Failed to fetch {symbol}: {last_err}")


# =========================
# Indicators & Stats
# =========================

def log_returns(close: pd.Series) -> pd.Series:
    """
    Log returns:
      r_t = ln(P_t / P_{t-1})
    Useful because they add over time and handle compounding nicely.
    """
    return np.log(close / close.shift(1)).dropna()


def pct_returns(close: pd.Series) -> pd.Series:
    """Simple returns: (P_t / P_{t-1}) - 1"""
    return close.pct_change().dropna()


def max_drawdown(close: pd.Series) -> tuple[float, int, int]:
    """
    Max Drawdown (depth) from equity curve:
      eq_t = Π (1 + r_t)
      dd_t = eq_t / peak_t - 1
    Returns:
      (max_dd, dd_duration_days, time_to_recover_days_or_-1)
    """
    rets = pct_returns(close)
    eq = (1 + rets).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1

    max_dd = float(dd.min())

    # Drawdown duration: longest consecutive period below peak
    below = dd < 0
    # group consecutive runs
    grp = (below != below.shift(1)).cumsum()
    durations = below.groupby(grp).sum()
    dd_duration = int(durations.max()) if len(durations) else 0

    # Time to recover from max drawdown:
    # find date of max dd, then first future date equity >= prior peak
    t_min = dd.idxmin() if len(dd) else None
    ttr = -1
    if t_min is not None:
        peak_before = peak.loc[:t_min].iloc[-1]
        future = eq.loc[t_min:]
        rec = future[future >= peak_before]
        if len(rec):
            ttr = int((rec.index[0] - t_min).days)

    return max_dd, dd_duration, ttr


def annualized_return(close: pd.Series, trading_days: int = 365) -> float:
    """
    CAGR approximation using total return and elapsed days.
      CAGR = (P_end / P_start)^(trading_days / N_days) - 1
    """
    close = close.dropna()
    if len(close) < 2:
        return float("nan")
    n_days = (close.index[-1] - close.index[0]).days
    if n_days <= 0:
        return float("nan")
    return float((close.iloc[-1] / close.iloc[0]) ** (trading_days / n_days) - 1)


def sharpe_ratio(rets: pd.Series, trading_days: int = 365, rf: float = 0.0) -> float:
    """
    Sharpe (risk-adjusted return):
      Sharpe = (E[r] - rf) / std(r) * sqrt(trading_days)
    For crypto we often use rf ~ 0 in dashboards unless you want to plug a rate in.
    """
    if len(rets) < 10:
        return float("nan")
    mu = rets.mean() - rf / trading_days
    sd = rets.std()
    return float(mu / sd * math.sqrt(trading_days)) if sd != 0 else float("nan")


def sortino_ratio(rets: pd.Series, trading_days: int = 365, rf: float = 0.0) -> float:
    """
    Sortino focuses on downside volatility only:
      Sortino = (E[r] - rf) / std(r | r<0) * sqrt(trading_days)
    """
    if len(rets) < 10:
        return float("nan")
    downside = rets[rets < 0]
    dd = downside.std()
    mu = rets.mean() - rf / trading_days
    return float(mu / dd * math.sqrt(trading_days)) if dd and dd != 0 else float("nan")


def realized_vol(rets: pd.Series, trading_days: int = 365) -> float:
    """Annualized realized vol: std(r) * sqrt(trading_days)"""
    if len(rets) < 10:
        return float("nan")
    return float(rets.std() * math.sqrt(trading_days))


def var_cvar(rets: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """
    Historical Value-at-Risk and Conditional VaR (Expected Shortfall).
      VaR_alpha = quantile(rets, alpha)
      CVaR_alpha = mean(rets[rets <= VaR_alpha])
    Reported as negative numbers typically.
    """
    if len(rets) < 50:
        return float("nan"), float("nan")
    var = float(rets.quantile(alpha))
    cvar = float(rets[rets <= var].mean())
    return var, cvar


def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple moving average"""
    return series.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """
    RSI (momentum oscillator):
      RS = avg_gain / avg_loss
      RSI = 100 - 100/(1 + RS)
    Using Wilder-style smoothing approximation via ewm.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rolling_beta(asset_close: pd.Series, btc_close: pd.Series, window: int = 60) -> float:
    """
    Rolling beta (last value) vs BTC:
      beta = Cov(r_a, r_b) / Var(r_b)
    Uses simple returns.
    """
    a = pct_returns(asset_close)
    b = pct_returns(btc_close)
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    if len(a) < window + 5:
        return float("nan")
    cov = a.rolling(window).cov(b)
    var = b.rolling(window).var()
    beta_series = cov / var.replace(0, np.nan)
    return float(beta_series.dropna().iloc[-1]) if beta_series.dropna().any() else float("nan")


def rolling_corr(asset_close: pd.Series, btc_close: pd.Series, window: int = 60) -> float:
    """Rolling correlation (last value) vs BTC"""
    a = pct_returns(asset_close)
    b = pct_returns(btc_close)
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    if len(a) < window + 5:
        return float("nan")
    c = a.rolling(window).corr(b)
    return float(c.dropna().iloc[-1]) if c.dropna().any() else float("nan")


def vol_regime(rets: pd.Series, lookback: int = 252) -> tuple[float, float]:
    """
    Volatility regime via percentile:
      - compute rolling 30d vol
      - compare latest to distribution over 'lookback' window
    Returns:
      (vol_30d_ann, vol_percentile_0to1)
    """
    if len(rets) < 60:
        return float("nan"), float("nan")
    vol_30 = rets.rolling(30).std() * math.sqrt(365)
    latest = vol_30.dropna().iloc[-1] if vol_30.dropna().any() else float("nan")
    hist = vol_30.dropna().tail(lookback)
    if len(hist) < 50 or np.isnan(latest):
        return float(latest), float("nan")
    pct = float((hist <= latest).mean())
    return float(latest), pct


def trend_regime(close: pd.Series) -> dict:
    """
    Simple cycle-ish regime:
      - Trend: price relative to 200D SMA
      - Momentum: RSI(14)
      - Medium-term trend: 50D vs 200D (golden/death cross-ish)
    """
    c = close.dropna()
    out = {}
    if len(c) < 210:
        # still compute what we can
        out["sma50"] = float(sma(c, 50).iloc[-1]) if len(c) >= 50 else float("nan")
        out["sma200"] = float(sma(c, 200).iloc[-1]) if len(c) >= 200 else float("nan")
        out["price_above_200sma"] = float("nan")
        out["golden_cross"] = float("nan")
        out["rsi14"] = float(rsi(c, 14).iloc[-1]) if len(c) >= 20 else float("nan")
        return out

    sma50 = sma(c, 50).iloc[-1]
    sma200 = sma(c, 200).iloc[-1]
    px = c.iloc[-1]
    out["sma50"] = float(sma50)
    out["sma200"] = float(sma200)
    out["price_above_200sma"] = float(px / sma200 - 1)
    out["golden_cross"] = float(sma50 / sma200 - 1)  # >0 = 50 above 200
    out["rsi14"] = float(rsi(c, 14).iloc[-1])

    # 200D slope as a rough "cycle trend" proxy (positive = rising long trend)
    sma200_series = sma(c, 200).dropna()
    if len(sma200_series) >= 30:
        out["sma200_slope_30d_%"] = float(sma200_series.pct_change(30).iloc[-1] * 100)
    else:
        out["sma200_slope_30d_%"] = float("nan")

    return out


def perf_stats(close: pd.Series) -> dict:
    """
    Performance + risk metrics:
      - total return
      - CAGR
      - max drawdown depth + duration + time to recover
      - realized vol
      - Sharpe / Sortino / Calmar
      - tail risk: VaR/CVaR
      - best/worst day
      - skew/kurtosis (distribution shape)
    """
    close = close.dropna()
    rets = pct_returns(close)
    if len(rets) == 0:
        return {}

    total_return = float(close.iloc[-1] / close.iloc[0] - 1)
    cagr = annualized_return(close)
    max_dd, dd_dur, ttr = max_drawdown(close)

    vol = realized_vol(rets)
    sharpe = sharpe_ratio(rets)
    sortino = sortino_ratio(rets)

    # Calmar = CAGR / |MaxDD| (how much return you got per unit of drawdown pain)
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 and not np.isnan(cagr) else float("nan")

    var5, cvar5 = var_cvar(rets, 0.05)

    return {
        "start_close": float(close.iloc[0]),
        "end_close": float(close.iloc[-1]),
        "total_return_%": total_return * 100,
        "cagr_%": cagr * 100,
        "max_drawdown_%": max_dd * 100,
        "dd_duration_days": dd_dur,
        "time_to_recover_days": ttr,
        "realized_vol_%_ann": vol * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "var_5%_day_%": var5 * 100,
        "cvar_5%_day_%": cvar5 * 100,
        "best_day_%": float(rets.max() * 100),
        "worst_day_%": float(rets.min() * 100),
        "skew": float(rets.skew()),
        "kurtosis": float(rets.kurtosis()),
        "days": int(rets.shape[0])
    }


def cycle_snapshot(symbol: str, df: pd.DataFrame, btc_df: pd.DataFrame | None = None) -> dict:
    """
    A "where are we in the cycle?" snapshot:
      - trend regime (200D, 50/200, RSI)
      - vol regime (30D vol percentile)
      - rolling corr/beta vs BTC (for USDT pairs)
      - recent momentum (30/90D returns)
    """
    close = df["close"].dropna()
    rets = pct_returns(close)

    out = {"symbol": symbol}

    # Recent momentum (simple, intuitive)
    if len(close) >= 31:
        out["ret_30d_%"] = float(close.pct_change(30).iloc[-1] * 100)
    else:
        out["ret_30d_%"] = float("nan")
    if len(close) >= 91:
        out["ret_90d_%"] = float(close.pct_change(90).iloc[-1] * 100)
    else:
        out["ret_90d_%"] = float("nan")

    # Trend/momentum regime
    out.update(trend_regime(close))

    # Vol regime
    vol30, vol_pct = vol_regime(rets)
    out["vol_30d_%_ann"] = vol30 * 100 if not np.isnan(vol30) else float("nan")
    out["vol_pctile_lookback"] = vol_pct  # 0..1; high => "hot"/riskier regime

    # BTC-relative stats for USDT pairs
    if btc_df is not None and symbol.endswith("USDT") and symbol != "BTCUSDT":
        out["corr60_vs_BTC"] = rolling_corr(close, btc_df["close"], 60)
        out["beta60_vs_BTC"] = rolling_beta(close, btc_df["close"], 60)

    # Simple human-readable regime label
    # Rules:
    # - Trend up if price > 200SMA and 200SMA slope positive
    # - Risk high if vol percentile > 0.7
    pab = out.get("price_above_200sma", np.nan)
    slope = out.get("sma200_slope_30d_%", np.nan)
    vp = out.get("vol_pctile_lookback", np.nan)

    trend = "unknown"
    if not np.isnan(pab) and not np.isnan(slope):
        if pab > 0 and slope > 0:
            trend = "uptrend"
        elif pab < 0 and slope < 0:
            trend = "downtrend"
        else:
            trend = "transition/chop"

    risk = "unknown"
    if not np.isnan(vp):
        if vp >= 0.7:
            risk = "high-vol"
        elif vp <= 0.3:
            risk = "low-vol"
        else:
            risk = "mid-vol"

    out["regime"] = f"{trend} + {risk}"

    return out


# =========================
# Run
# =========================

def main():
    # Decide window
    if USE_ROLLING_WINDOW:
        start = END - timedelta(days=ROLLING_DAYS)
    else:
        start = START

    print(f"Window: {start.date()} → {END.date()} (UTC), interval={INTERVAL}")

    session = requests.Session()
    cfg = FetchConfig()

    # Fetch all symbols
    dfs = {}
    for s in SYMBOLS:
        dfs[s] = fetch_klines_paginated(s, INTERVAL, start, END, session, cfg)

    # Performance table
    perf_rows = []
    for s in SYMBOLS:
        st = perf_stats(dfs[s]["close"])
        st["symbol"] = s
        perf_rows.append(st)

    perf = pd.DataFrame(perf_rows).set_index("symbol").sort_index()

    # Snapshot table (cycle/regime)
    btc_df = dfs.get("BTCUSDT")
    snap_rows = []
    for s in SYMBOLS:
        snap_rows.append(cycle_snapshot(s, dfs[s], btc_df))
    snap = pd.DataFrame(snap_rows).set_index("symbol").sort_index()

    # Alt/BTC ratios are already "risk-on vs BTC" proxies.
    # A simple way to read them:
    # - If ETHBTC and SOLBTC are trending up (above 200D & positive slope), risk appetite is improving.
    # - If they are trending down, market is more BTC-led / defensive.
    # (This is *not* a guarantee, but it's a useful dashboard lens.)

    print("\n=== Performance & Risk ===")
    print(perf.round(4))

    print("\n=== Cycle / Regime Snapshot ===")
    cols = [
        "regime", "ret_30d_%", "ret_90d_%",
        "price_above_200sma", "golden_cross", "sma200_slope_30d_%",
        "rsi14", "vol_30d_%_ann", "vol_pctile_lookback",
        "corr60_vs_BTC", "beta60_vs_BTC"
    ]
    # Keep only columns that exist (in case history is short)
    cols = [c for c in cols if c in snap.columns]
    print(snap[cols].round(4))

    # =========================
    # Export to /market_data + manifest
    # =========================

    # Folder next to this script
    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = ROOT_DIR / "market_data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Timestamp for traceability
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Save summary tables
    perf_path = DATA_DIR / "crypto_perf_risk.csv"
    snap_path = DATA_DIR / "crypto_cycle_snapshot.csv"
    perf.to_csv(perf_path)
    snap.to_csv(snap_path)

    # Save per-symbol OHLCV
    symbol_files = {}
    for s, df in dfs.items():
        p = DATA_DIR / f"{s}_1d.csv"
        df.to_csv(p)
        symbol_files[s] = p.name

    # Write manifest so the analyzer knows what to load
    manifest = {
        "generated_at_utc": generated_at,
        "interval": INTERVAL,
        "use_rolling_window": USE_ROLLING_WINDOW,
        "rolling_days": ROLLING_DAYS if USE_ROLLING_WINDOW else None,
        "start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_utc": END.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbols": SYMBOLS,
        "files": {
            "perf": perf_path.name,
            "snap": snap_path.name,
            "symbol_ohlcv": symbol_files
        }
    }
    with open(DATA_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved market data to: {DATA_DIR}")
    print("Wrote manifest.json for the analyzer.")

if __name__ == "__main__":
    main()