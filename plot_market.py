import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Matplotlib plotting config
# =========================
# - Keep defaults (no manual colors) for clean, consistent visuals.
# - One chart per figure to keep files readable and shareable.
FIGSIZE = (11, 6)
DPI = 160
GRID = True


# =========================
# Indicator parameters
# =========================
SMA_FAST = 50
SMA_SLOW = 200
VOL_WINDOW = 30
VOL_LOOKBACK = 252
TRADING_DAYS = 365  # crypto trades daily


# =========================
# IO helpers
# =========================

def load_manifest(data_dir: Path) -> dict:
    """
    Manifest created by your data script.
    Contains symbols + filenames so plots auto-adapt to SYMBOLS changes.
    """
    p = data_dir / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"manifest.json not found in: {data_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def read_symbol_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    """
    Reads OHLCV CSV exported by the data script.
    Expected columns: open, high, low, close, volume
    Index: datetime-like
    """
    p = data_dir / filename
    if not p.exists():
        raise FileNotFoundError(f"Symbol CSV not found: {p}")

    df = pd.read_csv(p, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.sort_index()
    return df


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_title(symbol: str) -> str:
    """
    Small UX improvement in chart titles.
    """
    if symbol.endswith("BTC") and not symbol.startswith("BTC"):
        return f"{symbol} (ratio vs BTC)"
    if symbol.endswith("USDT"):
        return f"{symbol} (quoted in USDT)"
    return symbol


# =========================
# Math helpers (indicators)
# =========================

def sma(series: pd.Series, n: int) -> pd.Series:
    """
    Simple Moving Average:
      SMA_n(t) = mean(price[t-n+1 : t])
    Trend filter commonly used for cycle/regime detection.
    """
    return series.rolling(n).mean()


def pct_returns(close: pd.Series) -> pd.Series:
    """
    Simple returns:
      r_t = close_t / close_{t-1} - 1
    """
    return close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def drawdown(close: pd.Series) -> pd.Series:
    """
    Drawdown (underwater series):
      eq_t   = Π (1 + r_t)
      peak_t = max(eq_0..eq_t)
      dd_t   = eq_t / peak_t - 1

    dd is 0 at peaks, negative during drawdowns.
    """
    r = pct_returns(close)
    if r.empty:
        return pd.Series(dtype=float)

    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1
    return dd


def rolling_vol_ann(close: pd.Series, window: int = VOL_WINDOW, trading_days: int = TRADING_DAYS) -> pd.Series:
    """
    Rolling annualized volatility:
      vol(t) = std(r_{t-window+1..t}) * sqrt(trading_days)
    """
    r = pct_returns(close)
    if r.empty:
        return pd.Series(dtype=float)

    vol = r.rolling(window).std() * np.sqrt(trading_days)
    return vol.dropna()


def vol_percentile(vol_series: pd.Series, lookback: int = VOL_LOOKBACK) -> float:
    """
    Percentile rank of latest vol vs recent history.
    Interpretation:
      - high percentile (~0.8-1.0): hot / unstable
      - low percentile  (~0.0-0.3): compressed
    Returns NaN if insufficient history.
    """
    v = vol_series.dropna()
    if len(v) < max(lookback, 60):
        return float("nan")

    hist = v.tail(lookback)
    latest = hist.iloc[-1]
    return float((hist <= latest).mean())


# =========================
# Plotting primitives
# =========================

def new_fig_ax(title: str, ylabel: str):
    """
    Standardized Matplotlib figure creation.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    if GRID:
        ax.grid(True, linewidth=0.6, alpha=0.5)
    return fig, ax


def save_fig(fig, out_path: Path):
    """
    Save and close to avoid memory growth on many symbols.
    """
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =========================
# Chart types
# =========================

def plot_price_sma(symbol: str, df: pd.DataFrame, out_dir: Path) -> Path | None:
    close = df.get("close", pd.Series(dtype=float)).dropna()
    if len(close) < 30:
        return None

    s50 = sma(close, SMA_FAST)
    s200 = sma(close, SMA_SLOW)

    fig, ax = new_fig_ax(
        title=f"{safe_title(symbol)} Price + SMAs",
        ylabel="Price"
    )

    ax.plot(close.index, close.values, label="Close")
    if s50.notna().any():
        ax.plot(s50.index, s50.values, label=f"SMA{SMA_FAST}")
    if s200.notna().any():
        ax.plot(s200.index, s200.values, label=f"SMA{SMA_SLOW}")

    ax.legend(loc="best")

    out_path = out_dir / f"{symbol}_price_sma.png"
    save_fig(fig, out_path)
    return out_path


def plot_drawdown(symbol: str, df: pd.DataFrame, out_dir: Path) -> Path | None:
    close = df.get("close", pd.Series(dtype=float)).dropna()
    if len(close) < 60:
        return None

    dd = drawdown(close)
    if dd.empty:
        return None

    fig, ax = new_fig_ax(
        title=f"{safe_title(symbol)} Drawdown (Underwater)",
        ylabel="Drawdown"
    )

    ax.plot(dd.index, dd.values, label="Drawdown")
    ax.axhline(0, linewidth=1)

    ax.legend(loc="best")

    out_path = out_dir / f"{symbol}_drawdown.png"
    save_fig(fig, out_path)
    return out_path


def plot_volatility(symbol: str, df: pd.DataFrame, out_dir: Path) -> Path | None:
    close = df.get("close", pd.Series(dtype=float)).dropna()
    if len(close) < 90:
        return None

    vol = rolling_vol_ann(close, window=VOL_WINDOW) * 100.0  # percent
    if len(vol) < 30:
        return None

    pct = vol_percentile(vol / 100.0, lookback=VOL_LOOKBACK)  # percentile uses raw vol
    # Note: we pass vol/100 to keep percentile logic on vol not percent. Same ranking either way.

    fig, ax = new_fig_ax(
        title=f"{safe_title(symbol)} {VOL_WINDOW}D Volatility Regime",
        ylabel="Annualized Vol (%)"
    )

    ax.plot(vol.index, vol.values, label=f"{VOL_WINDOW}D Vol (ann, %)")

    # Add small annotation text (no second axis)
    if not np.isnan(pct):
        ax.text(
            0.02, 0.95,
            f"Vol percentile (lookback {VOL_LOOKBACK}): {pct:.2f}",
            transform=ax.transAxes,
            va="top"
        )

    ax.legend(loc="best")

    out_path = out_dir / f"{symbol}_volatility.png"
    save_fig(fig, out_path)
    return out_path


# =========================
# Main
# =========================

def main():
    root = Path(__file__).resolve().parent
    data_dir = root / "market_data"
    charts_dir = data_dir / "charts"
    ensure_dir(charts_dir)

    manifest = load_manifest(data_dir)
    sym_files = manifest["files"]["symbol_ohlcv"]

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\nPlot run (UTC): {now_utc}")
    print(f"Reading from: {data_dir}")
    print(f"Saving to:    {charts_dir}\n")

    created = 0
    skipped = 0

    for sym, filename in sym_files.items():
        try:
            df = read_symbol_csv(data_dir, filename)

            p1 = plot_price_sma(sym, df, charts_dir)
            p2 = plot_drawdown(sym, df, charts_dir)
            p3 = plot_volatility(sym, df, charts_dir)

            made_any = any([p1, p2, p3])
            if made_any:
                created += 1
                print(f"✅ {sym}: charts saved")
            else:
                skipped += 1
                print(f"⚠️ {sym}: not enough data for charts")

        except Exception as e:
            skipped += 1
            print(f"⚠️ {sym}: skipped ({e})")

    print(f"\nDone.")
    print(f"- Symbols charted: {created}")
    print(f"- Symbols skipped: {skipped}")
    print(f"\nOpen charts folder:\n{charts_dir}\n")


if __name__ == "__main__":
    main()