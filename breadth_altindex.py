import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config (safe defaults)
# =========================

SMA_FAST = 50
SMA_SLOW = 200
RET_WINDOW = 30
HI_LO_WINDOW = 90

ALT_INDEX_START = 100.0
BTC_INDEX_START = 100.0


# =========================
# Small helpers
# =========================

def sma(s: pd.Series, n: int) -> pd.Series:
    # SMA_n(t) = mean of last n closes (trend filter)
    return s.rolling(n).mean()

def pct_returns(close: pd.Series) -> pd.Series:
    # r_t = P_t / P_{t-1} - 1
    return close.pct_change()

def load_manifest(data_dir: Path) -> dict:
    p = data_dir / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"manifest.json not found in {data_dir}")
    return json.loads(p.read_text(encoding="utf-8"))

def read_symbol_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    p = data_dir / filename
    df = pd.read_csv(p, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.sort_index()
    return df

def is_ratio_pair(sym: str) -> bool:
    # ETHBTC, SOLBTC, etc.
    return sym.endswith("BTC") and not sym.startswith("BTC")

def is_usdt_pair(sym: str) -> bool:
    return sym.endswith("USDT")

def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# =========================
# Core calculations
# =========================

def build_universe(data: dict) -> pd.DataFrame:
    """
    Returns wide dataframe of closes:
      index = dates
      columns = symbols
    """
    closes = {}
    for sym, df in data.items():
        if "close" in df.columns:
            closes[sym] = df["close"].astype(float)
    wide = pd.DataFrame(closes).sort_index()
    return wide

def breadth_timeseries(close_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Breadth signals computed across *USDT pairs only* (exclude BTCUSDT from breadth denominator).
    Breadth works best with 20–50+ symbols.
    """
    symbols = [c for c in close_wide.columns if is_usdt_pair(c) and c != "BTCUSDT"]
    if not symbols:
        raise ValueError("No USDT alt pairs found for breadth. Add more symbols like XRPUSDT, LINKUSDT, etc.")

    c = close_wide[symbols]

    s50 = c.apply(lambda x: sma(x, SMA_FAST))
    s200 = c.apply(lambda x: sma(x, SMA_SLOW))

    above50 = (c > s50)
    above200 = (c > s200)

    # 30D return breadth
    ret30 = c.pct_change(RET_WINDOW)
    pos30 = (ret30 > 0)

    # 90D highs/lows breadth
    roll_hi = c.rolling(HI_LO_WINDOW).max()
    roll_lo = c.rolling(HI_LO_WINDOW).min()
    new_high = (c >= roll_hi)  # exact touch
    new_low = (c <= roll_lo)

    out = pd.DataFrame(index=c.index)
    out["n_symbols"] = len(symbols)
    out["pct_above_sma50"] = above50.mean(axis=1)
    out["pct_above_sma200"] = above200.mean(axis=1)
    out["median_ret_30d"] = ret30.median(axis=1)
    out["pct_pos_ret_30d"] = pos30.mean(axis=1)
    out["new_highs_90d"] = new_high.sum(axis=1)
    out["new_lows_90d"] = new_low.sum(axis=1)

    return out.dropna(how="all")

def alt_index_vs_btc(close_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weight Alt Index vs BTC:
    - Universe: USDT alts only (exclude BTCUSDT)
    - Index: average of daily returns across alts
    - Compare to BTC index
    """
    alt_syms = [c for c in close_wide.columns if is_usdt_pair(c) and c != "BTCUSDT"]
    if "BTCUSDT" not in close_wide.columns:
        raise ValueError("BTCUSDT not found. Add BTCUSDT to SYMBOLS in the data script.")

    alt_close = close_wide[alt_syms]
    btc_close = close_wide["BTCUSDT"]

    alt_ret = alt_close.pct_change().replace([np.inf, -np.inf], np.nan)
    btc_ret = btc_close.pct_change().replace([np.inf, -np.inf], np.nan)

    # Equal-weight daily return (ignore NaNs)
    alt_eq_ret = alt_ret.mean(axis=1)

    # Build index levels
    alt_idx = (1 + alt_eq_ret.fillna(0)).cumprod() * ALT_INDEX_START
    btc_idx = (1 + btc_ret.fillna(0)).cumprod() * BTC_INDEX_START

    rel = alt_idx / btc_idx
    rel_log = np.log(rel.replace(0, np.nan))

    out = pd.DataFrame(index=close_wide.index)
    out["alt_index"] = alt_idx
    out["btc_index"] = btc_idx
    out["alt_over_btc"] = rel
    out["alt_over_btc_log"] = rel_log
    out["alt_universe_n"] = len(alt_syms)

    return out.dropna(how="all")

def write_summary_markdown(data_dir: Path, manifest: dict, breadth: pd.DataFrame, alt: pd.DataFrame):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_path = data_dir / f"breadth_altindex_report_{ts}.md"

    b = breadth.dropna().iloc[-1]
    a = alt.dropna().iloc[-1]

    def p(x, d=1):
        return "n/a" if pd.isna(x) else f"{x*100:.{d}f}%"

    lines = []
    lines.append("# Breadth + Alt Index Report")
    lines.append("")
    lines.append(f"- Generated (UTC): {ts}")
    lines.append(f"- Window (UTC): {manifest.get('start_utc')} → {manifest.get('end_utc')}")
    lines.append(f"- Universe size (alts): {int(a['alt_universe_n'])}")
    lines.append("")
    lines.append("## Breadth (market health)")
    lines.append(f"- % above SMA200: {p(b['pct_above_sma200'], 1)}")
    lines.append(f"- % above SMA50:  {p(b['pct_above_sma50'], 1)}")
    lines.append(f"- % positive 30D returns: {p(b['pct_pos_ret_30d'], 1)}")
    lines.append(f"- Median 30D return: {p(b['median_ret_30d'], 1)}")
    lines.append(f"- 90D new highs / lows: {int(b['new_highs_90d'])} / {int(b['new_lows_90d'])}")
    lines.append("")
    lines.append("## Alt Index vs BTC (risk-on gauge)")
    lines.append(f"- Alt Index: {a['alt_index']:.2f} | BTC Index: {a['btc_index']:.2f}")
    lines.append(f"- Alt/BTC strength (ratio): {a['alt_over_btc']:.4f}")
    lines.append("")
    lines.append("### How to read")
    lines.append("- Breadth rising = healthier participation (bull phase quality improves).")
    lines.append("- Breadth falling while BTC holds = narrow leadership (often fragile).")
    lines.append("- Alt/BTC rising = risk-on (alts leading). Falling = BTC dominance (risk-off).")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# =========================
# Charts
# =========================

def plot_breadth(breadth: pd.DataFrame, charts_dir: Path):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(breadth.index, breadth["pct_above_sma200"].values, label="% above SMA200")
    ax.plot(breadth.index, breadth["pct_above_sma50"].values, label="% above SMA50")
    ax.plot(breadth.index, breadth["pct_pos_ret_30d"].values, label="% positive 30D")

    ax.set_title("Market Breadth (alts)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Share (0..1)")
    ax.legend(loc="best")

    out = charts_dir / "breadth.png"
    save_fig(fig, out)
    return out

def plot_alt_vs_btc(alt: pd.DataFrame, charts_dir: Path):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(alt.index, alt["alt_over_btc"].values, label="Alt Index / BTC Index")

    ax.set_title("Alt Strength vs BTC (risk-on gauge)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Ratio")
    ax.legend(loc="best")

    out = charts_dir / "alt_strength_vs_btc.png"
    save_fig(fig, out)
    return out


# =========================
# Main
# =========================

def main():
    root = Path(__file__).resolve().parent
    data_dir = root / "market_data"
    charts_dir = data_dir / "charts"

    manifest = load_manifest(data_dir)
    sym_files = manifest["files"]["symbol_ohlcv"]

    # Load symbol data
    data = {}
    for sym, fname in sym_files.items():
        try:
            data[sym] = read_symbol_csv(data_dir, fname)
        except Exception as e:
            print(f"⚠️ Skipping {sym}: {e}")

    if not data:
        raise RuntimeError("No symbol CSVs found. Run your data script first.")

    close_wide = build_universe(data)

    # Breadth + Alt index
    breadth = breadth_timeseries(close_wide)
    alt = alt_index_vs_btc(close_wide)

    # Save CSV outputs
    breadth_path = data_dir / "breadth_timeseries.csv"
    alt_path = data_dir / "alt_index_timeseries.csv"
    breadth.to_csv(breadth_path)
    alt.to_csv(alt_path)

    # Charts
    b_png = plot_breadth(breadth.dropna(), charts_dir)
    a_png = plot_alt_vs_btc(alt.dropna(), charts_dir)

    # Markdown summary
    report_path = write_summary_markdown(data_dir, manifest, breadth, alt)

    print("\n✅ Breadth + Alt Index done")
    print(f"- Saved: {breadth_path.name}")
    print(f"- Saved: {alt_path.name}")
    print(f"- Chart: {b_png}")
    print(f"- Chart: {a_png}")
    print(f"- Report: {report_path.name}\n")


if __name__ == "__main__":
    main()