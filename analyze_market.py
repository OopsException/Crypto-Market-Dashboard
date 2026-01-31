import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# =========================
# Utility helpers
# =========================

def safe_float(x):
    """Convert to float safely; return np.nan on failure."""
    try:
        return float(x)
    except Exception:
        return np.nan


def pct(x, digits=2):
    """Format a float like a percent string, handling NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}%"


def num(x, digits=2):
    """Format a float numeric string, handling NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def clamp01(x):
    """Clamp x into [0,1]."""
    if np.isnan(x):
        return np.nan
    return max(0.0, min(1.0, x))


# =========================
# Interpretation rules
# =========================
# These are intentionally simple & explainable.
#
# Regime label is already computed by your first script, but we still score it.
#
# Why these rules:
# - "price above 200SMA" + positive "200SMA slope" is a robust long-term trend proxy.
# - vol percentile tells us if risk is compressed or expanded (cycle temperature).
# - ETHBTC/SOLBTC regimes are a strong "risk-on vs BTC" proxy in crypto.


def trend_score(row):
    """
    Trend score (0..100)
    Components:
    - price_above_200sma: positive = bullish bias
    - sma200_slope_30d_%: positive = rising long trend
    - golden_cross: 50SMA above 200SMA supports trend alignment
    - rsi14: momentum confirmation (>50 supportive)
    """
    pab = safe_float(row.get("price_above_200sma", np.nan))  # expressed as ratio, not %
    slope = safe_float(row.get("sma200_slope_30d_%", np.nan))  # already in %
    gc = safe_float(row.get("golden_cross", np.nan))  # ratio
    rsi = safe_float(row.get("rsi14", np.nan))

    # Normalize:
    # - pab: map -20%..+20% into 0..1 (outside gets clipped)
    pab_n = clamp01((pab + 0.20) / 0.40) if not np.isnan(pab) else np.nan

    # - slope: map -10%..+10% into 0..1
    slope_n = clamp01((slope + 10) / 20) if not np.isnan(slope) else np.nan

    # - golden cross: map -10%..+10% into 0..1
    gc_n = clamp01((gc + 0.10) / 0.20) if not np.isnan(gc) else np.nan

    # - rsi: map 30..70 into 0..1
    rsi_n = clamp01((rsi - 30) / 40) if not np.isnan(rsi) else np.nan

    parts = [p for p in [pab_n, slope_n, gc_n, rsi_n] if not np.isnan(p)]
    if not parts:
        return np.nan
    return float(np.mean(parts) * 100)


def risk_heat(row):
    """
    Risk heat (0..100), higher = hotter / more unstable.
    Uses vol_pctile_lookback primarily.
    """
    vp = safe_float(row.get("vol_pctile_lookback", np.nan))
    if np.isnan(vp):
        return np.nan
    return float(clamp01(vp) * 100)


def posture_from_regime(regime: str):
    """
    Translate regime -> suggested posture.
    This is not a trade signal, it's a risk posture guide.
    """
    if not isinstance(regime, str):
        return "unknown"

    r = regime.lower()
    if "uptrend" in r and "low-vol" in r:
        return "risk-on (healthy trend): scale in on pullbacks, hold winners"
    if "uptrend" in r and "mid-vol" in r:
        return "risk-on (normal): trend-follow, keep risk controls"
    if "uptrend" in r and "high-vol" in r:
        return "risk-on but fragile: size down, expect violent swings"
    if "transition" in r or "chop" in r:
        return "range/transition: reduce size, be selective, expect whipsaws"
    if "downtrend" in r and "high-vol" in r:
        return "defensive: preserve capital, avoid leverage, focus on risk"
    if "downtrend" in r:
        return "defensive: rallies can fail, prioritize risk management"
    return "unknown"


# =========================
# Analyzer core
# =========================

def load_manifest(data_dir: Path) -> dict:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {data_dir}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tables(data_dir: Path, manifest: dict):
    perf_path = data_dir / manifest["files"]["perf"]
    snap_path = data_dir / manifest["files"]["snap"]

    perf = pd.read_csv(perf_path, index_col=0)
    snap = pd.read_csv(snap_path, index_col=0)

    # Ensure consistent index
    perf.index = perf.index.astype(str)
    snap.index = snap.index.astype(str)

    return perf, snap


def load_symbol_ohlcv(data_dir: Path, manifest: dict):
    """
    Load per-symbol OHLCV if needed later for deeper analysis.
    For now we load just to confirm availability and optionally compute recent structure.
    """
    sym_files = manifest["files"]["symbol_ohlcv"]
    ohlcv = {}
    for sym, filename in sym_files.items():
        p = data_dir / filename
        if p.exists():
            df = pd.read_csv(p, parse_dates=[0], index_col=0)
            # Expect columns open/high/low/close/volume
            ohlcv[sym] = df
    return ohlcv


def build_rankings(perf: pd.DataFrame, snap: pd.DataFrame):
    joined = snap.join(perf, how="left", rsuffix="_perf")

    # Compute scores
    joined["trend_score"] = joined.apply(trend_score, axis=1)
    joined["risk_heat"] = joined.apply(risk_heat, axis=1)

    # Convenience: mark ratio pairs vs USDT pairs
    joined["pair_type"] = np.where(joined.index.str.endswith("BTC"), "ratio_vs_btc", "usdt_pair")

    # Sort lists
    best_trend = joined.sort_values(["trend_score", "calmar"], ascending=False)
    hottest_risk = joined.sort_values(["risk_heat", "realized_vol_%_ann"], ascending=False)
    best_calmar = joined.sort_values(["calmar"], ascending=False)
    worst_dd = joined.sort_values(["max_drawdown_%"], ascending=True)  # more negative = worse

    return joined, best_trend, hottest_risk, best_calmar, worst_dd


def alt_strength_summary(joined: pd.DataFrame):
    """
    Use ETHBTC/SOLBTC/other *BTC ratio pairs* as risk appetite proxies.
    """
    ratios = joined[joined["pair_type"] == "ratio_vs_btc"].copy()
    if ratios.empty:
        return None, ratios

    # A simple "risk appetite" score:
    # - trend_score high + regime uptrend -> strong alt appetite
    # - ret_90d positive -> sustained
    ratios["risk_appetite_score"] = ratios["trend_score"].fillna(0) * 0.6 + ratios["ret_90d_%"].fillna(0) * 0.4

    ratios_sorted = ratios.sort_values("risk_appetite_score", ascending=False)

    # Aggregate view: if most ratios are uptrend, that’s risk-on.
    up_count = ratios["regime"].astype(str).str.contains("uptrend", case=False, na=False).sum()
    total = len(ratios)
    risk_on_ratio = up_count / total if total else np.nan

    summary = {
        "ratio_pairs_count": int(total),
        "ratio_pairs_uptrend": int(up_count),
        "risk_on_share": float(risk_on_ratio) if not np.isnan(risk_on_ratio) else np.nan,
    }
    return summary, ratios_sorted


def make_report(manifest, perf, snap, joined, best_trend, hottest_risk, best_calmar, worst_dd, ratio_summary, ratios_sorted):
    gen = manifest.get("generated_at_utc", "n/a")
    start = manifest.get("start_utc", "n/a")
    end = manifest.get("end_utc", "n/a")
    interval = manifest.get("interval", "n/a")
    symbols = manifest.get("symbols", [])

    # Executive reads: BTC first if present
    btc_row = joined.loc["BTCUSDT"] if "BTCUSDT" in joined.index else None

    lines = []
    lines.append("# Market Analysis Report")
    lines.append("")
    lines.append(f"- Generated (UTC): {gen}")
    lines.append(f"- Window (UTC): {start} → {end}")
    lines.append(f"- Interval: {interval}")
    lines.append(f"- Symbols: {', '.join(symbols)}")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")

    if btc_row is not None:
        btc_regime = str(btc_row.get("regime", "unknown"))
        btc_posture = posture_from_regime(btc_regime)
        btc_trend = num(btc_row.get("trend_score", np.nan), 1)
        btc_heat = num(btc_row.get("risk_heat", np.nan), 1)
        btc_90 = pct(btc_row.get("ret_90d_%", np.nan), 2)
        lines.append(f"- **BTC regime:** {btc_regime}  | posture: {btc_posture}")
        lines.append(f"- **BTC trend score / risk heat:** {btc_trend} / {btc_heat} (0–100)")
        lines.append(f"- **BTC 90D momentum:** {btc_90}")
    else:
        lines.append("- BTCUSDT not found; executive summary is based on available symbols.")

    if ratio_summary is not None:
        risk_on_share = ratio_summary["risk_on_share"]
        lines.append(f"- **Alt strength vs BTC (ratio pairs in uptrend):** {ratio_summary['ratio_pairs_uptrend']}/{ratio_summary['ratio_pairs_count']} "
                     f"({pct(risk_on_share*100 if not np.isnan(risk_on_share) else np.nan, 1)})")
    else:
        lines.append("- No *BTC ratio* pairs found (like ETHBTC). Add them to SYMBOLS for risk appetite signals.")

    lines.append("")
    lines.append("## Long-Term Investor Lens (Positioning)")
    lines.append("")
    lines.append("Use these as *risk posture* guidelines, not guarantees:")
    lines.append("- Prefer adding risk when BTC is **uptrend + low/mid vol** and major ratios (ETHBTC/SOLBTC) are also trending up.")
    lines.append("- Be defensive when BTC is **downtrend** or when risk heat is high across the board.")
    lines.append("- Favor assets with better **Calmar** (return per drawdown pain) and survivability (shorter drawdown duration).")
    lines.append("")

    lines.append("### Best “quality trend” (trend score + Calmar)")
    lines.append("")
    lines.append(_top_table(best_trend, cols=[
        "regime", "trend_score", "calmar", "ret_90d_%", "max_drawdown_%", "vol_30d_%_ann"
    ], n=5))

    lines.append("")
    lines.append("## Swing Trader Lens (2–12 week horizon)")
    lines.append("")
    lines.append("- Trade with the regime: trends are easier in uptrends; ranges punish breakout-chasing in chop.")
    lines.append("- Use **vol percentile** to adapt: high vol = smaller size + wider stops; low vol = watch for expansion.")
    lines.append("- Ratios (ETHBTC/SOLBTC) rising often aligns with better alt swings.")
    lines.append("")
    lines.append("### Strongest momentum (90D return)")
    lines.append("")
    top_mom = joined.sort_values("ret_90d_%", ascending=False)
    lines.append(_top_table(top_mom, cols=[
        "regime", "ret_30d_%", "ret_90d_%", "rsi14", "vol_pctile_lookback"
    ], n=5))

    lines.append("")
    lines.append("## Short-Term / Risk Manager Lens (1–14 days)")
    lines.append("")
    lines.append("- Watch tail-risk: **CVaR 5%** is your “bad-days average.” If it’s nasty, reduce leverage/size.")
    lines.append("- High **risk heat** implies unstable conditions: expect wicks, gaps, liquidation cascades.")
    lines.append("- If corr/beta vs BTC is high, diversification is limited: most alts will follow BTC.")
    lines.append("")
    lines.append("### Highest risk heat (vol percentile)")
    lines.append("")
    lines.append(_top_table(hottest_risk, cols=[
        "regime", "risk_heat", "vol_30d_%_ann", "cvar_5%_day_%", "max_drawdown_%"
    ], n=5))

    lines.append("")
    lines.append("## Drawdown Pain (Who hurts the most?)")
    lines.append("")
    lines.append(_top_table(worst_dd, cols=[
        "regime", "max_drawdown_%", "dd_duration_days", "time_to_recover_days", "calmar"
    ], n=5))

    if ratio_summary is not None and not ratios_sorted.empty:
        lines.append("")
        lines.append("## Alt/BTC Ratio Pairs (Risk Appetite Dashboard)")
        lines.append("")
        lines.append("Interpretation:")
        lines.append("- Ratio uptrends = alts gaining vs BTC (often risk-on).")
        lines.append("- Ratio downtrends = BTC dominance / defensive tape.")
        lines.append("")
        lines.append(_top_table(ratios_sorted, cols=[
            "regime", "trend_score", "ret_90d_%", "price_above_200sma", "sma200_slope_30d_%"
        ], n=min(10, len(ratios_sorted))))

    lines.append("")

    return "\n".join(lines)


def _top_table(df: pd.DataFrame, cols, n=5):
    """
    Create a compact markdown table for the report.
    """
    view = df.copy()
    view = view.loc[:, [c for c in cols if c in view.columns]].head(n)

    # Format selected columns
    for c in view.columns:
        if c.endswith("_%") or c in ("max_drawdown_%", "realized_vol_%_ann", "vol_30d_%_ann", "cvar_5%_day_%", "var_5%_day_%", "ret_30d_%", "ret_90d_%"):
            view[c] = view[c].apply(lambda x: pct(x, 2))
        elif c in ("trend_score", "risk_heat"):
            view[c] = view[c].apply(lambda x: num(x, 1))
        elif c in ("price_above_200sma", "golden_cross"):
            # these are ratios; present as %
            view[c] = view[c].apply(lambda x: pct(safe_float(x) * 100, 2))
        elif c in ("vol_pctile_lookback",):
            view[c] = view[c].apply(lambda x: num(safe_float(x), 2))
        elif c in ("rsi14", "sharpe", "sortino", "calmar"):
            view[c] = view[c].apply(lambda x: num(x, 2))

    view.insert(0, "symbol", view.index)
    return view.to_markdown(index=False)


def main():
    # Folder next to this script
    root = Path(__file__).resolve().parent
    data_dir = root / "market_data"

    manifest = load_manifest(data_dir)
    perf, snap = load_tables(data_dir, manifest)

    # Optional deeper access to OHLCV if you want later:
    _ohlcv = load_symbol_ohlcv(data_dir, manifest)

    joined, best_trend, hottest_risk, best_calmar, worst_dd = build_rankings(perf, snap)
    ratio_summary, ratios_sorted = alt_strength_summary(joined)

    report = make_report(
        manifest, perf, snap,
        joined, best_trend, hottest_risk, best_calmar, worst_dd,
        ratio_summary, ratios_sorted
    )

    # Save report
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_path = data_dir / f"market_report_{ts}.md"
    out_path.write_text(report, encoding="utf-8")

    # Print a condensed console version
    print("\n=== Market Report (condensed) ===\n")
    print("\n".join(report.splitlines()[:60]))
    print("\n... (full report saved) ...")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()