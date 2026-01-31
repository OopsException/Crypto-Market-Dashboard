import json
from openai import OpenAI
import pandas as pd
from pathlib import Path
from datetime import datetime

# =========================
# Create OpenAI Client
# =========================
client = OpenAI(api_key="")  # <- put your real key

# =========================
# Helper functions
# =========================

def load_manifest(data_dir: Path) -> dict:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found at {data_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))

def load_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    file_path = data_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    return pd.read_csv(file_path, index_col=0)

def get_last_value(df: pd.DataFrame, column: str):
    return df[column].dropna().iloc[-1]

def get_last_values_for_metrics(data_dir: Path) -> dict:
    perf = load_csv(data_dir, "crypto_perf_risk.csv")
    snap = load_csv(data_dir, "crypto_cycle_snapshot.csv")
    breadth = load_csv(data_dir, "breadth_timeseries.csv")
    alt_index = load_csv(data_dir, "alt_index_timeseries.csv")

    metrics = {
        "btc_regime": get_last_value(snap, "regime"),
        "btc_price_above_200sma": get_last_value(snap, "price_above_200sma"),
        "btc_sma200_slope": get_last_value(snap, "sma200_slope_30d_%"),
        "btc_rsi": get_last_value(snap, "rsi14"),
        "btc_vol_percentile": get_last_value(snap, "vol_pctile_lookback"),
        "btc_drawdown": get_last_value(perf, "max_drawdown_%"),
        "alt_index": get_last_value(alt_index, "alt_index"),
        "alt_btc_strength": get_last_value(alt_index, "alt_over_btc"),
        "breadth_above_sma200": get_last_value(breadth, "pct_above_sma200"),
        "breadth_new_highs": get_last_value(breadth, "new_highs_90d"),
        "breadth_new_lows": get_last_value(breadth, "new_lows_90d"),
    }
    return metrics

# =========================
# AI Call Function (Updated for new API)
# =========================

def call_openai_analysis(metrics: dict) -> str:
    prompt = f"""
    You are an expert market analyst with deep understanding of cryptocurrency and macroeconomics. 
    Using the following data from the market dashboard, give a high-level summary of the current market state, focusing on:
    - Market posture (risk-on / neutral / risk-off)
    - Cycle stage (early bull, mid bull, late bull, transition, chop, bear)
    - Key drivers (based on metrics)
    - Key risks (based on metrics)
    - Actionable plays for investors (long-term, swing, short-term)
    - Any invalidation triggers

    Market Metrics:
    - BTC Regime: {metrics['btc_regime']}
    - BTC Price above SMA200: {metrics['btc_price_above_200sma']}
    - BTC SMA200 slope: {metrics['btc_sma200_slope']}
    - BTC RSI: {metrics['btc_rsi']}
    - BTC Volatility Percentile: {metrics['btc_vol_percentile']}
    - BTC Max Drawdown: {metrics['btc_drawdown']}
    - Alt Index: {metrics['alt_index']}
    - Alt/BTC Strength: {metrics['alt_btc_strength']}
    - Breadth % Above SMA200: {metrics['breadth_above_sma200']}
    - Breadth New Highs: {metrics['breadth_new_highs']}
    - Breadth New Lows: {metrics['breadth_new_lows']}
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert market analyst with deep understanding of cryptocurrency and macroeconomics."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=1000
    )

    return response.choices[0].message.content.strip()

# =========================
# Final Report Generation
# =========================

def generate_report(data_dir: Path) -> None:
    metrics = get_last_values_for_metrics(data_dir)
    ai_summary = call_openai_analysis(metrics)

    report_path = data_dir / f"ai_market_summary_{datetime.now().strftime('%Y%m%d_%H%M%SZ')}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(ai_summary)

    print(f"\nAI-powered report saved to: {report_path}\n")

def main():
    data_dir = Path(__file__).resolve().parent / "market_data"
    generate_report(data_dir)

if __name__ == "__main__":
    main()