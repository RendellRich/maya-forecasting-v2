"""Copies the backtest results the v2 app needs into app/data (slim versions). Live files are read from live/."""
import os
import shutil
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(os.path.dirname(HERE), "app", "data")
os.makedirs(APP, exist_ok=True)
for lag in ["lag48", "lag24"]:
    src = os.path.join(HERE, "out", lag)
    for f in ["buyahead_metrics.csv", "cost_curve.csv", "coverage.csv", "risk_validation.csv", "risk_adjusted_buy.csv"]:
        shutil.copy(os.path.join(src, f), os.path.join(APP, f"{lag}_{f}"))
h = pd.read_parquet(os.path.join(HERE, "out", "lag48", "maya_v2_hourly.parquet"))
keep = ["period", "dem_actual", "dem_pace", "net_actual", "net_pace", "ren_actual", "ren_persist",
        "dem_P50", "dem_P75", "dem_P90", "dem_P95", "dem_P98", "net_P50", "net_P75", "net_P90", "net_P95", "net_P98",
        "ren_P05", "ren_P10", "ren_P50", "dem_pace_P90", "dem_pace_P95", "dem_pace_P98", "net_pace_P90", "net_pace_P95", "net_pace_P98"]
h[keep].astype({c: "float32" for c in keep if c != "period"}).to_parquet(os.path.join(APP, "backtest_hourly.parquet"))
d = pd.read_csv(os.path.join(HERE, "out", "lag48", "daily_risk.csv"), index_col=0)
d[["risk_v2", "tier_v2", "label", "short_hours", "max_short_vs_P95", "max_abs_err_P50"]].dropna(subset=["risk_v2"]).to_csv(
    os.path.join(APP, "backtest_daily_risk.csv"))
print("app data written to", APP, sorted(os.listdir(APP)))
