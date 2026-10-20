"""
Maya v2 step 2: rolling calibration of the quantile forecasts, and buy-ahead scoring against PACE.

Rolling adaptive calibration (adaptive conformal inference, walk-forward, daily steps):
  For target day d and level q, take every hour in days d-WINDOW-1 .. d-2 (all actuals known when the forecast
  for d is made on d-1), compute score = actual - raw_level, and add the q-th quantile of those scores.
  For renewable the levels are low (P02..P25) and the same formula pulls them down to "output you can count on".

Comparisons
  * PACE as published (its own day-ahead forecast)
  * PACE + cushion: PACE given the same rolling calibration, so both sides get a fair safety margin
  * Maya P50 (best guess) and Maya buy-ahead levels P75..P98

Cost view: if buying short costs K times more per MWh than buying a surplus, the cheapest level to buy is
q* = K / (K + 1)  (K = 9 -> P90, K = 19 -> P95).  For each K both forecasts are scored at q*.

Outputs: out/maya_v2_hourly.parquet + .csv.gz, out/buyahead_metrics.csv, out/cost_curve.csv, out/coverage.csv
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_SFX = "" if os.environ.get("MAYA_FEATURES", "v2") == "v2" else "_" + os.environ["MAYA_FEATURES"]
OUT = os.path.join(HERE, "out", f"lag{os.environ.get('MAYA_MIN_LAG', '48')}{_SFX}")
WINDOW, LAG, MIN_DAYS = 60, 2, 14
QD = [0.50, 0.75, 0.90, 0.95, 0.98]
QR = [0.02, 0.05, 0.10, 0.25, 0.50]
SCORE_FROM = "2024-03-06"          # same start as the v1 metrics (lags + calibration warm-up)


def qn(q):
    return f"P{int(round(q * 100)):02d}"


raw = pd.read_parquet(os.path.join(OUT, "quantile_forecasts_raw.parquet"))
day = raw.index.normalize()
days = np.array(sorted(day.unique()))


GAMMA = 0.02      # adaptive step: how fast the level reacts when recent days miss more (or less) than the target


def calibrate(base, actual, q, upper=True, window=WINDOW, gamma=GAMMA):
    """Walk-forward adaptive calibration (adaptive conformal inference, daily steps).
    Shift for day d = the lvl-th quantile of (actual - base) over the last `window` finished days (ending d-LAG).
    lvl starts at q and is nudged after each finished day: up if that day missed more often than the target,
    down if less. Upper levels (demand, net load) miss when actual > forecast; lower levels (renewable) when
    actual < forecast. Only information available on the morning of d-1 is used."""
    score = actual - base
    sc_d = {d: s.dropna().values for d, s in score.groupby(day)}
    target = (1 - q) if upper else q
    lvl, shifts = q, {}
    for i, d in enumerate(days):
        if i - LAG >= 0 and days[i - LAG] in shifts:
            s = sc_d.get(days[i - LAG], np.array([]))
            if len(s):
                miss = (s > shifts[days[i - LAG]]).mean() if upper else (s < shifts[days[i - LAG]]).mean()
                lvl = float(np.clip(lvl + (gamma if upper else -gamma) * (miss - target), 0.001, 0.999))
        hist = [sc_d[days[j]] for j in range(max(0, i - window - LAG + 1), i - LAG + 1) if len(sc_d.get(days[j], []))]
        shifts[d] = np.quantile(np.concatenate(hist), lvl) if len(hist) >= MIN_DAYS else 0.0
    return base + pd.Series(day.map(shifts).values, index=base.index).astype(float)


cal = pd.DataFrame(index=raw.index)
cal["period"] = raw["period"]
cal["dem_actual"], cal["dem_pace"], cal["ren_actual"] = raw["dem_actual"], raw["dem_pace"], raw["ren_actual"]
cal["net_actual"] = cal["dem_actual"] - cal["ren_actual"]
cal["ren_persist"] = raw["ren_actual"].shift(24)
cal["net_pace"] = cal["dem_pace"] - cal["ren_persist"]          # v1 net-load baseline

for q in QD:
    cal[f"dem_{qn(q)}"] = calibrate(raw[f"dem_raw_{qn(q)}"], cal["dem_actual"], q)
for q in QR:
    cal[f"ren_{qn(q)}"] = calibrate(raw[f"ren_raw_{qn(q)}"], cal["ren_actual"], q, upper=False).clip(lower=0)
for q in QD:
    net_raw = raw[f"dem_raw_{qn(q)}"] - raw[f"ren_raw_{qn(round(1 - q, 2))}"]
    cal[f"net_{qn(q)}"] = calibrate(net_raw, cal["net_actual"], q)
# keep levels in order after calibration
for pre, qs in [("dem", QD), ("net", QD)]:
    cols = [f"{pre}_{qn(q)}" for q in qs]
    cal[cols] = np.maximum.accumulate(cal[cols].values, axis=1)
cols = [f"ren_{qn(q)}" for q in QR]
cal[cols] = np.maximum.accumulate(cal[cols].values, axis=1)

# PACE + cushion (same treatment), at each upper level
for q in QD[1:]:
    cal[f"dem_pace_{qn(q)}"] = calibrate(cal["dem_pace"], cal["dem_actual"], q)
    cal[f"net_pace_{qn(q)}"] = calibrate(cal["net_pace"], cal["net_actual"], q)

# carry diagnostics for the risk score
for c in ["dem_raw_P50", "dem_gbm_P50", "dem_lstm_P50", "dem_raw_P95", "ren_raw_P50", "ren_raw_P05",
          "temp_fc", "ghi_fc", "cloud_fc", "temp_lag_actual", "dem_lag_min", "missing_grid", "missing_weather"]:
    cal[c] = raw[c]
cal.index.name = "timestamp_utc"
cal.to_parquet(os.path.join(OUT, "maya_v2_hourly.parquet"))
cal.round(3).reset_index().to_csv(os.path.join(OUT, "maya_v2_hourly.csv.gz"), index=False)


# ------------------------------- scoring -----------------------------------------------------------
def score(actual, f, upper=True):
    k = pd.concat([actual, f], axis=1).dropna()
    a, p = k.iloc[:, 0], k.iloc[:, 1]
    short = (a - p).clip(lower=0) if upper else (p - a).clip(lower=0)      # renewable: short = counted on more than came
    surplus = (p - a).clip(lower=0) if upper else (a - p).clip(lower=0)
    return dict(hours=len(k), under_pct=round((short > 0).mean() * 100, 1),
                avg_short_MWh=round(short.mean(), 1), avg_surplus_MWh=round(surplus.mean(), 1),
                worst_short_MWh=round(short.max(), 0), p99_short_MWh=round(short.quantile(0.99), 0),
                MAE=round((a - p).abs().mean(), 1))


sc = cal[cal.index >= SCORE_FROM]
rows = []
for per in ["A backtest 2024-25", "B forward test 2026", "All"]:
    o = sc if per == "All" else sc[sc.period == per]
    for series, act, cands in [
        ("Demand", "dem_actual", [("PACE", "dem_pace")] + [(f"PACE + cushion {qn(q)}", f"dem_pace_{qn(q)}") for q in QD[1:]]
         + [(f"Maya {qn(q)}", f"dem_{qn(q)}") for q in QD]),
        ("Net load", "net_actual", [("PACE - persistence", "net_pace")] + [(f"PACE + cushion {qn(q)}", f"net_pace_{qn(q)}") for q in QD[1:]]
         + [(f"Maya {qn(q)}", f"net_{qn(q)}") for q in QD]),
        ("Renewable", "ren_actual", [("Persistence", "ren_persist")] + [(f"Maya {qn(q)}", f"ren_{qn(q)}") for q in QR])]:
        for name, col in cands:
            rows.append(dict(period=per, series=series, forecast=name, **score(o[act], o[col], upper=series != "Renewable")))
met = pd.DataFrame(rows)
met.to_csv(os.path.join(OUT, "buyahead_metrics.csv"), index=False)

# coverage check: is P95 really under ~5% of the time?
cov = met[met.forecast.str.startswith("Maya")][["period", "series", "forecast", "under_pct"]].copy()
lvl = cov.forecast.str.extract(r"P(\d+)")[0].astype(int)
cov["target_under_pct"] = np.where(cov.series == "Renewable", lvl, 100 - lvl)
cov.to_csv(os.path.join(OUT, "coverage.csv"), index=False)

# cost curve: short costs K x surplus. Cheapest level q* = K/(K+1); both sides calibrated at q*.
cost_rows = []
for K in [1, 2, 3, 5, 9, 19, 49]:
    q = K / (K + 1)
    for series, act, maya_raw, pace in [("Demand", "dem_actual", "dem_raw", "dem_pace"), ("Net load", "net_actual", None, "net_pace")]:
        if series == "Demand":
            base = raw[f"dem_raw_{qn(min(QD, key=lambda x: abs(x - q)))}"]
        else:
            qq = min(QD, key=lambda x: abs(x - q))
            base = raw[f"dem_raw_{qn(qq)}"] - raw[f"ren_raw_{qn(round(1 - qq, 2))}"]
        m = calibrate(base, cal[act], q)
        p = calibrate(cal[pace], cal[act], q)
        for per in ["A backtest 2024-25", "B forward test 2026", "All"]:
            msk = (cal.index >= SCORE_FROM) & ((cal.period == per) if per != "All" else True)
            k = pd.concat([cal.loc[msk, act], m[msk], p[msk], cal.loc[msk, pace]], axis=1).dropna()
            a = k.iloc[:, 0]
            def cost(f):
                return (K * (a - f).clip(lower=0) + (f - a).clip(lower=0)).mean()
            cm, cp, c0 = cost(k.iloc[:, 1]), cost(k.iloc[:, 2]), cost(k.iloc[:, 3])
            cost_rows.append(dict(K=K, level=f"{q:.3f}", period=per, series=series, hours=len(k),
                                  maya_cost=round(cm, 1), pace_cushion_cost=round(cp, 1), pace_raw_cost=round(c0, 1),
                                  saving_vs_pace_cushion_pct=round((cp - cm) / cp * 100, 1),
                                  saving_vs_pace_raw_pct=round((c0 - cm) / c0 * 100, 1),
                                  maya_under_pct=round(((a - k.iloc[:, 1]) > 0).mean() * 100, 1),
                                  pace_cushion_under_pct=round(((a - k.iloc[:, 2]) > 0).mean() * 100, 1)))
pd.DataFrame(cost_rows).to_csv(os.path.join(OUT, "cost_curve.csv"), index=False)

pd.set_option("display.width", 250)
print(met[met.period != "All"].to_string(index=False))
print()
print(pd.DataFrame(cost_rows).query("period != 'All'").to_string(index=False))
