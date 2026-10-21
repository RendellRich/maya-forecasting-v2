"""
Maya v2 step 3: daily risk score (Green / Yellow / Red) for how much to trust tomorrow's forecast.

Question it answers: "How likely is it that tomorrow, at some hour, demand comes in above our P95 buy?"
Label (per day): 1 if any hour's actual demand exceeded the calibrated Maya P95.

Inputs per day (all known on the morning of the day before):
  width        mean P95 - P50 range from the quantile models
  disagree     mean |GBM P50 - LSTM P50|            (the two halves of the blend disagree)
  (PACE forecast gap is computed for reference but NOT used: EIA posts PACE's day-ahead forecast only
   shortly before the day starts, after a morning buy would be placed.)
  temp_max / temp_min / temp_swing (forecast vs. the lag hour), cloud cover
  recent_err   Maya P50 mean absolute error over the last 7 finished days
  calendar     weekday, month, US federal holiday, day after holiday
  data flags   share of hours with missing grid or weather data in the lag window

Two versions
  v1  width only (no training needed)
  v2  small gradient boosting classifier on all the inputs, trained walk-forward: each month is scored by a
      model trained only on earlier days (first scored month: Jan 2025).

Tiers: thresholds set from the training days' predicted risk: Green = lowest 60%, Yellow = next 30%, Red = top 10%.

Decision test: "risk-adjusted buy" = buy P95 on Green days, step up to P98 on Yellow and Red days.
(Tested alternatives: P98 on Red only barely changes anything; P90 on Green days saves surplus but adds short hours.)
Compared with a flat P95 every day: fewer short hours and a smaller worst-hour shortfall at about the same cost?

Outputs: out/daily_risk.csv, out/risk_validation.csv, models/risk_classifier.joblib
"""
import os
import numpy as np
import pandas as pd
import joblib
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
LAGDIR = f"lag{os.environ.get('MAYA_MIN_LAG', '48')}" + ("" if os.environ.get("MAYA_FEATURES", "v2") == "v2" else "_" + os.environ["MAYA_FEATURES"]) + ("" if os.environ.get("MAYA_KNOWN_LEAD", "2") == "2" else "_k" + os.environ["MAYA_KNOWN_LEAD"])
OUT, MOD = os.path.join(HERE, "out", LAGDIR), os.path.join(HERE, "models", LAGDIR)
h = pd.read_parquet(os.path.join(OUT, "maya_v2_hourly.parquet"))
h = h[h.index >= "2024-03-06"]

# Days run in Mountain Prevailing Time? Keep UTC days, matching how the forecasts are made and scored.
h["day"] = h.index.normalize()
h["short95"] = (h.dem_actual > h.dem_P95).astype(float).where(h.dem_actual.notna() & h.dem_P95.notna())
h["err50"] = (h.dem_actual - h.dem_P50).abs()
g = h.groupby("day")
d = pd.DataFrame({
    "width": (h.dem_raw_P95 - h.dem_raw_P50).groupby(h.day).mean(),
    "disagree": (h.dem_gbm_P50 - h.dem_lstm_P50).abs().groupby(h.day).mean(),
    "pace_gap": (h.dem_raw_P50 - h.dem_pace).abs().groupby(h.day).mean(),
    "pace_minus_maya": (h.dem_pace - h.dem_raw_P50).groupby(h.day).mean(),
    "temp_max": g.temp_fc.max(), "temp_min": g.temp_fc.min(),
    "temp_swing": (h.temp_fc - h.temp_lag_actual).abs().groupby(h.day).mean(),
    "cloud": g.cloud_fc.mean(),
    "data_flags": (h.missing_grid.fillna(0) + h.missing_weather.fillna(0)).groupby(h.day).mean(),
    "hours": g.dem_actual.count(),
    "label": g.short95.max(),
    "short_hours": g.short95.sum(),
    "max_short_vs_P95": (h.dem_actual - h.dem_P95).clip(lower=0).groupby(h.day).max(),
    "max_abs_err_P50": g.err50.max(),
})
daily_mae = g.err50.mean()
d["recent_err"] = daily_mae.shift(2).rolling(7, min_periods=3).mean()       # last 7 finished days, known at forecast time
hol = USFederalHolidayCalendar().holidays(d.index.min(), d.index.max() + pd.Timedelta(days=1))
d["weekday"] = d.index.dayofweek
d["month"] = d.index.month
d["holiday"] = d.index.isin(hol).astype(int)
d["after_holiday"] = d.index.isin(hol + pd.Timedelta(days=1)).astype(int)
d = d[(d.hours >= 20) & d.label.notna() & d.width.notna()].copy()                # skip partial days

FEATS = ["width", "disagree", "temp_max", "temp_min", "temp_swing", "cloud",
         "recent_err", "data_flags", "weekday", "month", "holiday", "after_holiday"]
CLF = dict(max_iter=150, max_depth=3, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0, random_state=42)


def tiers(score, ref):
    lo, hi = np.nanquantile(ref, 0.60), np.nanquantile(ref, 0.90)
    return np.where(score >= hi, "Red", np.where(score >= lo, "Yellow", "Green")), lo, hi


# v2 walk-forward by month
d["risk_v2"] = np.nan
d["tier_v2"] = None
months = pd.period_range("2025-01", d.index.max().to_period("M"), freq="M")
for m in months:
    te = d.index.to_period("M") == m
    tr = d.index < m.start_time - pd.Timedelta(days=2)           # only days whose actuals were known
    if te.sum() == 0:
        continue
    clf = HistGradientBoostingClassifier(**CLF).fit(d.loc[tr, FEATS], d.loc[tr, "label"])
    d.loc[te, "risk_v2"] = clf.predict_proba(d.loc[te, FEATS])[:, 1]
    t, lo, hi = tiers(d.loc[te, "risk_v2"].values, clf.predict_proba(d.loc[tr, FEATS])[:, 1])
    d.loc[te, "tier_v2"] = t

# v1 width-only, tiers from all earlier days (expanding), same 60/90 split
d["tier_v1"] = None
for m in months:
    te = d.index.to_period("M") == m
    tr = d.index < m.start_time - pd.Timedelta(days=2)
    t, _, _ = tiers(d.loc[te, "width"].values, d.loc[tr, "width"].values)
    d.loc[te, "tier_v1"] = t

# production classifier: all labelled days, saved for the live pipeline and the app
prod = HistGradientBoostingClassifier(**CLF).fit(d[FEATS], d["label"])
p_all = prod.predict_proba(d[FEATS])[:, 1]
joblib.dump({"model": prod, "features": FEATS, "green_max": float(np.quantile(p_all, 0.60)),
             "yellow_max": float(np.quantile(p_all, 0.90)), "trained_through": str(d.index.max().date())},
            os.path.join(MOD, "risk_classifier.joblib"))

# ------------------------------- validation --------------------------------------------------------
ev = d[d.risk_v2.notna()].copy()
ev["period"] = np.where(ev.index.year == 2026, "2026", "2025")
rows = []
for per, o in list(ev.groupby("period")) + [("2025-26", ev)]:
    for ver, col in [("v1 width only", "tier_v1"), ("v2 classifier", "tier_v2")]:
        for t in ["Green", "Yellow", "Red"]:
            k = o[o[col] == t]
            rows.append(dict(period=per, version=ver, tier=t, days=len(k), share_of_days_pct=round(len(k) / len(o) * 100, 1),
                             days_with_a_short_hour_pct=round(k.label.mean() * 100, 1),
                             avg_short_hours=round(k.short_hours.mean(), 2),
                             avg_worst_short_MWh=round(k.max_short_vs_P95.mean(), 0),
                             avg_worst_P50_miss_MWh=round(k.max_abs_err_P50.mean(), 0)))
    rows.append(dict(period=per, version="AUC", tier="v1 width", days=len(o),
                     share_of_days_pct=round(roc_auc_score(o.label, o.width), 3)))
    rows.append(dict(period=per, version="AUC", tier="v2 classifier", days=len(o),
                     share_of_days_pct=round(roc_auc_score(o.label, o.risk_v2), 3)))
    rows.append(dict(period=per, version="Brier", tier="v2 classifier", days=len(o),
                     share_of_days_pct=round(brier_score_loss(o.label, o.risk_v2), 3)))
val = pd.DataFrame(rows)

# ------------------------------- decision test: risk-adjusted buy ----------------------------------
lvl = {"Green": "dem_P95", "Yellow": "dem_P98", "Red": "dem_P98"}
hh = h[h.day.isin(ev.index)].copy()
hh["tier"] = hh.day.map(ev.tier_v2)
hh["buy_adj"] = [hh.at[i, lvl[t]] if isinstance(t, str) else np.nan for i, t in zip(hh.index, hh.tier)]
dec = []
for per, o in list(hh.groupby(np.where(hh.index.year == 2026, "2026", "2025"))) + [("2025-26", hh)]:
    k = o[["dem_actual", "dem_P95", "buy_adj", "dem_pace_P95", "dem_pace"]].dropna()
    for name, col in [("Flat Maya P95", "dem_P95"), ("Risk-adjusted Maya (P95 Green, P98 Yellow/Red)", "buy_adj"),
                      ("PACE + cushion P95", "dem_pace_P95"), ("PACE as published", "dem_pace")]:
        sh = (k.dem_actual - k[col]).clip(lower=0)
        su = (k[col] - k.dem_actual).clip(lower=0)
        dec.append(dict(period=per, policy=name, hours=len(k), short_hours_pct=round((sh > 0).mean() * 100, 1),
                        avg_short_MWh=round(sh.mean(), 1), avg_surplus_MWh=round(su.mean(), 1),
                        worst_short_MWh=round(sh.max(), 0), cost_K19=round((19 * sh + su).mean(), 1), cost_K9=round((9 * sh + su).mean(), 1)))
dec = pd.DataFrame(dec)

d.round(4).to_csv(os.path.join(OUT, "daily_risk.csv"))
val.to_csv(os.path.join(OUT, "risk_validation.csv"), index=False)
dec.to_csv(os.path.join(OUT, "risk_adjusted_buy.csv"), index=False)
pd.set_option("display.width", 250)
print(val.to_string(index=False))
print()
print(dec.to_string(index=False))
imp = pd.Series(prod.predict_proba(d[FEATS])[:, 1]).describe()
print("\nBase rate of days with >=1 short hour vs P95:", round(d.label.mean() * 100, 1), "%")
