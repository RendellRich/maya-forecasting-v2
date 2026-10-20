"""
Maya v2 buy strategies: hourly spike risk, hourly cushion rule, price-aware bid ladder, and baselines,
all scored at real CAISO EDAM prices.

Spike risk (per hour)
  Target  : real-time price more than SPIKE_USD above its normal level for that hour, where "normal" is the
            median real-time price at the same hour over the 14 days ending two days earlier (known at buy time).
            Before EDAM (May 2026) PACE had no day-ahead price, so this real-time-only target lets the model learn
            from Mar 2024 on. Since May 2026 it tracks real-time minus day-ahead closely.
  Inputs  : all known at the 10 am Mountain buy deadline the day before: local hour, weekday, month, ECMWF
            forecast temperature / sun / cloud, Maya's forecast net load and its hour-to-hour ramp, forecast
            uncertainty (P95 - P50), forecast load vs. the same hour's recent normal, and recent real-time price
            level and spike count (data ending two days before).
  Validation: walk-forward by month from Jan 2025, each month scored by a model trained only on earlier data.

Strategies (net load = what Maui Hiva buys)
  persistence      : no forecast, buy the same hour two days earlier (what was last fully known)
  pace             : PACE's published day-ahead forecast minus yesterday's renewables
  maya_day_rule    : P95 on Green days, P98 on Yellow / Red days (daily risk tier)
  maya_hour_rule   : P95, stepped up to P98 in the hours with the highest spike risk
  maya_ladder      : bid ladder (recommended). Block 1 = the daily-rule amount (P95 on Green days, P98 on Yellow / Red
                     days), bid at any price, so it is always bought. Block 2 = the extra up to P98 on Green days,
                     bid at a limit price equal to the expected real-time price for that hour (recent normal + spike
                     risk x typical spike premium); it is only bought if the day-ahead price clears at or below that
                     limit. Anything not bought day-ahead is covered at real-time.
                     Tested alternatives (May-Oct 2026): Block 1 = P95 every day scored -$0.181/MWh vs -$0.201 for this
                     design; the hourly rule on its own (-$0.139) did not beat the daily rule (-$0.171).
Settlement: cost = DA x buy + RT x (actual - buy). Shown vs perfect foresight (buy exactly actual).

Outputs: out/strategies_hourly.parquet, out/spike_validation.csv, out/strategy_results.csv,
         models/lag48/spike_classifier.joblib
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT, MOD, DATA, LIVE = (os.path.join(HERE, "out"), os.path.join(HERE, "models", "lag48"),
                        os.path.join(HERE, "data"), os.path.join(ROOT, "live"))
SPIKE_USD = 25.0
HIGH_RISK_SHARE = 0.20        # step up to P98 in the riskiest 20% of hours (threshold set on training data)
FEATS = ["lh", "weekday", "month", "temp_fc", "ghi_fc", "cloud_fc", "net_p50", "ramp", "width", "load_vs_normal",
         "rt_norm", "rt_level7", "spikes7"]
CLF = dict(max_iter=200, max_depth=4, learning_rate=0.05, min_samples_leaf=40, l2_regularization=1.0, random_state=42)
LOCAL = "America/Denver"


# ------------------------------- data -------------------------------------------------------------
def load_prices():
    p26 = pd.read_csv(os.path.join(DATA, "pace_prices_2026.csv"), parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    rt = p26.rt
    hist = os.path.join(DATA, "pace_rt_history_2024_2026.csv")
    if os.path.exists(hist):
        h = pd.read_csv(hist, parse_dates=["timestamp_utc"]).set_index("timestamp_utc").rt
        rt = pd.concat([h[h.index < p26.index.min()], rt])
    return rt[~rt.index.duplicated(keep="last")].sort_index(), p26.da


def load_forecasts():
    """Hourly Maya net-load levels, actuals, PACE, weather and daily tier: backtest (to Sep 1 2026) + live."""
    b = pd.read_parquet(os.path.join(OUT, "lag48", "maya_v2_hourly.parquet"))
    d = pd.read_csv(os.path.join(OUT, "lag48", "daily_risk.csv"), index_col=0, parse_dates=True)
    cols = ["net_P50", "net_P90", "net_P95", "net_P98", "dem_P50", "dem_P95", "dem_P98", "dem_actual", "ren_actual",
            "dem_pace", "temp_fc", "ghi_fc", "cloud_fc"]
    f = b[cols].copy()
    f["tier"] = pd.Series(f.index.normalize().map(d.tier_v2), index=f.index)
    lf = os.path.join(LIVE, "live_hourly.csv")
    if os.path.exists(lf):
        l = pd.read_csv(lf, parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
        l = l[l.index > f.index.max()].rename(columns={"pace_dayahead": "dem_pace"})
        for c in cols + ["tier"]:
            if c not in l:
                l[c] = np.nan
        f = pd.concat([f, l[cols + ["tier"]]])
    f = f[~f.index.duplicated(keep="last")].sort_index()
    f["net_actual"] = f.dem_actual - f.ren_actual
    f["net_pace"] = f.dem_pace - f.ren_actual.shift(24)
    return f


def same_hour_trailing(s, days, fn, lag_h=48):
    """Stat over the same hour of the `days` days ending lag_h hours earlier (so it is known at buy time)."""
    x = s.shift(lag_h)
    return x.groupby(x.index.hour).transform(lambda v: v.rolling(days, min_periods=max(3, days // 3)).agg(fn))


def features(f, rt):
    idx = f.index
    loc = idx.tz_localize("UTC").tz_convert(LOCAL)
    X = pd.DataFrame(index=idx)
    X["lh"], X["weekday"], X["month"] = loc.hour, loc.dayofweek, loc.month
    X["temp_fc"], X["ghi_fc"], X["cloud_fc"] = f.temp_fc, f.ghi_fc, f.cloud_fc
    X["net_p50"] = f.net_P50
    X["ramp"] = f.net_P50 - f.net_P50.shift(1)
    X["width"] = f.net_P95 - f.net_P50
    X["load_vs_normal"] = f.net_P50 - same_hour_trailing(f.net_actual, 14, "mean").reindex(idx)
    rt_h = rt.reindex(pd.date_range(min(rt.index.min(), idx.min()), max(rt.index.max(), idx.max()), freq="h"))
    norm = same_hour_trailing(rt_h, 14, "median")
    X["rt_norm"] = norm.reindex(idx)
    X["rt_level7"] = rt_h.shift(48).rolling(24 * 7, min_periods=48).mean().reindex(idx)
    spike_hist = ((rt_h - norm) > SPIKE_USD).astype(float).where(rt_h.notna() & norm.notna())
    X["spikes7"] = spike_hist.shift(48).rolling(24 * 7, min_periods=48).sum().reindex(idx)
    y = ((rt_h - norm) > SPIKE_USD).astype(float).where(rt_h.notna() & norm.notna()).reindex(idx)
    return X, y


# ------------------------------- spike model ------------------------------------------------------
def walk_forward(X, y, start="2025-01"):
    ok = y.notna() & X[["net_p50", "rt_norm"]].notna().all(axis=1)
    prob = pd.Series(np.nan, index=X.index)
    thr = pd.Series(np.nan, index=X.index)
    months = pd.period_range(start, X.index.max().to_period("M"), freq="M")
    for m in months:
        te = (X.index.to_period("M") == m) & X[["net_p50"]].notna().all(axis=1).values
        tr = ok & (X.index < m.start_time - pd.Timedelta(days=2))
        if te.sum() == 0 or tr.sum() < 2000 or y[tr].sum() < 20:
            continue
        clf = HistGradientBoostingClassifier(**CLF).fit(X.loc[tr, FEATS], y[tr])
        prob[te] = clf.predict_proba(X.loc[te, FEATS])[:, 1]
        thr[te] = np.quantile(clf.predict_proba(X.loc[tr, FEATS])[:, 1], 1 - HIGH_RISK_SHARE)
    return prob, thr


def fit_production(X, y):
    ok = y.notna() & X[["net_p50", "rt_norm"]].notna().all(axis=1)
    clf = HistGradientBoostingClassifier(**CLF).fit(X.loc[ok, FEATS], y[ok])
    p = clf.predict_proba(X.loc[ok, FEATS])[:, 1]
    spread_hist = None
    return {"model": clf, "features": FEATS, "threshold": float(np.quantile(p, 1 - HIGH_RISK_SHARE)),
            "base_rate": float(y[ok].mean()), "spike_usd": SPIKE_USD, "trained_through": str(X.index[ok].max())}


# ------------------------------- strategies -------------------------------------------------------
def expected_rt(X, prob, spike_premium):
    """Expected real-time price for the hour: recent normal + chance of a spike x typical spike premium."""
    return X.rt_norm + prob * spike_premium


def add_strategies(f, X, prob, thr, spike_premium):
    s = pd.DataFrame(index=f.index)
    s["actual"] = f.net_actual
    s["perfect"] = f.net_actual
    s["persistence"] = f.net_actual.shift(48)
    s["pace"] = f.net_pace
    s["maya_P95"] = f.net_P95
    s["maya_day_rule"] = np.where(f.tier.isin(["Yellow", "Red"]), f.net_P98, f.net_P95)
    high = (prob >= thr).fillna(False)
    s["spike_prob"] = prob
    s["high_risk_hour"] = high.astype(int)
    s["maya_hour_rule"] = np.where(high, f.net_P98, f.net_P95)
    s["ladder_block1"] = s["maya_day_rule"]
    s["ladder_block2"] = (f.net_P98 - s["maya_day_rule"]).clip(lower=0)
    s["ladder_limit"] = expected_rt(X, prob, spike_premium)
    return s


def settle(s, da, rt, col):
    k = s[["actual"]].join(pd.DataFrame({"da": da, "rt": rt}), how="inner")
    if col == "maya_ladder":
        buy = s.ladder_block1 + np.where(da.reindex(s.index) <= s.ladder_limit, s.ladder_block2, 0)
    else:
        buy = s[col]
    k["buy"] = pd.Series(buy, index=s.index).reindex(k.index)
    k = k.dropna()
    gap = k.actual - k.buy
    extra = ((k.rt - k.da) * gap)
    return dict(hours=len(k), extra_per_MWh=extra.sum() / k.actual.sum(), extra_total=extra.sum(),
                short_pct=(gap > 0).mean() * 100, volume_pct=(k.buy.sum() / k.actual.sum() - 1) * 100,
                worst_short=gap.clip(lower=0).max(), bill=(k.da * k.buy + k.rt * gap).sum(), mwh=k.actual.sum())


NAMES = {"persistence": "No forecast (same hour 2 days ago)", "pace": "PACE as published", "maya_P95": "Maya P95",
         "maya_day_rule": "Maya daily rule (P95, P98 on Yellow/Red days)",
         "maya_hour_rule": "Maya hourly rule (P95, P98 in high spike-risk hours)",
         "maya_ladder": "Maya bid ladder (daily rule + extra to P98 if day-ahead is below expected real-time)"}


def build(write=True):
    rt, da = load_prices()
    f = load_forecasts()
    X, y = features(f, rt)
    prob, thr = walk_forward(X, y)
    # typical spike premium over normal, from data before each scored month would be ideal; a single long-run value
    # from 2024-25 (before any scored settlement period) keeps it simple and leak-free for 2026
    pre = (X.index < "2026-01-01") & (y == 1)
    spike_premium = float((rt.reindex(X.index) - X.rt_norm)[pre].median()) if pre.sum() else 40.0
    s = add_strategies(f, X, prob, thr, spike_premium)
    s["da"], s["rt"] = da.reindex(s.index), rt.reindex(s.index)
    s["maya_ladder"] = s.ladder_block1 + np.where(s.da <= s.ladder_limit, s.ladder_block2, 0)   # quantity actually cleared
    s.loc[s.da.isna(), "maya_ladder"] = np.nan
    s["spike_actual"] = y
    # validation of the spike model
    ev = pd.DataFrame({"p": prob, "y": y, "thr": thr}).dropna()
    rows = []
    for per, o in [("2025", ev[ev.index.year == 2025]), ("2026", ev[ev.index.year == 2026]), ("2025-26", ev)]:
        if len(o) and o.y.nunique() == 2:
            hi = o.p >= o.thr
            rows.append(dict(period=per, hours=len(o), spike_rate_pct=round(o.y.mean() * 100, 2), auc=round(roc_auc_score(o.y, o.p), 3),
                             high_risk_share_pct=round(hi.mean() * 100, 1),
                             spike_rate_high_pct=round(o.y[hi].mean() * 100, 2), spike_rate_other_pct=round(o.y[~hi].mean() * 100, 2),
                             share_of_spikes_caught_pct=round(o.y[hi].sum() / o.y.sum() * 100, 1)))
    val = pd.DataFrame(rows)
    res = []
    for col in ["perfect", "persistence", "pace", "maya_P95", "maya_day_rule", "maya_hour_rule", "maya_ladder"]:
        r = settle(s, da, rt, col)
        res.append(dict(strategy=NAMES.get(col, "Perfect foresight"), key=col, **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))
    res = pd.DataFrame(res)
    prod = fit_production(X, y)
    prod["spike_premium"] = spike_premium
    # hours with no outcome yet (the next buy day): score with the production model trained on everything so far
    fut = s.actual.isna() & X[["net_p50", "rt_norm"]].notna().all(axis=1)
    if fut.any():
        p_f = prod["model"].predict_proba(X.loc[fut, FEATS])[:, 1]
        s.loc[fut, "spike_prob"] = p_f
        s.loc[fut, "high_risk_hour"] = (p_f >= prod["threshold"]).astype(int)
        s.loc[fut, "maya_hour_rule"] = np.where(p_f >= prod["threshold"], f.net_P98[fut], f.net_P95[fut])
        s.loc[fut, "ladder_limit"] = expected_rt(X.loc[fut], pd.Series(p_f, index=X.index[fut]), spike_premium)
    if write:
        s.to_parquet(os.path.join(OUT, "strategies_hourly.parquet"))
        # small copy for the dashboard: 2026 hours from the start of EDAM pricing on (committed daily by the job)
        app_cols = ["actual", "perfect", "persistence", "pace", "maya_P95", "maya_day_rule", "maya_hour_rule", "maya_ladder",
                    "ladder_block1", "ladder_block2", "ladder_limit", "spike_prob", "high_risk_hour", "spike_actual", "da", "rt"]
        s.loc[s.index >= "2026-04-25", app_cols].astype("float32").to_parquet(os.path.join(ROOT, "app", "data", "strategies_2026.parquet"))
        val.to_csv(os.path.join(OUT, "spike_validation.csv"), index=False)
        val.to_csv(os.path.join(ROOT, "app", "data", "spike_validation.csv"), index=False)
        res.to_csv(os.path.join(ROOT, "app", "data", "strategy_results.csv"), index=False)
        res.to_csv(os.path.join(OUT, "strategy_results.csv"), index=False)
        joblib.dump(prod, os.path.join(MOD, "spike_classifier.joblib"))
    return s, val, res, spike_premium


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    s, val, res, prem = build()
    print("spike premium used for expected real-time:", round(prem, 1))
    print(val.to_string(index=False))
    print(res.drop(columns=["key"]).to_string(index=False))
