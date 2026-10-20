"""
Maya v2 week-ahead outlook: daily net-load totals for the 6 days after the buy day (lead 2 to 7 days from the
10 am Mountain deadline), as calibrated ranges.

Model   : quantile gradient boosting on daily totals (Mountain Standard Time days).
Inputs  : forecast weather for the target day (mean / max / min temperature, heating and cooling degrees, total
          sunshine, cloud), calendar (weekday, season, holiday), and recent net load that is fully known at the
          deadline: the last complete day, the last 7 complete days' mean, and the same weekday in the latest
          complete week. The lead time is an input too.
Training: 2019 and 2021-2023 with actual (reanalysis) weather, every lead 2-7.
Test    : Feb 2024 on, with the ECMWF forecast actually available at each lead (Open-Meteo previous_day2..7).
Calibration: per lead and level, adaptive conformal calibration: the shift is the lvl-th quantile of errors over
          the last CAL_DAYS days whose actuals were known by the issue date, and lvl is nudged after each of those
          days by whether it missed (GAMMA). Errors persist across days (weather regimes), so a fixed level would
          be too narrow; the adaptive level keeps P90 above the actual on about 90% of days at every lead.
Baseline: same weekday in the latest complete week.

Usage : python week_ahead.py           train, test, save model (also run monthly by the retrain workflow)
Live  : live_forecast.py calls outlook(...) every day.
Outputs: models/lag48/week_ahead.joblib, out/week_ahead_test.csv, app/data/week_ahead_eval.csv
"""
import os
import numpy as np
import pandas as pd
import joblib
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA, OUT, MOD, LIVE = (os.path.join(HERE, "data"), os.path.join(HERE, "out"), os.path.join(HERE, "models", "lag48"),
                        os.path.join(ROOT, "live"))
LEADS = range(2, 8)
QS = [0.10, 0.50, 0.90, 0.95]
CAL_DAYS, GAMMA = 240, 0.01
FEATS = ["lead", "t_mean", "t_max", "t_min", "hdd", "cdd", "ghi_sum", "cloud", "dow", "doy_sin", "doy_cos", "holiday",
         "last_day", "last7_mean", "same_wd", "same_wd_gap"]
GBM = dict(max_iter=400, max_depth=4, learning_rate=0.04, min_samples_leaf=20, random_state=42)
HOL = USFederalHolidayCalendar().holidays("2018-01-01", "2028-12-31")


def mst_day(index):
    return (index - pd.Timedelta(hours=7)).normalize()          # hour-ending UTC -> Mountain Standard Time day


def daily_weather(w):
    """Hourly weather (Temperature (C), GHI (W/m2), Cloud Cover (%)) -> daily features by MST day."""
    g = w.groupby(mst_day(w.index))
    t = g["Temperature (C)"]
    d = pd.DataFrame({"t_mean": t.mean(), "t_max": t.max(), "t_min": t.min(),
                      "hdd": (18 - w["Temperature (C)"]).clip(lower=0).groupby(mst_day(w.index)).mean(),
                      "cdd": (w["Temperature (C)"] - 18).clip(lower=0).groupby(mst_day(w.index)).mean(),
                      "ghi_sum": g["GHI (W/m2)"].sum() / 1000, "cloud": g["Cloud Cover (%)"].mean(), "hours": t.count()})
    return d[d.hours >= 20].drop(columns="hours")


def daily_net(net_hourly):
    g = net_hourly.groupby(mst_day(net_hourly.index))
    s = g.sum(min_count=23)
    return s[g.count() >= 23]


def rows(net_daily, wx_daily, targets, leads):
    """Feature rows for each (target day, lead). Issue date = target - lead (the buy-deadline day)."""
    out = []
    nd = net_daily
    for T in targets:
        if T not in wx_daily.index:
            continue
        for L in leads:
            issue = T - pd.Timedelta(days=L)
            # last complete MST day at the deadline (provisional runs before it is complete use the newest complete day;
            # never binds on history)
            last_known = min(issue - pd.Timedelta(days=1), nd.index.max())
            k = 1
            while T - pd.Timedelta(days=7 * k) > last_known:
                k += 1
            sw = T - pd.Timedelta(days=7 * k)
            w7 = nd.reindex(pd.date_range(last_known - pd.Timedelta(days=6), last_known, freq="D"))
            if last_known not in nd.index or sw not in nd.index or w7.count() < 5:
                continue
            r = dict(target=T, lead=L, issue=issue, **wx_daily.loc[T].to_dict(), dow=T.dayofweek,
                     doy_sin=np.sin(2 * np.pi * T.dayofyear / 365.25), doy_cos=np.cos(2 * np.pi * T.dayofyear / 365.25),
                     holiday=int(T in HOL), last_day=nd[last_known], last7_mean=w7.mean(), same_wd=nd[sw], same_wd_gap=7 * k,
                     actual=nd.get(T, np.nan))
            out.append(r)
    return pd.DataFrame(out)


def load_history():
    m = pd.read_csv(os.path.join(DATA, "master_hourly_cleaned_ALL_YEARS_v2.csv"),
                    usecols=["timestamp_utc", "Demand (MWh)", "Site Renewable Output Modeled (MWh)", "Temperature (C)", "GHI (W/m2)", "Cloud Cover (%)"])
    m["timestamp_utc"] = pd.to_datetime(m.timestamp_utc).dt.tz_localize(None)
    m = m.set_index("timestamp_utc").sort_index()
    hl = os.path.join(LIVE, "history_live.parquet")
    if os.path.exists(hl):
        h = pd.read_parquet(hl)
        h = h[h.index > m.index.max()][m.columns.intersection(h.columns)]
        m = pd.concat([m, h])
    m = m[~m.index.duplicated(keep="last")]
    net = m["Demand (MWh)"] - m["Site Renewable Output Modeled (MWh)"]
    return m, net


def lead_forecasts():
    f = pd.read_csv(os.path.join(DATA, "ecmwf_leadtime_2024_2026.csv.gz"), parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    f = f.rename(columns={"temperature_2m": "Temperature (C)", "shortwave_radiation": "GHI (W/m2)", "cloud_cover": "Cloud Cover (%)"})
    return {L: daily_weather(g) for L, g in f.groupby("lead_day")}


def fit(train):
    X, y = train[FEATS], train["actual"]
    return {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **GBM).fit(X, y) for q in QS}


def predict(models, X):
    p = np.column_stack([models[q].predict(X[FEATS]) for q in QS])
    p = np.maximum.accumulate(p, axis=1)
    return pd.DataFrame(p, columns=[f"P{int(q * 100):02d}" for q in QS], index=X.index)


def adaptive_calibration(test):
    """Walk-forward per lead and level. Only targets whose actual was known by the issue date (target <= issue - 1)
    are used, both for the error window and for nudging the level. Returns calibrated columns and final state."""
    out = test.copy()
    state = {}
    for L, g in test.sort_values("issue").groupby("lead"):
        issues, targets = g.issue.values, g.target.values
        for q in QS:
            c = f"P{int(q * 100):02d}"
            upper = q >= 0.5
            target_miss = (1 - q) if upper else q
            score = (g.actual - g[f"raw_{c}"]).values
            lvl, used, shifts = q, np.zeros(len(g), bool), np.zeros(len(g))
            for i in range(len(g)):
                known = targets <= issues[i] - np.timedelta64(1, "D")
                newly = known & ~used                       # outcomes that became known since the last issue
                for j in np.where(newly)[0]:
                    miss = (score[j] > shifts[j]) if upper else (score[j] < shifts[j])
                    lvl = float(np.clip(lvl + (GAMMA if upper else -GAMMA) * (miss - target_miss), 0.001, 0.999))
                used |= newly
                win = known & (targets > issues[i] - np.timedelta64(1 + CAL_DAYS, "D"))
                shifts[i] = np.quantile(score[win], lvl) if win.sum() >= 30 else 0.0
            out.loc[g.index, c] = g[f"raw_{c}"].values + shifts
            last = targets > targets.max() - np.timedelta64(CAL_DAYS, "D")
            state[(L, c)] = float(np.quantile(score[last], lvl))
    return out, state


def main():
    m, net = load_history()
    nd = daily_net(net)
    wx_act = daily_weather(m[["Temperature (C)", "GHI (W/m2)", "Cloud Cover (%)"]].dropna())
    # training rows: actual weather, 2019 and 2021-2023
    tr_days = [d for d in nd.index if (d.year in (2019, 2021, 2022, 2023))]
    train = rows(nd, wx_act, tr_days, LEADS)
    models = fit(train)
    # test rows: forecast weather at each lead, Feb 2024 on
    lf = lead_forecasts()
    test = []
    for L, wx in lf.items():
        r = rows(nd, wx, [d for d in wx.index if d >= pd.Timestamp("2024-02-10")], [L])
        if len(r):
            test.append(r)
    test = pd.concat(test, ignore_index=True)
    raw = predict(models, test)
    for c in raw:
        test[f"raw_{c}"] = raw[c].values
    test = test.dropna(subset=["actual"]).sort_values(["target", "lead"]).reset_index(drop=True)
    test, state = adaptive_calibration(test)
    test["naive"] = test.same_wd
    ev = []
    t = test[test.target >= "2024-08-01"]                       # after calibration warm-up
    for L, g in t.groupby("lead"):
        ev.append(dict(lead_days=L, days=len(g), mae_P50=round((g.actual - g.P50).abs().mean(), 0),
                       mae_naive=round((g.actual - g.naive).abs().mean(), 0),
                       better_than_naive_pct=round((1 - (g.actual - g.P50).abs().mean() / (g.actual - g.naive).abs().mean()) * 100, 1),
                       mape_P50=round(((g.actual - g.P50).abs() / g.actual).mean() * 100, 2),
                       above_P90_pct=round((g.actual > g.P90).mean() * 100, 1), above_P95_pct=round((g.actual > g.P95).mean() * 100, 1),
                       below_P10_pct=round((g.actual < g.P10).mean() * 100, 1)))
    ev = pd.DataFrame(ev)
    # production: retrain on all known days with actual weather; shifts from the latest test window
    all_days = [d for d in nd.index if d.year != 2020 and d in wx_act.index]
    prod = fit(rows(nd, wx_act, all_days, LEADS))
    shifts = {L: {c: state[(L, c)] for c in raw.columns} for L in LEADS}
    joblib.dump({"models": prod, "shifts": shifts, "features": FEATS, "trained_through": str(max(all_days).date())},
                os.path.join(MOD, "week_ahead.joblib"))
    test.to_csv(os.path.join(OUT, "week_ahead_test.csv"), index=False)
    ev.to_csv(os.path.join(ROOT, "app", "data", "week_ahead_eval.csv"), index=False)
    print(ev.to_string(index=False))


def outlook(df_hourly_actual_net, w_latest, buy_day, mod_path=os.path.join(MOD, "week_ahead.joblib")):
    """Live: daily net-load ranges for buy_day+1 .. buy_day+6 (lead 2..7 from the deadline on buy_day-1)."""
    if not os.path.exists(mod_path):
        return pd.DataFrame()
    pack = joblib.load(mod_path)
    nd = daily_net(df_hourly_actual_net.dropna())
    wx = daily_weather(w_latest[["Temperature (C)", "GHI (W/m2)", "Cloud Cover (%)"]].dropna())
    issue = buy_day - pd.Timedelta(days=1)
    out = []
    for L in LEADS:
        T = issue + pd.Timedelta(days=L)
        r = rows(nd, wx, [T], [L])
        if len(r):
            p = predict(pack["models"], r).iloc[0]
            out.append(dict(day=T.date(), lead_days=L, **{c: p[c] + pack["shifts"][L][c] for c in p.index}))
    return pd.DataFrame(out)


if __name__ == "__main__":
    main()
