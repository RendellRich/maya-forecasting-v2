"""
Maya v2 live forecast: the daily buy-ahead run.

What one run does
  1. Pulls PACE demand + PACE day-ahead forecast from the EIA API (EIA-930), and ECMWF weather for the Milford
     point from Open-Meteo (same model and point as training).
  2. Rebuilds the Milford-area renewable output estimate for recent hours (same physics + calibration as
     build_renewable_target_v3.py: pvlib tracker model x 2025 average EIA-923 factor + 2025 same-month baseload).
  3. Forecasts every buy day from the end of the training data (Sep 2 2026) up to the NEXT buy day, using the
     production 48 h models (models/lag48). Past days use the ECMWF forecast issued the day before
     (previous_day1, as in training); the next buy day uses the latest ECMWF run.
  4. Calibrates every level with the same adaptive method as the backtest, using the backtest history plus
     all live days, so P95 keeps meaning "short on about 5% of hours".
  5. Scores the risk tier for the next buy day and sets the recommended buy:
       Green -> P95,  Yellow / Red -> P98   (net load = PACE demand - Milford renewable)
  6. Scores every finished live day against actual demand (the live track record), and in dollars using
     CAISO EDAM day-ahead and real-time PACE prices (prices.py).

Buy day = a Mountain Standard Time calendar day (07:00 UTC to 06:00 UTC next day), bought at 16:00 UTC
(10 am MDT / 9 am MST) the day before. Run it any time after about 16:15 UTC for the official forecast;
an earlier run is marked provisional because the newest demand reading is not in yet.

Usage
  python live_forecast.py            normal daily run (writes live/)
  python live_forecast.py --verify   check: rebuild the 2026 backtest from the saved models and compare
Environment: EIA_API_KEY (falls back to EIA's DEMO_KEY, which is rate limited).
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import joblib
from pandas.tseries.holiday import USFederalHolidayCalendar
from maya_common import QD, QR, qn, calibrate, solar_physics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, MOD = os.path.join(HERE, "data"), os.path.join(HERE, "models", "lag48")
BACKTEST = os.path.join(HERE, "out", "lag48", "quantile_forecasts_raw.parquet")
LIVE = os.path.join(os.path.dirname(HERE), "live")
MIN_LAG, ISSUE_HOUR_UTC, WIN = 48, 16, 168
LAT, LON = 38.4167, -113.0167
SOLAR_FACTOR_2025 = 0.815          # mean 2025 EIA-923 / physics factor (same rule v3 uses for months not yet published)
BASELOAD_2025 = {1: 54.82, 2: 54.4, 3: 55.7, 4: 54.51, 5: 55.72, 6: 50.02, 7: 49.17, 8: 47.0, 9: 44.57,
                 10: 57.58, 11: 62.66, 12: 61.6}        # geothermal/hydro/biomass MWh per hour, 2025 same month
D, R = "Demand (MWh)", "Site Renewable Output Modeled (MWh)"
WX_D = ["GHI (W/m2)", "Temperature (C)", "Wind Speed 10m (m/s)", "Wind Speed 100m (m/s)", "Wind Speed 80m (m/s)"]
SEQ_D = ["Temperature (C)", "GHI (W/m2)"]
WX_R = ["GHI (W/m2)", "Temperature (C)", "DNI (W/m2)", "DHI (W/m2)", "Installed Solar DC (MW)", "Cloud Cover (%)", "Clear-sky Index"]
CAL = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month", "is_weekend"]
OM_VARS = {"temperature_2m": "Temperature (C)", "shortwave_radiation": "GHI (W/m2)",
           "direct_normal_irradiance": "DNI (W/m2)", "diffuse_radiation": "DHI (W/m2)", "cloud_cover": "Cloud Cover (%)",
           "wind_speed_10m": "Wind Speed 10m (m/s)", "wind_speed_100m": "Wind Speed 100m (m/s)"}
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/root/.ccr/ca-bundle.crt") if os.path.exists("/root/.ccr/ca-bundle.crt") else None


# ------------------------------- data -------------------------------------------------------------
def get_json(url, params, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code} from {url.split('/')[2]}, retrying", flush=True)
        except requests.RequestException as e:
            print(f"  {e.__class__.__name__} from {url.split('/')[2]}, retrying", flush=True)
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"could not reach {url}")


def fetch_eia(start, end):
    """PACE hourly demand (D) and PACE day-ahead demand forecast (DF), UTC hour-ending."""
    key = os.environ.get("EIA_API_KEY") or "DEMO_KEY"      # empty secret -> demo key
    out = {}
    for t in ["D", "DF"]:
        rows, off = [], 0
        while True:
            j = get_json("https://api.eia.gov/v2/electricity/rto/region-data/data/", {
                "api_key": key, "frequency": "hourly", "data[0]": "value", "facets[respondent][]": "PACE",
                "facets[type][]": t, "start": start.strftime("%Y-%m-%dT%H"), "end": end.strftime("%Y-%m-%dT%H"),
                "sort[0][column]": "period", "sort[0][direction]": "asc", "offset": off, "length": 5000})
            data = j["response"]["data"]
            rows += data
            if len(data) < 5000:
                break
            off += 5000
        s = pd.Series({pd.Timestamp(x["period"]): float(x["value"]) for x in rows if x["value"] is not None}, dtype=float)
        out[t] = s.sort_index()
    return out["D"], out["DF"]


def fetch_weather(start_date, end_date):
    """ECMWF IFS 0.25 for the Milford point. Returns (latest-run values, previous_day1 values)."""
    j = get_json("https://previous-runs-api.open-meteo.com/v1/forecast", {
        "latitude": LAT, "longitude": LON, "models": "ecmwf_ifs025", "timezone": "GMT", "wind_speed_unit": "ms",
        "start_date": str(start_date), "end_date": str(end_date),
        "hourly": ",".join(list(OM_VARS) + [v + "_previous_day1" for v in OM_VARS])})
    h = pd.DataFrame(j["hourly"])
    h.index = pd.to_datetime(h.pop("time"))
    latest = h[list(OM_VARS)].rename(columns=OM_VARS)
    prev1 = h[[v + "_previous_day1" for v in OM_VARS]].rename(columns={v + "_previous_day1": c for v, c in OM_VARS.items()})
    return latest, prev1


WEATHER_STORE = os.path.join(DATA, "weather_live_{}.csv")


def cached_weather(master_end, target_day):
    """Weather since the training data ends, kept in data/weather_live_*.csv so each run only downloads the
    last few days plus the forecast (keeps the Open-Meteo request small and under its rate limits).
    Past days keep the values first stored, which is what the forecast at that time would have seen."""
    start_all = (master_end - pd.Timedelta(days=10)).normalize()
    stored = {}
    for k in ["latest", "prev1"]:
        f = WEATHER_STORE.format(k)
        stored[k] = pd.read_csv(f, parse_dates=["timestamp_utc"]).set_index("timestamp_utc") if os.path.exists(f) else pd.DataFrame()
    have = stored["latest"].index.max() if not stored["latest"].empty else None
    start = start_all if have is None else max(start_all, (have - pd.Timedelta(days=4)).normalize())
    new_latest, new_prev1 = fetch_weather(start.date(), (target_day + pd.Timedelta(days=2)).date())
    out = []
    for k, new in [("latest", new_latest), ("prev1", new_prev1)]:
        old = stored[k]
        # previous_day1 values for past hours never change; "latest" for past hours is the analysis-like short-range value
        merged = pd.concat([old[old.index < new.index.min()], new]) if not old.empty else new
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.index.name = "timestamp_utc"
        merged.round(3).to_csv(WEATHER_STORE.format(k))
        out.append(merged)
    return out[0], out[1]


def clean_demand(s):
    """EIA's newest hours are often partial; drop the last 2 hours and obvious bad values."""
    s = s.copy()
    if len(s):
        s = s[s.index <= s.index.max() - pd.Timedelta(hours=2)]
    s[(s < 2000) | (s > 15000)] = np.nan
    med = s.rolling(24 * 7, min_periods=48).median()
    s[(s - med).abs() > 0.45 * med] = np.nan
    return s


# ------------------------------- features (must match quantile_models.py, MIN_LAG = 48) -------------
def build_features(df, fc):
    ts = df.index
    year = ts.year.values
    df = df.copy()
    fc = fc.copy()
    df["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * ts.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * ts.dayofyear / 365.25)
    df["month"] = ts.month
    df["is_weekend"] = (ts.dayofweek >= 5).astype(int)
    df["Installed Solar DC (MW)"] = np.where(year <= 2020, 377.1, 476.1) * 1.25
    cs = df["Clear-sky GHI (W/m2)"]
    fc["Wind Speed 80m (m/s)"] = fc["Wind Speed 10m (m/s)"].clip(lower=0.01) * 8 ** (
        np.log(fc["Wind Speed 100m (m/s)"].clip(lower=0.01) / fc["Wind Speed 10m (m/s)"].clip(lower=0.01)) / np.log(10))
    fc["Clear-sky Index"] = np.where(cs >= 20, (fc["GHI (W/m2)"] / cs).clip(0, 1.5), np.nan)
    fc["Installed Solar DC (MW)"] = df["Installed Solar DC (MW)"]

    def lags(y):
        o = pd.DataFrame(index=y.index)
        L = MIN_LAG
        o[f"lag{L}"], o[f"lag{L + 24}"], o["lag168"] = y.shift(L), y.shift(L + 24), y.shift(168)
        o["roll7"] = pd.concat([y.shift(24 * k) for k in range(L // 24, L // 24 + 7)], axis=1).mean(axis=1)
        return o

    def buy_time(y, temp_target):
        local_day = (ts - pd.Timedelta(hours=7)).normalize()
        last_known = local_day - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR_UTC - 2)
        # provisional runs (before the buy-time reading exists): fall back to the newest reading available.
        # Never binds on history, so backtest features are unchanged (checked by --verify).
        newest = y.last_valid_index()
        last_known = pd.DatetimeIndex(np.minimum(last_known.values, np.datetime64(newest)))
        e = pd.DataFrame(index=ts)
        e["last_known"] = y.reindex(last_known).values
        e["last_known_change_1d"] = e["last_known"] - y.reindex(last_known - pd.Timedelta(days=1)).values
        e["hours_since_known"] = (ts - last_known) / pd.Timedelta(hours=1)
        e["same_hour_latest"] = np.where(ts - pd.Timedelta(hours=24) <= last_known, y.shift(24).values, y.shift(48).values)
        e["temp_change_vs_lag"] = temp_target - df["Temperature (C)"].shift(MIN_LAG)
        return e

    y = df[D].astype(float)
    G = pd.concat([df[WX_D + CAL], lags(y)], axis=1)
    for c in WX_D:
        G[c] = fc[c]
    now = df[WX_D + CAL].copy()
    for c in WX_D:
        now[c] = fc[c]
    e_fc = buy_time(y, fc["Temperature (C)"])
    G, now = pd.concat([G, e_fc], axis=1), pd.concat([now, e_fc], axis=1)
    seq_df = pd.concat([y.rename("y"), df[SEQ_D], df[["hour_sin", "hour_cos", "doy_sin", "doy_cos"]]], axis=1)
    seq_ok = seq_df.notna().all(axis=1).astype(int).rolling(WIN).sum().shift(MIN_LAG).fillna(0).values == WIN
    yr = df[R].astype(float)
    H = pd.concat([df[WX_R + CAL], lags(yr)], axis=1)
    for c in WX_R:
        H[c] = fc[c]
    return dict(G=G, now=now, seq_df=seq_df, seq_ok=seq_ok, H=H, e=e_fc)


class QNet(nn.Module):
    def __init__(self, n_seq, n_now, nq, hidden=48):
        super().__init__()
        self.lstm = nn.LSTM(n_seq, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_now, 64), nn.ReLU(), nn.Linear(64, nq))

    def forward(self, seq, now):
        _, (h, _) = self.lstm(seq)
        z = self.head(torch.cat([h[-1], now], dim=1))
        return torch.cat([z[:, :1], z[:, :1] + torch.cumsum(nn.functional.softplus(z[:, 1:]), dim=1)], dim=1)


def load_models():
    gbms = joblib.load(os.path.join(MOD, "demand_gbm_quantiles.joblib"))
    ck = torch.load(os.path.join(MOD, "demand_lstm_quantiles.pt"), weights_only=False)
    net = QNet(ck["n_seq"], ck["n_now"], len(ck["quantiles"]))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    sc = {k: pd.Series(v) for k, v in ck["scaler"].items()}
    ren = joblib.load(os.path.join(MOD, "renewable_gbm_quantiles.joblib"))
    return gbms, net, sc, ck, ren


def predict(F, target_mask, models):
    gbms, net, sc, ck, ren = models
    G, now, seq_df, seq_ok, H = F["G"], F["now"], F["seq_df"][ck["seq_cols"]], F["seq_ok"], F["H"]
    now = now[ck["now_cols"]]
    ts = G.index
    ok = target_mask & G.notna().all(axis=1).values & seq_ok
    ids = np.where(ok)[0]
    res = pd.DataFrame(index=ts[ids])
    if len(ids):
        S = torch.tensor(((seq_df - sc["mu_s"]) / sc["sd_s"]).fillna(0).values, dtype=torch.float32)
        N = torch.tensor(((now - sc["mu_n"]) / sc["sd_n"]).fillna(0).values, dtype=torch.float32)
        mu, sd = float(sc["mu_s"]["y"]), float(sc["sd_s"]["y"])
        off = torch.arange(-WIN + 1, 1)
        with torch.no_grad():
            pl = np.concatenate([net(S[(torch.tensor(ids[b:b + 2048] - MIN_LAG)[:, None] + off[None, :])], N[ids[b:b + 2048]]).numpy()
                                 for b in range(0, len(ids), 2048)]) * sd + mu
        gm = np.maximum.accumulate(np.column_stack([gbms[q].predict(G.iloc[ids]) for q in QD]), axis=1)
        for j, q in enumerate(QD):
            res[f"dem_gbm_{qn(q)}"], res[f"dem_lstm_{qn(q)}"] = gm[:, j], pl[:, j]
            res[f"dem_raw_{qn(q)}"] = (gm[:, j] + pl[:, j]) / 2
    need = [c for c in H.columns if c != "Clear-sky Index"]
    rid = np.where(target_mask & H[need].notna().all(axis=1).values)[0]
    rr = pd.DataFrame(index=ts[rid])
    if len(rid):
        p = np.maximum.accumulate(np.column_stack([np.clip(ren[q].predict(H.iloc[rid]), 0, None) for q in QR]), axis=1)
        for j, q in enumerate(QR):
            rr[f"ren_raw_{qn(q)}"] = p[:, j]
    return res.join(rr, how="outer")


# ------------------------------- helpers -----------------------------------------------------------
def load_master():
    m = pd.read_csv(os.path.join(DATA, "master_hourly_cleaned_ALL_YEARS_v2.csv"))
    m["timestamp_utc"] = pd.to_datetime(m["timestamp_utc"]).dt.tz_localize(None)
    m = m.set_index("timestamp_utc").sort_index()
    fc = pd.read_csv(os.path.join(DATA, "ecmwf_dayahead_forecast_2024-2026.csv"),
                     parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    return m, fc


def clear_sky_climatology(m):
    k = m["Clear-sky GHI (W/m2)"]
    k = k[k.index.year <= 2025]
    return k.groupby([np.minimum(k.index.dayofyear, 365), k.index.hour]).mean()


def buy_day_hours(day):
    start = pd.Timestamp(day) + pd.Timedelta(hours=7)
    return pd.date_range(start, periods=24, freq="h")


def next_buy_day(now_utc):
    """The day whose buy is the next one to be placed (or was placed today if before 16:00 UTC... see docstring)."""
    issue_today = now_utc.normalize() + pd.Timedelta(hours=ISSUE_HOUR_UTC)
    issue_date = now_utc.normalize() if now_utc <= issue_today + pd.Timedelta(hours=4) else now_utc.normalize() + pd.Timedelta(days=1)
    return issue_date + pd.Timedelta(days=1)


DRIVER_GROUPS = {
    "Recent demand": ["lag48", "lag72", "lag168", "roll7", "last_known", "last_known_change_1d", "hours_since_known", "same_hour_latest"],
    "Temperature": ["Temperature (C)", "temp_change_vs_lag"],
    "Sun and wind": ["GHI (W/m2)", "Wind Speed 10m (m/s)", "Wind Speed 100m (m/s)", "Wind Speed 80m (m/s)"],
    "Calendar": ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month", "is_weekend"],
}


def explain(G, index):
    """Mean SHAP contribution per hour, by driver group, for each buy day (demand GBM P50).
    'baseline' is the model's average hour over its training years."""
    if len(index) == 0:
        return pd.DataFrame()
    import shap
    g = joblib.load(os.path.join(MOD, "demand_gbm_quantiles.joblib"))[0.5]
    X = G.loc[index]
    ex = shap.TreeExplainer(g)
    sv = pd.DataFrame(ex.shap_values(X), index=X.index, columns=X.columns)
    grp = pd.DataFrame({k: sv[[c for c in v if c in sv]].sum(axis=1) for k, v in DRIVER_GROUPS.items()})
    grp["buy_day"] = (grp.index - pd.Timedelta(hours=7)).normalize().date
    out = grp.groupby("buy_day").mean()
    out.insert(0, "baseline", float(np.ravel(ex.expected_value)[0]))
    out["gbm_P50_mean"] = out.drop(columns="baseline").sum(axis=1) + out["baseline"]
    return out.round(1)


# ------------------------------- verify mode -------------------------------------------------------
def verify():
    m, fc = load_master()
    fc = fc.reindex(m.index)
    F = build_features(m, fc)
    mask = (m.index.year == 2026) & (m.index <= m[D].last_valid_index())
    p = predict(F, np.asarray(mask), load_models())
    bt = pd.read_parquet(BACKTEST)
    bt = bt[bt.period == "B forward test 2026"]
    cols = [c for c in p.columns if c in bt.columns]
    k = p[cols].join(bt[cols], rsuffix="_bt", how="inner")
    worst = max((k[c] - k[c + "_bt"]).abs().max() for c in cols)
    print(f"verify: {len(k)} hours, {len(cols)} columns, largest difference vs backtest = {worst:.6f} MWh")
    return worst < 1e-3


# ------------------------------- main run ----------------------------------------------------------
def run(now_utc=None):
    t0 = time.time()
    now_utc = pd.Timestamp.now("UTC").tz_localize(None).floor("h") if now_utc is None else pd.Timestamp(now_utc)
    target_day = next_buy_day(now_utc)
    os.makedirs(LIVE, exist_ok=True)
    m, fc_bt = load_master()
    master_end = m[D].last_valid_index()
    live_start_day = (master_end + pd.Timedelta(hours=1) - pd.Timedelta(hours=7)).normalize() + pd.Timedelta(days=1)
    idx = pd.date_range(m.index.min(), buy_day_hours(target_day)[-1], freq="h")
    print(f"Run {now_utc} UTC | next buy day {target_day.date()} | live days from {live_start_day.date()}", flush=True)

    # --- fetch
    d_eia, df_eia = fetch_eia(master_end - pd.Timedelta(days=10), now_utc + pd.Timedelta(days=2))
    d_eia = clean_demand(d_eia[d_eia.index <= now_utc])          # never use readings after the run time
    df_eia = df_eia[df_eia.index <= now_utc + pd.Timedelta(days=2)]
    w_latest, w_prev1 = cached_weather(master_end, target_day)
    print(f"  EIA demand through {d_eia.index.max()} | weather {w_latest.index.min()} .. {w_latest.index.max()}", flush=True)

    # --- hourly table: master history, then live hours
    df = m.reindex(idx)
    live_h = idx > master_end
    df.loc[live_h, D] = d_eia.reindex(idx[live_h]).values
    df.loc[live_h, "Day-Ahead Demand Forecast (MWh)"] = df_eia.reindex(idx[live_h]).values
    for c in ["Temperature (C)", "GHI (W/m2)", "DNI (W/m2)", "DHI (W/m2)", "Cloud Cover (%)", "Wind Speed 10m (m/s)", "Wind Speed 100m (m/s)"]:
        df.loc[live_h, c] = w_latest[c].reindex(idx[live_h]).values           # observed-weather stand-in (latest run)
    csc = clear_sky_climatology(m)
    key = list(zip(np.minimum(idx[live_h].dayofyear, 365), idx[live_h].hour))
    df.loc[live_h, "Clear-sky GHI (W/m2)"] = csc.reindex(key).values
    known = live_h & (idx <= now_utc)
    sol = solar_physics(idx[known], df.loc[known, "GHI (W/m2)"], df.loc[known, "DNI (W/m2)"],
                        df.loc[known, "DHI (W/m2)"], df.loc[known, "Temperature (C)"]) * SOLAR_FACTOR_2025
    mst_month = (idx[known] - pd.Timedelta(minutes=30) - pd.Timedelta(hours=7)).month
    df.loc[known, R] = sol.values + np.array([BASELOAD_2025[x] for x in mst_month])

    # forecast weather: backtest file through its end, previous_day1 for later past days, latest run for the target day
    fcw = fc_bt.reindex(idx)
    later = idx > fc_bt.index.max()
    for c in w_prev1.columns:
        fcw.loc[later, c] = w_prev1[c].reindex(idx[later]).values
    tgt = np.isin(idx, buy_day_hours(target_day))
    for c in w_latest.columns:
        fcw.loc[tgt, c] = w_latest[c].reindex(idx[tgt]).values

    # --- forecast all live days (backfill) + the target day
    F = build_features(df, fcw)
    live_days = pd.date_range(live_start_day, target_day, freq="D")
    want = np.isin(idx, np.concatenate([buy_day_hours(d) for d in live_days]))
    raw = predict(F, want, load_models())
    raw["buy_day"] = (raw.index - pd.Timedelta(hours=7)).normalize()
    last_known_target = buy_day_hours(target_day)[0] - pd.Timedelta(hours=7 + 24 - ISSUE_HOUR_UTC + 2)
    provisional = bool(d_eia.index.max() < last_known_target) if len(d_eia) else True
    raw["issued"] = np.where(raw.buy_day == target_day, str(now_utc), "backfill (day-ahead weather, same rules)")

    # --- why: SHAP drivers of the demand GBM P50 (half of the blend), grouped, mean per hour of each buy day
    drivers = explain(F["G"], raw.index[raw["dem_raw_P50"].notna()] if "dem_raw_P50" in raw else raw.index[:0])

    # --- calibration on backtest + live history
    bt = pd.read_parquet(BACKTEST)
    lv = raw[[c for c in raw.columns if c.startswith(("dem_raw", "ren_raw", "dem_gbm", "dem_lstm"))]]
    hist = pd.concat([bt[lv.columns], lv[lv.index > bt.index.max()]]).sort_index()
    act_d = df[D].reindex(hist.index)
    act_r = df[R].reindex(hist.index)
    act_r[hist.index > now_utc] = np.nan
    cal = pd.DataFrame(index=hist.index)
    for q in QD:
        cal[f"dem_{qn(q)}"] = calibrate(hist[f"dem_raw_{qn(q)}"], act_d, q)
        cal[f"net_{qn(q)}"] = calibrate(hist[f"dem_raw_{qn(q)}"] - hist[f"ren_raw_{qn(round(1 - q, 2))}"], act_d - act_r, q)
    for q in QR:
        cal[f"ren_{qn(q)}"] = calibrate(hist[f"ren_raw_{qn(q)}"], act_r, q, upper=False).clip(lower=0)
    for pre, qs in [("dem", QD), ("net", QD), ("ren", QR)]:
        cols = [f"{pre}_{qn(q)}" for q in qs]
        cal[cols] = np.maximum.accumulate(cal[cols].values, axis=1)
    cal = cal[cal.index > master_end]

    # --- risk tier per live buy day (features as in risk_score.py, over the buy day's 24 hours)
    rk = joblib.load(os.path.join(MOD, "risk_classifier.joblib"))
    h = cal.join(raw[["dem_raw_P50", "dem_raw_P95", "dem_gbm_P50", "dem_lstm_P50", "buy_day"]])
    h["dem_actual"] = df[D].reindex(h.index)
    h["temp_fc"], h["cloud_fc"] = fcw["Temperature (C)"].reindex(h.index), fcw["Cloud Cover (%)"].reindex(h.index)
    h["ghi_fc"] = fcw["GHI (W/m2)"].reindex(h.index)
    h["temp_lag"] = df["Temperature (C)"].shift(MIN_LAG).reindex(h.index)
    allerr = pd.concat([(bt["dem_actual"] - calibrate(bt["dem_raw_P50"], bt["dem_actual"], 0.5)).abs(),
                        (h["dem_actual"] - h["dem_P50"]).abs()]).sort_index()
    daily_err = allerr.groupby((allerr.index - pd.Timedelta(hours=7)).normalize()).mean()
    hol = USFederalHolidayCalendar().holidays("2026-01-01", "2027-12-31")
    rows = []
    for dday, o in h.groupby("buy_day"):
        prior = daily_err[(daily_err.index <= dday - pd.Timedelta(days=2)) & (daily_err.index > dday - pd.Timedelta(days=9))]
        x = dict(width=(o.dem_raw_P95 - o.dem_raw_P50).mean(), disagree=(o.dem_gbm_P50 - o.dem_lstm_P50).abs().mean(),
                 temp_max=o.temp_fc.max(), temp_min=o.temp_fc.min(), temp_swing=(o.temp_fc - o.temp_lag).abs().mean(),
                 cloud=o.cloud_fc.mean(), recent_err=prior.mean() if len(prior) >= 3 else np.nan, data_flags=0.0,
                 weekday=dday.dayofweek, month=dday.month, holiday=int(dday in hol),
                 after_holiday=int((dday - pd.Timedelta(days=1)) in hol))
        p = float(rk["model"].predict_proba(pd.DataFrame([x])[rk["features"]])[:, 1][0])
        tier = "Red" if p >= rk["yellow_max"] else ("Yellow" if p >= rk["green_max"] else "Green")
        rows.append(dict(buy_day=dday.date(), risk=round(p, 3), tier=tier, buy_level="P95" if tier == "Green" else "P98", **x))
    days = pd.DataFrame(rows).set_index("buy_day")

    # --- hourly output with the recommended buy
    h["tier"] = h.buy_day.dt.date.map(days.tier)
    h["buy_level"] = h.buy_day.dt.date.map(days.buy_level)
    h["buy_net_MWh"] = np.where(h.buy_level == "P98", h.net_P98, h.net_P95)
    h["buy_demand_MWh"] = np.where(h.buy_level == "P98", h.dem_P98, h.dem_P95)
    h["pace_dayahead"] = df["Day-Ahead Demand Forecast (MWh)"].reindex(h.index)
    h["ren_actual"] = df[R].reindex(h.index)
    h["issued"] = raw["issued"].reindex(h.index)
    h.index.name = "timestamp_utc"

    # --- wholesale prices (CAISO EDAM, PACE): settle each finished day in dollars. Never blocks the forecast.
    try:
        import prices
        P = prices.update(now_utc)
        h = h.join(P[["da", "rt"]].rename(columns={"da": "price_da", "rt": "price_rt"}))
    except Exception as e:
        print(f"  prices unavailable this run ({e.__class__.__name__}); track record without dollars", flush=True)
        h["price_da"], h["price_rt"] = np.nan, np.nan
    h["net_actual"] = h["dem_actual"] - h["ren_actual"]
    h["pace_net_buy"] = h["pace_dayahead"] - h["ren_actual"].shift(24)

    # --- track record: finished live days
    tr = []
    for dday, o in h[h.buy_day < target_day].groupby("buy_day"):
        k = o.dropna(subset=["dem_actual", "buy_demand_MWh"])
        if len(k) < 20:
            continue
        a = k.dem_actual
        sh, su = (a - k.buy_demand_MWh).clip(lower=0), (k.buy_demand_MWh - a).clip(lower=0)
        kp = k.dropna(subset=["pace_dayahead"])
        kd = k.dropna(subset=["price_da", "price_rt", "net_actual", "buy_net_MWh"])
        dollars = {}
        if len(kd) >= 20:   # extra cost vs perfect foresight = (RT - DA) x (actual - buy), net load basis
            sp = kd.price_rt - kd.price_da
            dollars = dict(price_da_avg=round(kd.price_da.mean(), 2), price_rt_avg=round(kd.price_rt.mean(), 2),
                           maya_extra_cost=round((sp * (kd.net_actual - kd.buy_net_MWh)).sum(), 0),
                           maya_short_cost=round((sp * (kd.net_actual - kd.buy_net_MWh).clip(lower=0)).sum(), 0))
            kq = kd.dropna(subset=["pace_net_buy"])
            if len(kq) >= 20:
                dollars["pace_extra_cost"] = round(((kq.price_rt - kq.price_da) * (kq.net_actual - kq.pace_net_buy)).sum(), 0)
        tr.append(dict(buy_day=dday.date(), tier=o.tier.iloc[0], buy_level=o.buy_level.iloc[0], hours=len(k),
                       short_hours=int((sh > 0).sum()), short_MWh=round(sh.sum(), 0), surplus_MWh=round(su.sum(), 0),
                       worst_short_MWh=round(sh.max(), 0), maya_P50_MAE=round((a - k.dem_P50).abs().mean(), 1),
                       pace_MAE=round((kp.dem_actual - kp.pace_dayahead).abs().mean(), 1) if len(kp) else np.nan,
                       pace_short_hours=int((kp.dem_actual > kp.pace_dayahead).sum()) if len(kp) else np.nan, **dollars))
    track = pd.DataFrame(tr)

    # --- write
    tgt_rows = h[h.buy_day == target_day]
    summary = dict(run_utc=str(now_utc), buy_day=str(target_day.date()), provisional=provisional,
                   newest_demand_hour_utc=str(d_eia.index.max()), buy_deadline_utc=str(target_day - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR_UTC)),
                   tier=days.loc[target_day.date(), "tier"], risk=float(days.loc[target_day.date(), "risk"]),
                   buy_level=days.loc[target_day.date(), "buy_level"],
                   buy_net_MWh_total=round(float(tgt_rows.buy_net_MWh.sum()), 0),
                   buy_demand_MWh_total=round(float(tgt_rows.buy_demand_MWh.sum()), 0),
                   p50_demand_MWh_total=round(float(tgt_rows.dem_P50.sum()), 0),
                   hours=int(tgt_rows.buy_net_MWh.notna().sum()), live_days_scored=int(len(track)),
                   weather="ECMWF IFS 0.25 (Open-Meteo), latest run for the buy day",
                   note=("Provisional: run before the buy-time demand reading was published; the official run is after 16:15 UTC."
                         if provisional else ""))
    h.reset_index().round(2).to_csv(os.path.join(LIVE, "live_hourly.csv"), index=False)
    days.round(4).to_csv(os.path.join(LIVE, "live_days.csv"))
    track.to_csv(os.path.join(LIVE, "track_record.csv"), index=False)
    tgt_rows.reset_index().round(1).to_csv(os.path.join(LIVE, "next_buy_day.csv"), index=False)
    drivers.to_csv(os.path.join(LIVE, "drivers.csv"))
    json.dump(summary, open(os.path.join(LIVE, "summary.json"), "w"), indent=2)
    # --- strategies: hourly spike risk, bid ladder for the buy day, and the scored strategy history
    try:
        import strategies
        st_s, _, st_res, _ = strategies.build()
        nxt = st_s[st_s.index.isin(tgt_rows.index)]
        summary["ladder"] = dict(block1_MWh=round(float(nxt.ladder_block1.sum()), 0), block2_MWh=round(float(nxt.ladder_block2.sum()), 0),
                                 high_risk_hours=int(nxt.high_risk_hour.sum()))
        json.dump(summary, open(os.path.join(LIVE, "summary.json"), "w"), indent=2)
    except Exception as e:
        print(f"  strategies step skipped ({e.__class__.__name__}: {e})", flush=True)
    print(json.dumps(summary, indent=2))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(0 if verify() else 1)
    run(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None)
