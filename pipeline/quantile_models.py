"""
Maya v2: buy-ahead (quantile) forecasts.  Built Oct 19-20 from final_models.py (same features, same periods).

Why: Maui Hiva buys power the day before. Buying short means paying spot prices on the day, which costs far more
than carrying a small surplus. So instead of one middle-of-the-road number, Maya v2 gives a range per hour:

  demand and net load : P50 (best guess) and upper levels P75 / P90 / P95 / P98  (buy-ahead numbers)
  renewable output    : P50 and lower levels  P25 / P10 / P05 / P02              (output you can count on)

Method
  1. Quantile GBM per level (HistGradientBoostingRegressor, loss="quantile").
  2. Demand LSTM with one output per level, trained on pinball (quantile) loss.  Demand level = mean of GBM and LSTM.
  3. Rolling calibration: for every target day, look at the last 60 days of finished errors (ending 2 days earlier, so
     every actual was known when the forecast was made) and shift each level so it really is under that often.
     A "P95" then means the actual comes in above it on about 5% of hours, on recent evidence.

All forecasts are day-ahead and out-of-sample, same periods as final_models.py (lags set by MIN_LAG below):
  A 2024-02-04 .. 2025-12-31 backtest   (demand trained 2019, 2021-23; renewable 2024 on reanalysis, 2025 on 2024 ECMWF)
  B 2026-01-01 .. last demand hour       (demand trained 2019, 2021-25; renewable on ECMWF Feb 2024 - Dec 2025)

Inputs : data/master_hourly_cleaned_ALL_YEARS_v2.csv, data/ecmwf_dayahead_forecast_2024-2026.csv
Outputs: out/quantile_forecasts_raw.parquet (before calibration), then calibrate_and_score.py does the rest.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(os.cpu_count() or 2)
HERE = os.path.dirname(os.path.abspath(__file__))
# MIN_LAG: newest demand hour the model may use, in hours before the target hour.
#   24 = v1 setting (yesterday, same hour). Not all of those hours are known when a day-ahead buy is placed.
#   48 = honest buy-ahead setting: on the morning of the day before, every hour of the target day still has
#        demand from two days earlier available (John's 48 h rule). Default.
MIN_LAG = int(os.environ.get("MAYA_MIN_LAG", "48"))
DATA = os.path.join(HERE, "data")
# MAYA_FEATURES: v2 = production inputs as of Oct 20; v3 = adds weekday / holiday / school calendar, population-weighted
# city weather (cities.py) and tuned settings for the upper quantile GBMs (time-series CV on 2022-2023).
FEATURES = os.environ.get("MAYA_FEATURES", "v2")
SUFFIX = "" if FEATURES == "v2" else f"_{FEATURES}"
OUT, MOD = os.path.join(HERE, "out", f"lag{MIN_LAG}{SUFFIX}"), os.path.join(HERE, "models", f"lag{MIN_LAG}{SUFFIX}")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MOD, exist_ok=True)

WIN, GAP, LSTM_EPOCHS = 168, MIN_LAG, 11
QD = [0.50, 0.75, 0.90, 0.95, 0.98]          # demand: upper levels
QR = [0.02, 0.05, 0.10, 0.25, 0.50]          # renewable: lower levels
GBM_ARGS = dict(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42)

# ------------------------------- features (identical to final_models.py) ---------------------------
df = pd.read_csv(os.path.join(DATA, "master_hourly_cleaned_ALL_YEARS_v2.csv"))
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"]).dt.tz_localize(None)
df = df.set_index("timestamp_utc").sort_index()
fc = pd.read_csv(os.path.join(DATA, "ecmwf_dayahead_forecast_2024-2026.csv"),
                 parse_dates=["timestamp_utc"]).set_index("timestamp_utc").reindex(df.index)
year, ts = df.index.year.values, df.index

df["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
df["doy_sin"] = np.sin(2 * np.pi * ts.dayofyear / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * ts.dayofyear / 365.25)
df["month"] = ts.month
df["is_weekend"] = (ts.dayofweek >= 5).astype(int)
df["Installed Solar DC (MW)"] = np.where(year <= 2020, 377.1, 476.1) * 1.25
cs = df["Clear-sky GHI (W/m2)"]
df["Clear-sky Index"] = np.where(cs >= 20, (df["GHI (W/m2)"] / cs).clip(0, 1.5), np.nan)
fc["Wind Speed 80m (m/s)"] = fc["Wind Speed 10m (m/s)"].clip(lower=0.01) * 8 ** (
    np.log(fc["Wind Speed 100m (m/s)"].clip(lower=0.01) / fc["Wind Speed 10m (m/s)"].clip(lower=0.01)) / np.log(10))
fc["Clear-sky Index"] = np.where(cs >= 20, (fc["GHI (W/m2)"] / cs).clip(0, 1.5), np.nan)
fc["Installed Solar DC (MW)"] = df["Installed Solar DC (MW)"]
CAL = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month", "is_weekend"]
CAL_D = list(CAL)
TEMP_COL = "Temperature (C)"
if FEATURES == "v3":
    from maya_common import CAL_EXTRA, CITY_COLS, GBM_UPPER, calendar_extra, city_inputs
    df = df.join(calendar_extra(df.index))
    era = pd.read_csv(os.path.join(DATA, "cities_era5_2019_2026.csv.gz"), parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    cfc = pd.read_csv(os.path.join(DATA, "cities_ecmwf_dayahead_2024_2026.csv.gz"), parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    a, f = city_inputs(era, cfc, df.index)
    for c in CITY_COLS:
        df[c], fc[c] = a[c], f[c]
    CAL_D = CAL + CAL_EXTRA

D, R = "Demand (MWh)", "Site Renewable Output Modeled (MWh)"
WX_D = ["GHI (W/m2)", "Temperature (C)", "Wind Speed 10m (m/s)", "Wind Speed 100m (m/s)", "Wind Speed 80m (m/s)"]
SEQ_D = ["Temperature (C)", "GHI (W/m2)"]
if FEATURES == "v3":
    WX_D = WX_D + CITY_COLS
    SEQ_D = SEQ_D + ["pw_temp"]
WX_R = ["GHI (W/m2)", "Temperature (C)", "DNI (W/m2)", "DHI (W/m2)", "Installed Solar DC (MW)", "Cloud Cover (%)", "Clear-sky Index"]

last_demand = df[D].last_valid_index()
A = (ts >= "2024-02-04") & (year <= 2025)
B = (year == 2026) & (ts <= last_demand)


def lags(y):
    o = pd.DataFrame(index=y.index)
    L = MIN_LAG
    o[f"lag{L}"], o[f"lag{L + 24}"], o["lag168"] = y.shift(L), y.shift(L + 24), y.shift(168)
    o["roll7"] = pd.concat([y.shift(24 * k) for k in range(L // 24, L // 24 + 7)], axis=1).mean(axis=1)
    return o


def with_fc(frame, cols):
    out = frame.copy()
    for c in cols:
        out[c] = fc[c]
    return out


def qname(q):
    return f"P{int(round(q * 100)):02d}"


# ------------------------------- DEMAND ------------------------------------------------------------
y = df[D].astype(float)


def buy_time_features(y, temp_target):
    """Extra inputs that are known when the day-ahead buy is placed (only used when MIN_LAG = 48).
    Buy placed at ISSUE_HOUR_UTC on the Mountain-local day before the target (local day = UTC-7);
    newest demand reading then = 2 h before the buy (EIA posts PACE demand within about 1-2 h)."""
    local_day = (ts - pd.Timedelta(hours=7)).normalize()
    last_known = local_day - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR_UTC - 2)
    e = pd.DataFrame(index=ts)
    e["last_known"] = y.reindex(last_known).values                                    # newest reading at buy time
    e["last_known_change_1d"] = e["last_known"] - y.reindex(last_known - pd.Timedelta(days=1)).values
    e["hours_since_known"] = (ts - last_known) / pd.Timedelta(hours=1)
    e["same_hour_latest"] = np.where(ts - pd.Timedelta(hours=24) <= last_known, y.shift(24).values, y.shift(48).values)
    e["temp_change_vs_lag"] = temp_target - df[TEMP_COL].shift(MIN_LAG)       # target temp vs. temp at lag hour
    return e


ISSUE_HOUR_UTC = 16          # 10 am MDT / 9 am MST
G_act = pd.concat([df[WX_D + CAL_D], lags(y)], axis=1)
G_fc = with_fc(G_act, WX_D)
now_act = df[WX_D + CAL_D]
now_fc = with_fc(now_act, WX_D)
if MIN_LAG >= 48:
    e_act, e_fc = buy_time_features(y, df[TEMP_COL]), buy_time_features(y, fc[TEMP_COL])
    G_act, G_fc = pd.concat([G_act, e_act], axis=1), pd.concat([G_fc, e_fc], axis=1)
    now_act, now_fc = pd.concat([now_act, e_act], axis=1), pd.concat([now_fc, e_fc], axis=1)
seq_df = pd.concat([y.rename("y"), df[SEQ_D], df[["hour_sin", "hour_cos", "doy_sin", "doy_cos"]]], axis=1)
seq_ok = seq_df.notna().all(axis=1).astype(int).rolling(WIN).sum().shift(GAP).fillna(0).values == WIN
train_ok = y.notna().values & G_act.notna().all(axis=1).values & seq_ok
pred_ok = G_fc.notna().all(axis=1).values & seq_ok
QT = torch.tensor(QD, dtype=torch.float32)


class QNet(nn.Module):
    """Same LSTM as production, but the head returns one value per quantile level.
    Levels are built as P50 + cumulative positive steps, so they can never cross."""
    def __init__(self, n_seq, n_now, nq, hidden=48):
        super().__init__()
        self.lstm = nn.LSTM(n_seq, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_now, 64), nn.ReLU(), nn.Linear(64, nq))

    def forward(self, seq, now):
        _, (h, _) = self.lstm(seq)
        z = self.head(torch.cat([h[-1], now], dim=1))
        return torch.cat([z[:, :1], z[:, :1] + torch.cumsum(nn.functional.softplus(z[:, 1:]), dim=1)], dim=1)


def pinball(pred, target):
    e = target[:, None] - pred
    return torch.maximum(QT * e, (QT - 1) * e).mean()


def fit_demand(train_years, tag):
    tr = np.where(train_ok & np.isin(year, train_years))[0]
    gbms = {}
    for q in QD:
        t0 = time.time()
        gbms[q] = HistGradientBoostingRegressor(loss="quantile", quantile=q, **(GBM_UPPER if FEATURES == "v3" and q > 0.5 else GBM_ARGS)).fit(G_act.iloc[tr], y.values[tr])
        print(f"  demand GBM {tag} {qname(q)} {time.time() - t0:.0f}s", flush=True)
    sc = dict(mu_s=seq_df.iloc[tr].mean(), sd_s=seq_df.iloc[tr].std().replace(0, 1),
              mu_n=now_act.iloc[tr].mean(), sd_n=now_act.iloc[tr].std().replace(0, 1))
    S_arr = torch.tensor(((seq_df - sc["mu_s"]) / sc["sd_s"]).fillna(0).values, dtype=torch.float32)
    N_act = torch.tensor(((now_act - sc["mu_n"]) / sc["sd_n"]).fillna(0).values, dtype=torch.float32)
    mu, sd = float(sc["mu_s"]["y"]), float(sc["sd_s"]["y"])
    yt = torch.tensor((y.values - mu) / sd, dtype=torch.float32)
    torch.manual_seed(42)
    net = QNet(S_arr.shape[1], N_act.shape[1], len(QD))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    off = torch.arange(-WIN + 1, 1)
    for ep in range(LSTM_EPOCHS):
        t0 = time.time()
        net.train()
        perm = np.random.RandomState(ep).permutation(tr)
        for b in range(0, len(perm), 256):
            bi = perm[b:b + 256]
            seq = S_arr[(torch.tensor(bi - GAP)[:, None] + off[None, :])]
            loss = pinball(net(seq, N_act[bi]), yt[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  demand LSTM {tag} epoch {ep + 1}/{LSTM_EPOCHS} {time.time() - t0:.0f}s", flush=True)
    return gbms, net, sc


def predict_demand(gbms, net, sc, mask):
    ids = np.where(mask & pred_ok)[0]
    S_arr = torch.tensor(((seq_df - sc["mu_s"]) / sc["sd_s"]).fillna(0).values, dtype=torch.float32)
    N_fc = torch.tensor(((now_fc - sc["mu_n"]) / sc["sd_n"]).fillna(0).values, dtype=torch.float32)
    mu, sd = float(sc["mu_s"]["y"]), float(sc["sd_s"]["y"])
    off = torch.arange(-WIN + 1, 1)
    net.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(ids), 2048):
            bi = ids[b:b + 2048]
            out.append(net(S_arr[(torch.tensor(bi - GAP)[:, None] + off[None, :])], N_fc[bi]).numpy())
    pl = np.concatenate(out) * sd + mu
    res = pd.DataFrame(index=ts[ids])
    for j, q in enumerate(QD):
        g = gbms[q].predict(G_fc.iloc[ids])
        res[f"dem_gbm_{qname(q)}"] = g
        res[f"dem_lstm_{qname(q)}"] = pl[:, j]
    gm = np.maximum.accumulate(res[[f"dem_gbm_{qname(q)}" for q in QD]].values, axis=1)  # GBM levels can cross; fix order
    for j, q in enumerate(QD):
        res[f"dem_gbm_{qname(q)}"] = gm[:, j]
        res[f"dem_raw_{qname(q)}"] = (gm[:, j] + pl[:, j]) / 2
    return res


print("Demand, period A (train 2019, 2021-2023)", flush=True)
gA, nA, sA = fit_demand([2019, 2021, 2022, 2023], "A")
dA = predict_demand(gA, nA, sA, A)
print("Demand, period B / production (train 2019, 2021-2025)", flush=True)
gB, nB, sB = fit_demand([2019, 2021, 2022, 2023, 2024, 2025], "B")
dB = predict_demand(gB, nB, sB, B)
joblib.dump(gB, os.path.join(MOD, "demand_gbm_quantiles.joblib"))
torch.save({"state_dict": nB.state_dict(), "quantiles": QD, "n_seq": len(seq_df.columns), "n_now": len(now_act.columns),
            "scaler": {k: v.to_dict() for k, v in sB.items()}, "seq_cols": list(seq_df.columns),
            "now_cols": list(now_act.columns), "window": WIN, "gap": GAP}, os.path.join(MOD, "demand_lstm_quantiles.pt"))

# ------------------------------- RENEWABLE ---------------------------------------------------------
yr = df[R].astype(float)
H_act = pd.concat([df[WX_R + CAL], lags(yr)], axis=1)
H_fc = with_fc(H_act, WX_R)
need = [c for c in H_act.columns if c != "Clear-sky Index"]
ok_act = H_act[need].notna().all(axis=1).values & yr.notna().values
ok_fc = H_fc[need].notna().all(axis=1).values
fc_hist = ok_fc & yr.notna().values & np.asarray(ts >= "2024-02-04")


def fit_r(X, mask):
    return {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **GBM_ARGS).fit(X[mask], yr.values[mask]) for q in QR}


print("Renewable quantile GBMs", flush=True)
r24 = fit_r(H_act, ok_act & np.isin(year, range(2019, 2024)))
r25 = fit_r(H_fc, fc_hist & (year == 2024))
r26 = fit_r(H_fc, fc_hist & np.isin(year, [2024, 2025]))
joblib.dump(r26, os.path.join(MOD, "renewable_gbm_quantiles.joblib"))
ren = pd.DataFrame(np.nan, index=ts, columns=[f"ren_raw_{qname(q)}" for q in QR])
for mdl, m in [(r24, A & (year == 2024)), (r25, A & (year == 2025)), (r26, B)]:
    ids = np.where(m & ok_fc)[0]
    p = np.column_stack([np.clip(mdl[q].predict(H_fc.iloc[ids]), 0, None) for q in QR])
    p = np.maximum.accumulate(p, axis=1)      # keep P02 <= P05 <= ... <= P50
    ren.iloc[ids] = p

# ------------------------------- ASSEMBLE ----------------------------------------------------------
out = pd.DataFrame(index=ts[A | B])
out["period"] = np.where(out.index.year == 2026, "B forward test 2026", "A backtest 2024-25")
out["dem_actual"] = df[D]
out["dem_pace"] = df["Day-Ahead Demand Forecast (MWh)"]
out = out.join(pd.concat([dA, dB]))
out["ren_actual"] = df[R]
out = out.join(ren)
out["temp_fc"] = fc["Temperature (C)"]
out["ghi_fc"] = fc["GHI (W/m2)"]
out["cloud_fc"] = fc["Cloud Cover (%)"]
out["temp_lag_actual"] = df["Temperature (C)"].shift(MIN_LAG)   # known at buy time
out["dem_lag_min"] = y.shift(MIN_LAG)
out["missing_grid"] = df["_missing_grid_data"].astype(float)
out["missing_weather"] = df["_missing_weather_data"].astype(float)
out.index.name = "timestamp_utc"
out.to_parquet(os.path.join(OUT, "quantile_forecasts_raw.parquet"))
print("Saved out/quantile_forecasts_raw.parquet", out.shape, flush=True)
