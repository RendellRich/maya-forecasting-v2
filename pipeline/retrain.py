"""
Maya v2 monthly retrain (champion / challenger).

Why: the production models were trained on data through 2025 and then frozen. Daily recalibration keeps the buy
levels honest as conditions change, but the underlying models do not learn new patterns. Once a month this script
checks whether a model trained on more recent data would do better, and only switches if it clearly does.

How
  1. Training data = master dataset (2019, 2021 - Sep 1 2026; 2020 excluded) + the live history saved by each daily
     run (EIA demand, weather, Milford renewable estimate).
  2. Holdout = the most recent HOLDOUT_DAYS days with actuals. Neither model has seen them.
  3. Challenger = same model design (quantile GBM + quantile LSTM for demand, quantile GBM for renewable), trained
     on everything up to 2 days before the holdout.
  4. Both champion (current production) and challenger forecast the holdout with day-ahead weather (as issued),
     scored by pinball loss averaged over the quantile levels (lower is better).
  5. If the challenger beats the champion by at least MIN_GAIN on demand without being worse on renewable, it is
     refit on all data through 2 days ago and promoted. Otherwise nothing changes.
Past live forecasts are never recomputed (live/forecast_archive.parquet), so a promoted model cannot rewrite the
track record.

Usage: python retrain.py            check and promote if better
       python retrain.py --dry-run  check only
Outputs: live/retrain_log.csv (one row per check), models/lag48/* and model_version.txt when promoted.
"""
import json
import os
import shutil
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_forecast as L
from maya_common import QD, QR, qn

HOLDOUT_DAYS, MIN_GAIN, LSTM_EPOCHS = 45, 0.01, 11
GBM_ARGS = dict(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42)
WX = ["Temperature (C)", "GHI (W/m2)", "DNI (W/m2)", "DHI (W/m2)", "Cloud Cover (%)", "Wind Speed 10m (m/s)", "Wind Speed 100m (m/s)"]
LOG = os.path.join(L.LIVE, "retrain_log.csv")


def assemble(v3=False):
    """Hourly history (actuals + actual-weather stand-ins) and the day-ahead forecast weather for every hour."""
    m, fc_bt = L.load_master()
    hl = pd.read_parquet(os.path.join(L.LIVE, "history_live.parquet"))
    base = [c for c in hl.columns if not c.startswith("fc_")]
    df = pd.concat([m, hl[base][hl.index > m.index.max()]]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(idx)
    fcw = fc_bt.reindex(idx)
    live_fc = hl[[c for c in hl.columns if c.startswith("fc_")]].rename(columns=lambda c: c[3:])
    later = idx > fc_bt.index.max()
    for c in live_fc.columns:
        fcw.loc[later, c] = live_fc[c].reindex(idx[later]).values
    if v3:   # city weather: ERA5 + archived day-ahead forecasts for history, live history after
        from maya_common import CITY_COLS, city_inputs
        era = L.update_city_era5()
        cfc = pd.read_csv(os.path.join(L.DATA, "cities_ecmwf_dayahead_2024_2026.csv.gz"), parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
        raw_fc = cfc
        live_raw = os.path.join(L.DATA, "city_weather_live_prev1.csv")
        if os.path.exists(live_raw):     # live day-ahead forecasts after the archive ends
            lr = pd.read_csv(live_raw, parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
            raw_fc = pd.concat([cfc, lr[lr.index > cfc.index.max()]])
        lat = os.path.join(L.DATA, "city_weather_live_latest.csv")
        a, f = city_inputs(era, raw_fc, idx, pd.read_csv(lat, parse_dates=["timestamp_utc"]).set_index("timestamp_utc") if os.path.exists(lat) else None)
        for c in CITY_COLS:
            df[c], fcw[c] = a[c].values, f[c].values
    return df, fcw


def pinball(y, pred, q):
    e = y - pred
    return np.nanmean(np.maximum(q * e, (q - 1) * e))


def fit(df, fcw, end, v3=False):
    """Train demand (GBM + LSTM quantiles) and renewable (GBM quantiles) on data before `end`."""
    year = df.index.year.values
    from maya_common import CITY_COLS, GBM_UPPER
    wx = WX + (CITY_COLS if v3 else [])
    F_act = L.build_features(df, df[wx], v3=v3)        # training inputs: actual weather (as for the production models)
    y = df[L.D].astype(float)
    G, now, seq_df, seq_ok = F_act["G"], F_act["now"], F_act["seq_df"], F_act["seq_ok"]
    tr = np.where(y.notna().values & G.notna().all(axis=1).values & seq_ok & (year != 2020) & (df.index < end))[0]
    gbms = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **(GBM_UPPER if v3 and q > 0.5 else GBM_ARGS))
            .fit(G.iloc[tr], y.values[tr]) for q in QD}
    sc = dict(mu_s=seq_df.iloc[tr].mean(), sd_s=seq_df.iloc[tr].std().replace(0, 1),
              mu_n=now.iloc[tr].mean(), sd_n=now.iloc[tr].std().replace(0, 1))
    S = torch.tensor(((seq_df - sc["mu_s"]) / sc["sd_s"]).fillna(0).values, dtype=torch.float32)
    N = torch.tensor(((now - sc["mu_n"]) / sc["sd_n"]).fillna(0).values, dtype=torch.float32)
    mu, sd = float(sc["mu_s"]["y"]), float(sc["sd_s"]["y"])
    yt = torch.tensor((y.values - mu) / sd, dtype=torch.float32)
    QT = torch.tensor(QD, dtype=torch.float32)
    torch.manual_seed(42)
    net = L.QNet(S.shape[1], N.shape[1], len(QD))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    off = torch.arange(-L.WIN + 1, 1)
    for ep in range(LSTM_EPOCHS):
        net.train()
        perm = np.random.RandomState(ep).permutation(tr)
        for b in range(0, len(perm), 256):
            bi = perm[b:b + 256]
            pred = net(S[(torch.tensor(bi - L.MIN_LAG)[:, None] + off[None, :])], N[bi])
            e = yt[bi][:, None] - pred
            loss = torch.maximum(QT * e, (QT - 1) * e).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    ck = {"state_dict": net.state_dict(), "quantiles": QD, "n_seq": S.shape[1], "n_now": N.shape[1],
          "scaler": {k: v.to_dict() for k, v in sc.items()}, "seq_cols": list(seq_df.columns), "now_cols": list(now.columns),
          "window": L.WIN, "gap": L.MIN_LAG}
    # renewable: trained on day-ahead forecast weather, as in production (ECMWF archive starts Feb 2024)
    F_fc = L.build_features(df, fcw, v3=v3)
    H = F_fc["H"]
    yr = df[L.R].astype(float)
    need = [c for c in H.columns if c != "Clear-sky Index"]
    rm = H[need].notna().all(axis=1).values & yr.notna().values & (df.index >= "2024-02-04") & (df.index < end)
    ren = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **GBM_ARGS).fit(H[rm], yr.values[rm]) for q in QR}
    return gbms, net, sc, ck, ren


def score(models, F_fc, df, mask):
    p = L.predict(F_fc, mask, models)
    y, yr = df[L.D].reindex(p.index), df[L.R].reindex(p.index)
    dem = np.mean([pinball(y.values, p[f"dem_raw_{qn(q)}"].values, q) for q in QD])
    ren = np.mean([pinball(yr.values, p[f"ren_raw_{qn(q)}"].values, q) for q in QR])
    return dem, ren, int(p["dem_raw_P50"].notna().sum())


def save(models, version):
    gbms, net, sc, ck, ren = models
    if not os.path.isdir(L.ORIGINAL):        # keep the backtest's models for --verify
        os.makedirs(L.ORIGINAL)
        for f in ["demand_gbm_quantiles.joblib", "demand_lstm_quantiles.pt", "renewable_gbm_quantiles.joblib"]:
            shutil.copy(os.path.join(L.MOD, f), os.path.join(L.ORIGINAL, f))
    stamp = time.strftime("%Y%m%d")
    bak = os.path.join(L.MOD, f"previous_{stamp}")
    os.makedirs(bak, exist_ok=True)
    for f in ["demand_gbm_quantiles.joblib", "demand_lstm_quantiles.pt", "renewable_gbm_quantiles.joblib", "model_version.txt"]:
        if os.path.exists(os.path.join(L.MOD, f)):
            shutil.copy(os.path.join(L.MOD, f), os.path.join(bak, f))
    joblib.dump(gbms, os.path.join(L.MOD, "demand_gbm_quantiles.joblib"))
    torch.save(ck, os.path.join(L.MOD, "demand_lstm_quantiles.pt"))
    joblib.dump(ren, os.path.join(L.MOD, "renewable_gbm_quantiles.joblib"))
    open(os.path.join(L.MOD, "model_version.txt"), "w").write(version)


def main(dry_run=False):
    t0 = time.time()
    champ = L.load_models()
    v3 = "dow" in champ[0][0.5].feature_names_in_
    df, fcw = assemble(v3)
    last = df[L.D].last_valid_index()
    hold_end = last.normalize()
    hold_start = hold_end - pd.Timedelta(days=HOLDOUT_DAYS)
    train_end = hold_start - pd.Timedelta(days=2)
    F_fc = L.build_features(df, fcw, v3=v3)
    mask = np.asarray((df.index >= hold_start) & (df.index < hold_end))
    print(f"holdout {hold_start.date()} .. {hold_end.date()} | challenger trained on data before {train_end.date()}", flush=True)
    c_dem, c_ren, n = score(champ, F_fc, df, mask)
    chal = fit(df, fcw, train_end, v3)
    ch_models = (chal[0], chal[1], {k: pd.Series(v) for k, v in chal[3]["scaler"].items()}, chal[3], chal[4])
    h_dem, h_ren, _ = score(ch_models, F_fc, df, mask)
    gain = (c_dem - h_dem) / c_dem
    promote = bool(gain >= MIN_GAIN and h_ren <= c_ren * 1.02 and not dry_run)
    row = dict(checked_utc=pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M"), holdout_start=str(hold_start.date()),
               holdout_end=str(hold_end.date()), holdout_hours=n, champion=L.MODEL_VERSION,
               champion_demand_pinball=round(c_dem, 2), challenger_demand_pinball=round(h_dem, 2), demand_gain_pct=round(gain * 100, 2),
               champion_renewable_pinball=round(c_ren, 3), challenger_renewable_pinball=round(h_ren, 3),
               decision="promoted" if promote else ("dry run" if dry_run else "kept current model"))
    if promote:
        final_end = last.normalize() - pd.Timedelta(days=1)
        version = f"{'v3' if v3 else 'v2'}.{pd.Timestamp.now('UTC'):%Y%m%d} (retrained through {final_end.date()})"
        save(fit(df, fcw, final_end, v3), version)
        row["new_model"] = version
    log = pd.concat([pd.read_csv(LOG), pd.DataFrame([row])]) if os.path.exists(LOG) else pd.DataFrame([row])
    log.to_csv(LOG, index=False)
    print(json.dumps(row, indent=2), f"\ndone in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
