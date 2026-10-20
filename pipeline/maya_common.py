"""Shared helpers for Maya v2: adaptive calibration, level names, and the Milford renewable physics model."""
import numpy as np
import pandas as pd

WINDOW, LAG, MIN_DAYS, GAMMA = 60, 2, 14, 0.02
QD = [0.50, 0.75, 0.90, 0.95, 0.98]          # demand / net load: upper levels
QR = [0.02, 0.05, 0.10, 0.25, 0.50]          # renewable: lower levels


def qn(q):
    return f"P{int(round(q * 100)):02d}"


def calibrate(base, actual, q, upper=True, window=WINDOW, gamma=GAMMA, lag=LAG, min_days=MIN_DAYS, return_state=False):
    """Walk-forward adaptive calibration (adaptive conformal inference, daily steps).
    Shift for day d = the lvl-th quantile of (actual - base) over the last `window` finished days (ending d-lag).
    lvl starts at q and is nudged after each finished day: up if that day missed more often than the target,
    down if less. Upper levels (demand, net load) miss when actual > forecast; lower levels (renewable) when
    actual < forecast. Uses only what is known on the morning of the day before the target day.
    Days are UTC calendar days of the index."""
    day = base.index.normalize()
    days = np.array(sorted(day.unique()))
    score = actual.reindex(base.index) - base
    sc_d = {d: s.dropna().values for d, s in score.groupby(day)}
    target = (1 - q) if upper else q
    lvl, shifts = q, {}
    for i, d in enumerate(days):
        if i - lag >= 0 and days[i - lag] in shifts:
            s = sc_d.get(days[i - lag], np.array([]))
            if len(s):
                miss = (s > shifts[days[i - lag]]).mean() if upper else (s < shifts[days[i - lag]]).mean()
                lvl = float(np.clip(lvl + (gamma if upper else -gamma) * (miss - target), 0.001, 0.999))
        hist = [sc_d[days[j]] for j in range(max(0, i - window - lag + 1), i - lag + 1) if len(sc_d.get(days[j], []))]
        shifts[d] = np.quantile(np.concatenate(hist), lvl) if len(hist) >= min_days else 0.0
    out = base + pd.Series(day.map(shifts).values, index=base.index).astype(float)
    return (out, lvl) if return_state else out


# ------------------------------- renewable physics (same as build_renewable_target_v3.py) -------------
SITE_LAT, SITE_LON, ALT = 38.6411, -112.8488, 1550
GCR, MAX_ANGLE, DC_AC, INV_EFF, PV_GAMMA, NOCT, ALBEDO = 0.40, 60, 1.25, 0.96, -0.004, 45.0, 0.25


def solar_physics(index, ghi, dni, dhi, temp, ac_mw=476.1):
    """Uncalibrated hourly AC output (MWh) of the Milford-area solar fleet from hour-ending weather."""
    import pvlib
    times = pd.DatetimeIndex(index - pd.Timedelta(minutes=30)).tz_localize("UTC")
    loc = pvlib.location.Location(SITE_LAT, SITE_LON, tz="UTC", altitude=ALT)
    sp = loc.get_solarposition(times)
    tr = pvlib.tracking.singleaxis(sp.apparent_zenith, sp.azimuth, axis_tilt=0, axis_azimuth=180,
                                   max_angle=MAX_ANGLE, backtrack=True, gcr=GCR)
    poa = pvlib.irradiance.get_total_irradiance(
        tr.surface_tilt.fillna(0), tr.surface_azimuth.fillna(180), sp.apparent_zenith, sp.azimuth,
        dni=np.clip(np.asarray(dni, float), 0, None), ghi=np.clip(np.asarray(ghi, float), 0, None),
        dhi=np.clip(np.asarray(dhi, float), 0, None), dni_extra=pvlib.irradiance.get_extra_radiation(times),
        model="haydavies", albedo=ALBEDO)["poa_global"].fillna(0).clip(lower=0).values
    t_cell = np.asarray(temp, float) + poa / 800.0 * (NOCT - 20.0)
    ac = np.where(np.asarray(index.year) <= 2020, 377.1, ac_mw)
    p_dc = np.clip(ac * DC_AC * poa / 1000.0 * (1 + PV_GAMMA * (t_cell - 25.0)), 0, None)
    return pd.Series(np.minimum(p_dc * INV_EFF, ac), index=index)


# ------------------------------- v3 inputs: calendar, city weather, tuned upper quantiles --------------
CAL_EXTRA = ["dow", "holiday", "bridge", "school_out"]
CITY_COLS = ["pw_temp"]      # dew point and Salt Lake City alone were tested and added nothing
GBM_UPPER = dict(max_iter=300, max_depth=4, learning_rate=0.03, min_samples_leaf=20, l2_regularization=1.0, random_state=42)


def calendar_extra(idx):
    """Weekday, holidays (federal + Utah Pioneer Day, Christmas Eve, day after Thanksgiving), bridge days and the
    Utah school calendar (summer break Jun 5 - Aug 20, winter break Dec 22 - Jan 2), on the Mountain Standard clock."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    loc = idx - pd.Timedelta(hours=7)
    d = loc.normalize()
    fed = USFederalHolidayCalendar().holidays("2018-01-01", "2030-12-31")
    yrs = range(2018, 2031)
    extra = [pd.Timestamp(f"{y}-07-24") for y in yrs] + [pd.Timestamp(f"{y}-12-24") for y in yrs] + \
            [h + pd.Timedelta(days=1) for h in fed if h.month == 11 and h.day >= 22]
    hol = fed.union(pd.DatetimeIndex(extra))
    e = pd.DataFrame(index=idx)
    e["dow"] = loc.dayofweek
    e["holiday"] = d.isin(hol).astype(int)
    e["bridge"] = (d.isin(hol - pd.Timedelta(days=1)) | d.isin(hol + pd.Timedelta(days=1))).astype(int)
    md = loc.month * 100 + loc.day
    e["school_out"] = (((md >= 605) & (md <= 820)) | (md >= 1222) | (md <= 102)).astype(int)
    return e


def city_features(raw):
    """Population-weighted temperature and dew point over PACE load centers (cities.py), plus Salt Lake City alone.
    `raw` has columns '<key>_temperature_2m' and '<key>_dew_point_2m'."""
    from cities import CITIES
    w = pd.Series({k: v["w"] for k, v in CITIES.items()})
    w = w / w.sum()
    t = raw[[f"{k}_temperature_2m" for k in w.index]].values
    dp = raw[[f"{k}_dew_point_2m" for k in w.index]].values
    out = pd.DataFrame(index=raw.index)
    out["pw_temp"] = t @ w.values
    out["pw_dew"] = dp @ w.values
    out["slc_temp"] = raw["slc_temperature_2m"].values
    return out


BIAS_WIN_DAYS, BIAS_DELAY_DAYS = 60, 8


def forecast_bias(fc, act, win=BIAS_WIN_DAYS, delay=BIAS_DELAY_DAYS):
    """Trailing bias of the ECMWF day-ahead city forecast vs ERA5, by hour of day: mean of (forecast - ERA5) over the
    `win` days ending `delay` days before each hour (ERA5 is published ~5 days late, so this is known at buy time).
    ECMWF runs ~0.9 C colder than ERA5 at these points; training uses ERA5, so the forecast is shifted to match."""
    d = (fc - act)
    out = pd.DataFrame(np.nan, index=d.index, columns=d.columns)
    hrs = d.index.hour
    for h in range(24):
        sel = hrs == h
        out[sel] = d[sel].rolling(win, min_periods=20).mean().shift(delay).values
    return out.ffill().fillna(0.0)


def city_inputs(era_raw, fc_raw, idx, latest_raw=None):
    """Population-weighted temperature (CITY_COLS) for every hour of `idx`:
    act = ERA5, gaps (ERA5 runs ~5 days behind) filled with the bias-corrected latest run, then the corrected forecast;
    fc  = ECMWF day-ahead forecast minus its trailing bias. `idx` must be a continuous hourly index."""
    a = city_features(era_raw).reindex(idx)[CITY_COLS]
    f = city_features(fc_raw).reindex(idx)[CITY_COLS]
    b = forecast_bias(f, a)
    f = f - b
    if latest_raw is not None:
        a = a.fillna(city_features(latest_raw).reindex(idx)[CITY_COLS] - b)
    return a.fillna(f), f
