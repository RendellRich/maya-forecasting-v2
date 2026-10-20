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
