"""
PACE wholesale prices from CAISO OASIS (free, no key). PacifiCorp has traded in CAISO's Extended Day-Ahead
Market (EDAM) since it launched on May 1 2026, so these are the prices a PACE-area buyer actually faces.

  day-ahead : PRC_LMP,      market DAM,  hourly
  real-time : PRC_RTPD_LMP, market RTPD, 15-minute (fifteen-minute market), averaged to the hour
  node      : ELAP_PACE-APND (PACE default load aggregation point)
Timestamps are converted to UTC hour-ending, matching the EIA demand data.

Settlement for a buyer who buys B MWh day-ahead and uses A MWh:
  cost = DA x B + RT x (A - B)            (short: pay real-time for the gap; surplus: sold back at real-time)
  extra cost vs perfect foresight = (RT - DA) x (A - B)

Usage: python prices.py            -> updates data/pace_prices_2026.csv through today
"""
import io
import os
import time
import zipfile
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "data", "pace_prices_2026.csv")
NODE = "ELAP_PACE-APND"
URL = "https://oasis.caiso.com/oasisapi/SingleZip"
EDAM_START = pd.Timestamp("2026-05-01 07:00")


def _get(query, run, start, end):
    p = dict(queryname=query, market_run_id=run, node=NODE, resultformat=6, version=12 if query == "PRC_LMP" else 3,
             startdatetime=start.strftime("%Y%m%dT%H:%M-0000"), enddatetime=end.strftime("%Y%m%dT%H:%M-0000"))
    for i in range(4):
        try:
            r = requests.get(URL, params=p, timeout=120)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            name = z.namelist()[0]
            if name.endswith(".csv"):
                d = pd.read_csv(z.open(name))
                d = d[d.XML_DATA_ITEM == "LMP_PRC"]
                v = "MW" if "MW" in d.columns else "PRC"
                t = pd.to_datetime(d.INTERVALSTARTTIME_GMT, utc=True).dt.tz_localize(None).dt.floor("h") + pd.Timedelta(hours=1)
                return d.groupby(t)[v].mean()
        except Exception as e:  # network hiccup or OASIS rate limit
            print(f"  OASIS retry ({e.__class__.__name__})", flush=True)
        time.sleep(8 * (i + 1))
    return pd.Series(dtype=float)


def fetch(start, end):
    """Hourly DA and RT prices between two UTC timestamps. OASIS allows about one request every 5 s."""
    out = {}
    for col, query, run, step in [("da", "PRC_LMP", "DAM", 30), ("rt", "PRC_RTPD_LMP", "RTPD", 10)]:
        parts, s = [], start.floor("D") + pd.Timedelta(hours=7)
        while s < end:
            e = min(s + pd.Timedelta(days=step), end)
            parts.append(_get(query, run, s, e))
            s = e
            time.sleep(6)
        out[col] = pd.concat(parts) if parts else pd.Series(dtype=float)
    p = pd.DataFrame(out)
    p = p[~p.index.duplicated(keep="last")].sort_index()
    p["spread"] = p.rt - p.da
    p.index.name = "timestamp_utc"
    return p


def update(now_utc=None, store=STORE):
    """Adds new hours to the stored price file (re-pulls the last 3 days, which OASIS can revise)."""
    now_utc = pd.Timestamp.now("UTC").tz_localize(None) if now_utc is None else pd.Timestamp(now_utc)
    old = pd.read_csv(store, parse_dates=["timestamp_utc"]).set_index("timestamp_utc") if os.path.exists(store) else pd.DataFrame()
    start = EDAM_START if old.empty else old.index.max() - pd.Timedelta(days=3)
    new = fetch(start, now_utc.ceil("h") + pd.Timedelta(days=1))           # includes tomorrow's day-ahead prices once published
    allp = pd.concat([old[old.index < start], new]) if not old.empty else new
    allp = allp[~allp.index.duplicated(keep="last")].sort_index()
    allp.round(3).to_csv(store)
    return allp


if __name__ == "__main__":
    p = update()
    print(len(p), "hours", p.index.min(), "..", p.index.max())
    print(p.describe().round(1))
