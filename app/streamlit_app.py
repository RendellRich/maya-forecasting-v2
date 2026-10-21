"""
Maya v2: buy-ahead decision tool (end-of-semester version).
How much power should Maui Hiva buy the day before, given that running short on the day costs far more
than carrying a small surplus? Built on calibrated quantile forecasts, a daily risk tier and a live daily run.
The original Maya app (maya-forecasting) is unchanged and separate.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data"
LIVE = ROOT.parent / "live"
LOCAL_TZ = "America/Denver"
C_ACTUAL, C_MAYA, C_BASE, C_BUY, C_BAND = "#1f1f1f", "#2a78d6", "#eb6834", "#0f8a6a", "rgba(42,120,214,0.15)"
TIER_COL = {"Green": "#1e9e5a", "Yellow": "#d9a400", "Red": "#d1443a"}
LEVELS = ["P75", "P90", "P95", "P98"]

st.set_page_config(page_title="Maya v2 - Buy-Ahead Tool", page_icon="⚡", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_backtest():
    h = pd.read_parquet(DATA / "backtest_hourly.parquet")
    h = h[h.index >= "2024-03-06"]
    return h


@st.cache_data
def load_csv(name):
    return pd.read_csv(DATA / name)


@st.cache_data(ttl=600)
def load_live():
    if not (LIVE / "summary.json").exists():
        return None
    out = dict(summary=json.loads((LIVE / "summary.json").read_text()))
    out["hourly"] = pd.read_csv(LIVE / "live_hourly.csv", parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    out["days"] = pd.read_csv(LIVE / "live_days.csv", parse_dates=["buy_day"]).set_index("buy_day")
    out["track"] = pd.read_csv(LIVE / "track_record.csv", parse_dates=["buy_day"])
    out["drivers"] = pd.read_csv(LIVE / "drivers.csv", parse_dates=["buy_day"]).set_index("buy_day") if (LIVE / "drivers.csv").exists() else None
    return out


def local(idx):
    return pd.DatetimeIndex(idx).tz_localize("UTC").tz_convert(LOCAL_TZ)


def fmt(x, d=0):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"


def money(x):
    return f"${x:,.0f}"


def kpi(col, label, value, sub="", color=None):
    """Plain KPI card (no up/down arrows: most of these numbers are facts, not changes)."""
    c = f"color:{color};" if color else ""
    value, sub = str(value).replace("$", "&#36;"), str(sub).replace("$", "&#36;")
    col.markdown(f"<div style='font-size:0.9rem;color:#555'>{label}</div>"
                 f"<div style='font-size:1.9rem;line-height:1.25;{c}'>{value}</div>"
                 f"<div style='font-size:0.85rem;color:#666'>{sub}</div>", unsafe_allow_html=True)


def buy_cost(actual, buy, c_short, c_over):
    k = pd.concat([actual, buy], axis=1).dropna()
    a, b = k.iloc[:, 0], k.iloc[:, 1]
    short, over = (a - b).clip(lower=0), (b - a).clip(lower=0)
    return dict(hours=len(k), short_pct=(short > 0).mean() * 100, short_MWh=short.sum(), over_MWh=over.sum(),
                worst=short.max(), cost=(c_short * short + c_over * over).sum(), days=len(k) / 24)


@st.cache_data(ttl=600)
def load_prices():
    for f in [ROOT.parent / "pipeline" / "data" / "pace_prices_2026.csv", DATA / "pace_prices_2026.csv"]:
        if f.exists():
            return pd.read_csv(f, parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    return None


def settle(actual, buy, P):
    """Real settlement: buy B day-ahead at DA, cover or sell the gap at real-time RT. Returns extra $ vs perfect foresight."""
    k = pd.concat([actual.rename("a"), buy.rename("b"), P.da, P.rt], axis=1).dropna()
    gap = k.a - k.b
    return dict(hours=len(k), extra=((k.rt - k.da) * gap).sum(), mwh=k.a.sum(), short_pct=(gap > 0).mean() * 100,
                volume=k.b.sum() / k.a.sum())


@st.cache_data(ttl=600)
def load_strategies():
    f = DATA / "strategies_2026.parquet"
    return pd.read_parquet(f) if f.exists() else None


bt = load_backtest()
live = load_live()
prices = load_prices()
SD = load_strategies()

# ----------------------------------------------------------------------------- sidebar: Maui Hiva's size
st.sidebar.header("Maui Hiva's size")
PACE_NET_AVG = float(bt["net_actual"].mean())
size_mode = st.sidebar.radio("Show numbers for", ["Maui Hiva", "All of PACE"], index=0, horizontal=True)
if size_mode == "Maui Hiva":
    mh_mw = st.sidebar.number_input("Maui Hiva's average load, MW", min_value=0.1, value=100.0, step=10.0,
                                    help="Average power Maui Hiva buys per hour. 1 MW for an hour = 1 MWh.")
    SCALE = mh_mw / PACE_NET_AVG
    st.sidebar.caption(f"That is {SCALE * 100:.2f}% of PACE's average net load ({PACE_NET_AVG:,.0f} MW). Every MWh and dollar "
                       "figure is scaled to this size, assuming Maui Hiva's hourly pattern follows PACE's. 100 MW is a "
                       "placeholder until Maui Hiva confirms its load (or shares meter data).")
else:
    SCALE = 1.0
UNIT_NOTE = "Maui Hiva scale" if SCALE != 1.0 else "all of PACE"
NOT_MWH = {"period", "temp_fc", "cloud_fc", "temp_lag", "price_da", "price_rt", "tier", "buy_level", "buy_day", "issued"}


def scaled(df, cols=None):
    """Copy of df with MWh / $ columns multiplied by SCALE (shares, %, prices and counts untouched)."""
    if df is None or SCALE == 1.0:
        return df
    d = df.copy()
    cols = cols if cols is not None else [c for c in d.columns if c not in NOT_MWH and pd.api.types.is_numeric_dtype(d[c])]
    d[cols] = d[cols] * SCALE
    return d


bt = scaled(bt)
if live is not None:
    live = dict(live)
    live["hourly"] = scaled(live["hourly"])
    live["track"] = scaled(live["track"], [c for c in ["short_MWh", "surplus_MWh", "worst_short_MWh", "maya_P50_MAE", "pace_MAE",
                                                       "maya_extra_cost", "maya_short_cost", "pace_extra_cost"] if c in live["track"]])
    live["drivers"] = scaled(live["drivers"])
    live["days"] = scaled(live["days"], ["width", "disagree", "recent_err"])
if SD is not None:
    SD = scaled(SD, [c for c in SD.columns if c in {"actual", "perfect", "persistence", "pace", "maya_P95", "maya_day_rule",
                                                           "maya_hour_rule", "maya_ladder", "ladder_block1", "ladder_block2"}])
_load_csv_raw = load_csv
SCALE_COLS = {"lag48_risk_validation.csv": ["avg_worst_short_MWh", "avg_worst_P50_miss_MWh"],
              "lag48_risk_adjusted_buy.csv": ["avg_short_MWh", "avg_surplus_MWh", "worst_short_MWh", "cost_K19", "cost_K9"]}


def load_csv(name):
    d = _load_csv_raw(name)
    return scaled(d, SCALE_COLS[name]) if name in SCALE_COLS else d

# ----------------------------------------------------------------------------- sidebar: the client's costs
pdf_f = DATA / "maya_v2_summary.pdf"
if pdf_f.exists():
    st.sidebar.download_button("Download one-page summary (PDF)", pdf_f.read_bytes(), file_name="Maya_v2_summary.pdf",
                               mime="application/pdf", use_container_width=True,
                               help="Results and method on one page, refreshed by the daily run (100 MW placeholder size).")
st.sidebar.divider()
st.sidebar.header("Cost scenario")
st.sidebar.caption("What-if pricing for the cost tables. Real-price results (CAISO day-ahead and real-time, PACE) are shown "
                   "separately and do not depend on this.")
PRESETS = {"Typical 2026 day (from real PACE prices)": (12.0, 8.0), "Heat wave or cold snap": (150.0, 8.0),
           "Grid emergency": (1000.0, 8.0), "Custom": None}
preset = st.sidebar.radio("Scenario", list(PRESETS), index=0)
if PRESETS[preset] is None:
    c_short = st.sidebar.number_input("Extra cost per MWh bought short (real-time minus day-ahead price), $/MWh", min_value=0.5, value=50.0, step=5.0)
    c_over = st.sidebar.number_input("Loss per surplus MWh (day-ahead minus real-time resale price), $/MWh", min_value=0.5, value=8.0, step=0.5)
else:
    c_short, c_over = PRESETS[preset]
    st.sidebar.markdown(f"Short MWh: **\\${c_short:,.0f}** extra. Surplus MWh: **\\${c_over:,.0f}** lost.")
K = c_short / c_over
q_star = K / (K + 1)
st.sidebar.markdown(f"Short costs **{K:,.1f}x** a surplus MWh, so the cheapest level in theory is "
                    f"**P{min(q_star * 100, 99.9):.0f}**.")
st.sidebar.caption("Typical day: May to Oct 2026 PACE prices, when real-time was above day-ahead the gap averaged \\$12/MWh; "
                   "when below, \\$8/MWh. Spikes are rare but large (real-time hit \\$1,058 on Jul 24).")
series = st.sidebar.radio("What Maui Hiva buys", ["Net load (demand minus Milford renewables)", "PACE demand"], index=0)
pre = "net" if series.startswith("Net") else "dem"

# ----------------------------------------------------------------------------- header
st.title("Maya v2: how much to buy the day before")
st.caption("Calibrated buy-ahead forecasts for PacifiCorp East demand and Milford, Utah renewable output. "
           "Built so that buying short, the expensive mistake, happens rarely and on purpose. "
           "Every forecast uses only what is known at the buy deadline (10 am Mountain the day before). "
           f"MWh and dollar figures are shown for {UNIT_NOTE}.")

tab_buy, tab_shift, tab_track, tab_bt, tab_save, tab_risk, tab_how = st.tabs(
    ["Tomorrow's buy", "Shift load", "Live track record", "Maya vs PACE", "What it would have saved", "Risk score", "How it works"])

# ============================================================================= TOMORROW
with tab_buy:
    if live is None:
        st.info("No live run yet. Run `python pipeline/live_forecast.py` to create the first forecast.")
    else:
        S, H, Dy = live["summary"], live["hourly"], live["days"]
        bday = pd.Timestamp(S["buy_day"])
        T = H[H.buy_day == S["buy_day"]].copy()
        tier, lvl = S["tier"], S["buy_level"]
        if S.get("provisional"):
            st.warning(f"{S.get('note', 'Provisional forecast.')} Newest demand reading used: "
                       f"{pd.Timestamp(S['newest_demand_hour_utc']).tz_localize('UTC').tz_convert(LOCAL_TZ):%a %I %p} MT.")
        rule = st.radio("Buy rule", ["Maya rule (P95 on Green days, P98 on Yellow or Red days; Block 1 of the bid ladder)",
                                     "Fixed P90", "Fixed P95", "Fixed P98"], horizontal=True)
        use = lvl if rule.startswith("Maya") else rule.split()[-1]
        T["buy"] = T[f"{pre}_{use}"]
        T["best"] = T[f"{pre}_P50"]
        dl = local(T.index)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Buy day**<br><span style='font-size:1.6rem'>{bday:%a %b %d, %Y}</span><br>"
                    f"<span style='color:#666'>buy deadline {pd.Timestamp(S['buy_deadline_utc']).tz_localize('UTC').tz_convert(LOCAL_TZ):%a %I:%M %p} MT</span>",
                    unsafe_allow_html=True)
        c2.markdown(f"**Risk tier**<br><span style='font-size:1.6rem;color:{TIER_COL[tier]}'>&#9679; {tier}</span><br>"
                    f"<span style='color:#666'>{S['risk'] * 100:.0f}% chance of at least one short hour at P95</span>", unsafe_allow_html=True)
        nh = int(T.buy.notna().sum())
        kpi(c3, f"Recommended buy ({use})", f"{fmt(T.buy.sum())} MWh", f"for the day ({nh} of 24 hours ready)" if nh < 24 else "for the day, 24 hours", C_BUY)
        kpi(c4, "Safety cushion over best guess", f"{fmt((T.buy - T.best).sum())} MWh",
            f"{(T.buy - T.best).sum() / T.best.sum() * 100:.1f}% above the best guess of {fmt(T.best.sum())} MWh")

        fig = go.Figure()
        fig.add_scatter(x=dl, y=T[f"{pre}_P98"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=dl, y=T[f"{pre}_P50"], fill="tonexty", fillcolor=C_BAND, line=dict(width=0), name="Range up to P98", hoverinfo="skip")
        fig.add_scatter(x=dl, y=T.best, name="Best guess (P50)", line=dict(color=C_MAYA, width=1.8, dash="dot"))
        fig.add_scatter(x=dl, y=T.buy, name=f"Buy ({use})", line=dict(color=C_BUY, width=3))
        if pre == "dem" and T.pace_dayahead.notna().any():
            fig.add_scatter(x=dl, y=T.pace_dayahead, name="PACE day-ahead", line=dict(color=C_BASE, width=1.2, dash="dash"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="MWh per hour",
                          legend=dict(orientation="h", y=1.08), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        wk_f = LIVE / "week_ahead.csv"
        if wk_f.exists() and pre == "net":
            WK = pd.read_csv(wk_f, parse_dates=["day"])
            if len(WK):
                for c in ["P10", "P50", "P90", "P95"]:
                    WK[c] = WK[c] * SCALE
                with st.expander("Week ahead: daily net load for the next 7 days", expanded=True):
                    full = int(T.best.notna().sum()) == 24      # buy day only once all 24 hours are forecast
                    days = ([bday] if full else []) + list(WK.day)
                    p50 = ([T.best.sum()] if full else []) + list(WK.P50)
                    lo = ([T.best.sum()] if full else []) + list(WK.P10)
                    hi = ([T[f"{pre}_P95"].sum()] if full else []) + list(WK.P95)
                    fw = go.Figure()
                    fw.add_bar(x=days, y=[h - l for h, l in zip(hi, lo)], base=lo, marker_color=C_BAND, name="Range (P10 to P95)",
                               hovertemplate="%{x|%a %b %d}: %{base:,.0f} to %{y:,.0f} more<extra></extra>")
                    fw.add_scatter(x=days, y=p50, mode="markers+lines", name="Best guess (P50)", line=dict(color=C_MAYA, width=2))
                    if full:
                        fw.add_scatter(x=[bday], y=[T.buy.sum()], mode="markers", name=f"Buy day: planned buy ({use})",
                                       marker=dict(color=C_BUY, size=12, symbol="diamond"))
                    fw.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="MWh per day",
                                     legend=dict(orientation="h", y=1.15), xaxis=dict(tickformat="%a %b %d"))
                    st.plotly_chart(fw, use_container_width=True)
                    ev_f = DATA / "week_ahead_eval.csv"
                    if ev_f.exists():
                        EV = pd.read_csv(ev_f)
                        st.caption(f"Days 2 to 7 come from a separate daily-total model run on ECMWF forecasts 2 to 7 days out. Tested on "
                                   f"Aug 2024 to Oct 2026 with the forecasts available at each lead: daily error "
                                   f"{EV.mape_P50.min():.1f}% to {EV.mape_P50.max():.1f}%, {EV.better_than_naive_pct.min():.0f}% to "
                                   f"{EV.better_than_naive_pct.max():.0f}% better than repeating the same weekday last week; the actual came in "
                                   f"above P95 on {EV.above_P95_pct.min():.0f}% to {EV.above_P95_pct.max():.0f}% of days. Use it for weekly "
                                   "planning; the day-ahead buy should still follow the hourly plan above.")

        left, right = st.columns([3, 2])
        with right:
            st.subheader("Why tomorrow looks like this")
            dv = live["drivers"]
            if dv is not None and bday in dv.index:
                r = dv.loc[bday]
                parts = r.drop(["baseline", "gbm_P50_mean"]).sort_values()
                f2 = go.Figure(go.Bar(x=parts.values, y=parts.index, orientation="h",
                                      marker_color=[C_BASE if v > 0 else C_MAYA for v in parts.values],
                                      text=[f"{v:+,.0f}" for v in parts.values], textposition="auto", cliponaxis=False))
                lim = max(abs(parts.values).max() * 1.25, 50)
                f2.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[-lim, lim]),
                                 xaxis_title="MWh per hour vs an average hour")
                st.plotly_chart(f2, use_container_width=True)
                st.caption(f"Average hour in training: {fmt(r.baseline)} MWh. Tomorrow's demand best guess (gradient boosting half of the model): "
                           f"{fmt(r.gbm_P50_mean)} MWh per hour. Bars show what pushes it up (orange) or down (blue), from SHAP values.")
            d = Dy.loc[bday]
            why = []
            if d.recent_err > 350 * SCALE:
                why.append("recent forecast misses have been large")
            if d.temp_max > 33 or d.temp_min < -8:
                why.append("temperatures are near the extremes")
            if d.temp_swing > 4:
                why.append("a big temperature change from two days earlier")
            if d.holiday or d.after_holiday:
                why.append("a holiday effect")
            st.markdown("**Risk inputs:** " + (", ".join(why) if why else "nothing unusual; a normal day") +
                        f". Temperature {d.temp_min:.0f} to {d.temp_max:.0f} C, recent miss {d.recent_err:.0f} MWh per hour.")
        with left:
            st.subheader("Hour by hour")
            tbl = pd.DataFrame({"Hour ending (MT)": dl.strftime("%a %I %p"), "Best guess (MWh)": T.best.round(0).values,
                                f"Buy {use} (MWh)": T.buy.round(0).values, "Cushion (MWh)": (T.buy - T.best).round(0).values,
                                "Renewables you can count on, P05 (MWh)": T.ren_P05.round(0).values})
            st.dataframe(tbl, hide_index=True, use_container_width=True, height=300)
            st.download_button("Download this buy plan (CSV)", tbl.to_csv(index=False), file_name=f"maya_buy_plan_{bday:%Y-%m-%d}.csv")
        if SD is not None and SD.index.isin(T.index).any():
            st.subheader("Bid ladder for the buy day")
            L = SD.reindex(T.index)
            st.markdown("How to submit the buy in the day-ahead market (the recommended strategy). **Block 1** is the Maya-rule "
                        "amount, bought at any price, so it is always covered. **Block 2** is the extra cushion up to P98, bid at a limit price equal to the expected "
                        "real-time price for that hour: it is bought only if the day-ahead price clears at or below that limit, "
                        "because otherwise covering any shortfall in real time is expected to be cheaper. Hours shaded red have a high "
                        "chance of a real-time price spike.")
            lad = pd.DataFrame({"Hour ending (MT)": dl.strftime("%a %I %p"),
                                "Block 1: buy at any price (MWh)": L.ladder_block1.round(1 if SCALE != 1 else 0).values,
                                "Block 2: extra (MWh)": L.ladder_block2.round(1 if SCALE != 1 else 0).values,
                                "Block 2 limit ($/MWh)": L.ladder_limit.round(0).values,
                                "Spike risk": (L.spike_prob * 100).round(0).map(lambda v: "n/a" if pd.isna(v) else f"{v:.0f}%").values,
                                "High risk": np.where(L.high_risk_hour == 1, "yes", "").tolist()})
            dec = "{:,.1f}" if SCALE != 1.0 else "{:,.0f}"
            st.dataframe(lad.style.apply(lambda r: ["background-color: rgba(209,68,58,0.12)" if r["High risk"] == "yes" else "" for _ in r], axis=1)
                         .format({"Block 1: buy at any price (MWh)": dec, "Block 2: extra (MWh)": dec, "Block 2 limit ($/MWh)": "${:,.0f}"}),
                         hide_index=True, use_container_width=True, height=320)
            st.download_button("Download the bid ladder (CSV)", lad.to_csv(index=False), file_name=f"maya_bid_ladder_{bday:%Y-%m-%d}.csv")
        st.caption(f"Run {S['run_utc']} UTC. Weather: {S['weather']}. Net load = PACE demand minus Milford-area renewable output.")

# ============================================================================= SHIFT LOAD
with tab_shift:
    sb_f = DATA / "shift_backtest.csv"
    if SD is None or "exp_price" not in SD or live is None or not sb_f.exists():
        st.info("Run `python pipeline/strategies.py` to build the load-shifting data.")
    else:
        st.markdown("If some of Maui Hiva's load is flexible (pumping, charging, cooling, batch processes), moving it from the "
                    "hours expected to be expensive into the hours expected to be cheap lowers the day-ahead bill. The expected "
                    "price for each hour is the average day-ahead price at that hour over the last 7 days, all published before "
                    "the buy deadline. On May to Oct 2026 prices it ranked the hours of the next day with a correlation of 0.86.")
        c1, c2 = st.columns(2)
        flex = c1.number_input("Flexible load Maui Hiva can move, MW", min_value=0.0, value=5.0, step=1.0,
                               help="Power that can run at a different hour of the same day without changing the day's total use.")
        kh = c2.slider("Hours to move it out of (and into)", 1, 6, 4)
        bday = pd.Timestamp(live["summary"]["buy_day"])
        T2 = SD.reindex(buy_hours := pd.date_range(bday + pd.Timedelta(hours=7), periods=24, freq="h"))
        if T2.exp_price.notna().sum() >= 20:
            e = T2.exp_price.astype(float)
            hi, lo = e.nlargest(kh).index, e.nsmallest(kh).index
            save_tmr = flex * (e[hi].sum() - e[lo].sum())
            x = local(T2.index)
            colors = ["#d1443a" if t in hi else ("#1e9e5a" if t in lo else "#c9c9c9") for t in T2.index]
            fig = go.Figure(go.Bar(x=x, y=e, marker_color=colors, hovertemplate="%{x|%a %I %p}: $%{y:.0f}/MWh<extra></extra>"))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Expected day-ahead price, $/MWh",
                              title=f"{bday:%a %b %d}: move load out of the red hours and into the green hours")
            st.plotly_chart(fig, use_container_width=True)
            k1, k2, k3 = st.columns(3)
            kpi(k1, "Move out of", ", ".join(f"{t:%I %p}".lstrip("0") for t in sorted(local(hi))), "Mountain time, hour ending")
            kpi(k2, "Move into", ", ".join(f"{t:%I %p}".lstrip("0") for t in sorted(local(lo))), "Mountain time, hour ending")
            kpi(k3, "Expected saving for the day", money(save_tmr), f"{flex:,.0f} MW x {kh} hours moved", C_BUY)
        SB = pd.read_csv(sb_f, parse_dates=["day"])
        plan, hind = SB[f"plan_k{kh}"] * flex, SB[f"hindsight_k{kh}"] * flex
        st.subheader(f"What this would have saved since {SB.day.min():%b %d}")
        k1, k2, k3 = st.columns(3)
        kpi(k1, "Saved by following the plan", money(plan.sum()), f"{len(SB)} days; {money(plan.mean())} per day on average", C_BUY)
        kpi(k2, "Upper bound with perfect hindsight", money(hind.sum()), f"the plan captured {plan.sum() / hind.sum() * 100:.0f}% of it")
        kpi(k3, "Days the plan saved money", f"{(plan > 0).mean() * 100:.0f}%", f"worst day {money(plan.min())}, best day {money(plan.max())}")
        fig = go.Figure()
        fig.add_bar(x=SB.day, y=hind, name="Perfect hindsight", marker_color="#d9d9d9")
        fig.add_bar(x=SB.day, y=plan, name="Maya plan", marker_color=C_BUY)
        fig.update_layout(barmode="overlay", height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="$ saved per day",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Assumes the flexible load can run at any hour of the same day with no extra cost or rebound, and that Maui Hiva's "
                   "purchases are too small to move prices. Savings are at the day-ahead prices that actually cleared. "
                   "Spike risk was tested as an extra signal for this ranking and did not improve it.")

# ============================================================================= LIVE TRACK RECORD
with tab_track:
    if live is None or live["track"].empty:
        st.info("No finished live days yet.")
    else:
        H = live["hourly"].dropna(subset=["dem_actual"])
        H = H[pd.to_datetime(H.buy_day) < pd.Timestamp(live["summary"]["buy_day"])]
        st.markdown(f"Every day since the training data ends (**{live['track'].buy_day.min():%b %d}** to **{live['track'].buy_day.max():%b %d, %Y}**), "
                    "forecast exactly as the live run would have: day-ahead ECMWF weather, the 10 am Mountain buy deadline, "
                    "no retraining. The models never saw these days.")
        act = H.dem_actual if pre == "dem" else H.dem_actual - H.ren_actual
        maya = buy_cost(act, H.buy_demand_MWh if pre == "dem" else H.buy_net_MWh, c_short, c_over)
        pace_b = H.pace_dayahead if pre == "dem" else H.pace_dayahead - H.ren_actual.shift(24)
        pace = buy_cost(act, pace_b, c_short, c_over)
        p50 = (H.dem_actual - H.dem_P50).abs().mean()
        pmae = (H.dem_actual - H.pace_dayahead).abs().mean()
        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "Hours bought short, Maya buy", f"{maya['short_pct']:.1f}%", f"PACE as published: {pace['short_pct']:.1f}%", C_BUY)
        if prices is not None:
            sm, sp_ = settle(act, H.buy_demand_MWh if pre == "dem" else H.buy_net_MWh, prices), settle(act, pace_b, prices)
            kpi(k2, "At real prices, vs perfect foresight", f"{sm['extra'] / sm['mwh']:+.2f} $/MWh",
                f"buying PACE's number: {sp_['extra'] / sp_['mwh']:+.2f} $/MWh (negative = cheaper than perfect)")
        else:
            kpi(k2, f"Cost per day ({preset.split(' (')[0].lower()})", money(maya["cost"] / maya["days"]),
                f"buying PACE's number: {money(pace['cost'] / pace['days'])}")
        kpi(k3, "Best-guess error, Maya (per hour)", f"{fmt(p50)} MWh", f"PACE: {fmt(pmae)} MWh, Maya {(pmae - p50) / pmae * 100:.0f}% lower")
        kpi(k4, "Worst single hour short, Maya", f"{fmt(maya['worst'])} MWh", f"PACE: {fmt(pace['worst'])} MWh")

        dd = pd.DataFrame({"actual": act, "buy": H.buy_demand_MWh if pre == "dem" else H.buy_net_MWh, "pace": pace_b,
                           "day": pd.to_datetime(H.buy_day)}).groupby("day").sum(min_count=20)
        fig = go.Figure()
        fig.add_bar(x=dd.index, y=dd.buy - dd.actual, name="Maya buy minus actual", marker_color=C_BUY)
        fig.add_bar(x=dd.index, y=dd.pace - dd.actual, name="PACE minus actual", marker_color=C_BASE)
        fig.add_hline(y=0, line_color="#999")
        fig.update_layout(barmode="group", height=330, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis_title="MWh per day (below 0 = net short for the day)", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Daily totals hide hourly shortfalls; the table counts short hours. "
                   "PACE's day-ahead forecast is shown for comparison only: EIA posts it shortly before the day starts, after a morning buy.")
        t = live["track"].copy()
        t["buy_day"] = t.buy_day.dt.strftime("%a %b %d")
        st.dataframe(t.rename(columns={"buy_day": "Buy day", "tier": "Tier", "buy_level": "Level", "short_hours": "Short hours",
                                       "short_MWh": "Short MWh", "surplus_MWh": "Surplus MWh", "worst_short_MWh": "Worst hour short",
                                       "maya_P50_MAE": "Maya best-guess error", "pace_MAE": "PACE error", "pace_short_hours": "PACE short hours",
                                       "price_da_avg": "Day-ahead $/MWh", "price_rt_avg": "Real-time $/MWh",
                                       "maya_extra_cost": "Maya $ vs perfect", "maya_short_cost": "Maya $ on short hours",
                                       "pace_extra_cost": "PACE number $ vs perfect"})
                     .drop(columns=["hours"]).iloc[::-1], hide_index=True, use_container_width=True, height=320)

# ============================================================================= BACKTEST
with tab_bt:
    if prices is not None:
        st.subheader("At real prices (CAISO day-ahead and real-time, PACE, since May 1 2026)")
        st.markdown("Each option buys its number day-ahead at the day-ahead price, then covers any gap (or sells any surplus) at the "
                    "real-time price. Shown as extra cost compared with perfect foresight, per MWh used. "
                    "Covers the 2026 forward test from May plus every live day since.")
        L = live["hourly"] if live is not None else None
        B26 = bt[bt.index >= "2026-05-01"]
        tierB = pd.Series(B26.index.normalize().map(load_csv("backtest_daily_risk.csv").assign(day=lambda x: pd.to_datetime(x.iloc[:, 0]))
                                                    .set_index("day").tier_v2), index=B26.index)
        cols = {f"{pre}_{l}" for l in ["P50"] + LEVELS}
        frames = [B26[list(cols)].assign(actual=B26[f"{pre}_actual"], pace=B26[f"{pre}_pace"], tier=tierB)]
        if L is not None:
            Lx = L[(L.index > B26.index.max()) & L.dem_actual.notna()]
            act_l = Lx.dem_actual if pre == "dem" else Lx.dem_actual - Lx.ren_actual
            pace_l = Lx.pace_dayahead if pre == "dem" else Lx.pace_dayahead - Lx.ren_actual.shift(24)
            frames.append(Lx[list(cols)].assign(actual=act_l, pace=pace_l, tier=Lx.tier))
        X = pd.concat(frames).sort_index()
        X["rule"] = np.where(X.tier == "Green", X[f"{pre}_P95"], X[f"{pre}_P98"])
        scale = settle(X.actual, X.rule, prices)["volume"] / settle(X.actual, X.pace, prices)["volume"]
        strat = {"PACE as published": X.pace, f"PACE scaled up {100 * (scale - 1):.0f}% (same volume as the Maya rule)": X.pace * scale}
        strat.update({f"Maya {l}": X[f"{pre}_{l}"] for l in ["P50"] + LEVELS})
        strat["Maya rule (P95, P98 on Yellow/Red)"] = X.rule
        rows = []
        for name, f in strat.items():
            r = settle(X.actual, f, prices)
            rows.append({"Option": name, "Hours": r["hours"], "Short hours %": round(r["short_pct"], 1),
                         "Buys vs actual": f"{(r['volume'] - 1) * 100:+.1f}%", "$ per MWh vs perfect": round(r["extra"] / r["mwh"], 3)})
        RP = pd.DataFrame(rows)
        st.dataframe(RP, hide_index=True, use_container_width=True)
        avg = prices.loc[prices.index.isin(X.index)]
        st.caption(f"Day-ahead averaged \\${avg.da.mean():.0f}/MWh and real-time \\${avg.rt.mean():.0f}/MWh over these hours. Because real-time "
                   "ran slightly above day-ahead, buying extra earned a little on average, so every cushioned option beats perfect foresight. "
                   "That edge comes from the market, not the forecast: scaling PACE's number up by the same volume earns about as much "
                   "but is short far more often. At real prices the differences are cents per MWh; Maya's value is buying the same volume "
                   "while being short about half as often, which is what protects against the rare spike hours.")

    st.divider()
    per = st.radio("Period", ["2024-25 backtest", "2026 forward test", "Both"], horizontal=True, index=1)
    pm = {"2024-25 backtest": ["A backtest 2024-25"], "2026 forward test": ["B forward test 2026"], "Both": ["A backtest 2024-25", "B forward test 2026"]}[per]
    B = bt[bt.period.isin(pm)]
    act = B[f"{pre}_actual"]
    opts = {f"Maya {l}": B[f"{pre}_{l}"] for l in LEVELS}
    opts["Maya best guess (P50)"] = B[f"{pre}_P50"]
    opts["PACE as published"] = B[f"{pre}_pace"]
    for l in ["P90", "P95", "P98"]:
        opts[f"PACE + cushion {l}"] = B[f"{pre}_pace_{l}"]
    rows = []
    for name, f in opts.items():
        r = buy_cost(act, f, c_short, c_over)
        rows.append({"Option": name, "Short hours %": round(r["short_pct"], 1), "Avg short MWh/h": round(r["short_MWh"] / r["hours"], 1),
                     "Avg surplus MWh/h": round(r["over_MWh"] / r["hours"], 0), "Worst hour short": round(r["worst"], 0),
                     "Cost per day": r["cost"] / r["days"]})
    R = pd.DataFrame(rows).sort_values("Cost per day")
    best = R.iloc[0]
    pace_raw = R.set_index("Option").loc["PACE as published", "Cost per day"]
    best_pace = R[R.Option.str.startswith("PACE")].iloc[0]
    best_maya = R[R.Option.str.startswith("Maya")].iloc[0]
    st.subheader(f"Scenario costs: {preset.split(' (')[0]}")
    k1, k2, k3 = st.columns(3)
    kpi(k1, "Cheapest option in this scenario", best.Option, f"{money(best['Cost per day'])} per day", C_BUY)
    kpi(k2, "Best Maya vs buying PACE's number", f"{(pace_raw - best_maya['Cost per day']) / pace_raw * 100:.0f}% cheaper",
        f"saves {money(pace_raw - best_maya['Cost per day'])} per day")
    gap = (best_pace['Cost per day'] - best_maya['Cost per day']) / best_pace['Cost per day'] * 100
    kpi(k3, "Best Maya vs best PACE + cushion", f"{abs(gap):.1f}% {'cheaper' if gap >= 0 else 'more expensive'}",
        f"{best_maya.Option} vs {best_pace.Option}")
    R["Cost per day"] = R["Cost per day"].map(money)
    st.dataframe(R, hide_index=True, use_container_width=True)
    st.caption("PACE + cushion = PACE's own forecast given the same calibrated safety margin as Maya, the fairest comparison. "
               "Note that a real buyer could not use PACE's number this way from EIA data (it is posted after a morning buy).")

    cA, cB = st.columns(2)
    with cA:
        st.subheader("Does P95 mean what it says?")
        cov = load_csv("lag48_coverage.csv")
        cov = cov[cov.period.isin(pm) & (cov.series == ("Net load" if pre == "net" else "Demand"))]
        fig = go.Figure()
        for p_, o in cov.groupby("period"):
            fig.add_scatter(x=o.forecast, y=o.under_pct, mode="markers+lines", name=p_[:1] == "A" and "2024-25" or "2026")
        fig.add_scatter(x=cov.drop_duplicates("forecast").forecast, y=cov.drop_duplicates("forecast").target_under_pct, name="Target",
                        mode="lines", line=dict(color="#999", dash="dot"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="% of hours short")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each level is recalibrated every day on the previous 60 finished days, so the share of short hours stays on target as conditions change.")
    with cB:
        st.subheader("Savings vs PACE + cushion, by cost ratio")
        cc = load_csv("lag48_cost_curve.csv")
        cc = cc[(cc.series == ("Net load" if pre == "net" else "Demand")) & cc.period.isin(pm if len(pm) == 1 else ["All"])]
        c24 = load_csv("lag24_cost_curve.csv")
        c24 = c24[(c24.series == ("Net load" if pre == "net" else "Demand")) & c24.period.isin(pm if len(pm) == 1 else ["All"])]
        fig = go.Figure()
        fig.add_scatter(x=cc.K, y=cc.saving_vs_pace_cushion_pct, name="Maya (honest 48 h inputs)", line=dict(color=C_BUY, width=3))
        fig.add_scatter(x=c24.K, y=c24.saving_vs_pace_cushion_pct, name="If yesterday's same hour were known", line=dict(color="#999", dash="dot"))
        fig.add_hline(y=0, line_color="#999")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(type="log", tickvals=[1, 2, 3, 5, 9, 19, 49], ticktext=["1x", "2x", "3x", "5x", "9x", "19x", "49x"]),
                          xaxis_title="Short cost / surplus cost (log)", yaxis_title="% cheaper than PACE + cushion",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The grey line uses yesterday's same-hour demand, which is not all known at a 10 am buy. It shows what an "
                   "afternoon buy deadline could be worth.")

# ============================================================================= SAVINGS
STRAT_NAMES = {"persistence": "No forecast (buy the same hour 2 days ago)", "pace": "PACE's forecast as published",
               "maya_P95": "Maya P95", "maya_day_rule": "Maya daily rule (P95, P98 on Yellow/Red days)",
               "maya_hour_rule": "Maya hourly rule (P98 in high spike-risk hours; tested, not recommended)",
               "maya_ladder": "Maya bid ladder (recommended: daily rule + extra to P98 if day-ahead is below expected real-time)"}
STRAT_COL = {"persistence": "#9a9a9a", "pace": C_BASE, "maya_P95": "#7fb0e8", "maya_day_rule": C_BUY,
             "maya_hour_rule": "#6b4fbb", "maya_ladder": "#b8860b"}
with tab_save:
    if SD is None or SD.da.notna().sum() == 0:
        st.info("Run `python pipeline/strategies.py` to build the strategy history.")
    else:
        S = SD[SD.da.notna() & SD.rt.notna() & SD.actual.notna()].copy()
        S["day"] = (S.index - pd.Timedelta(hours=7)).normalize()

        def bill(col, frame):
            k = frame[["actual", col, "da", "rt"]].dropna()
            gap = k.actual - k[col]
            return dict(bill=(k.da * k[col] + k.rt * gap).sum(), extra=((k.rt - k.da) * gap).sum(), short_pct=(gap > 0).mean() * 100,
                        short_mwh=gap.clip(lower=0).sum(), worst=gap.clip(lower=0).max(), mwh=k.actual.sum(), hours=len(k))
        st.markdown(f"What Maui Hiva's day-ahead power bill would have been from **{S.index.min():%b %d}** to **{S.index.max():%b %d, %Y}** "
                    f"({len(S) // 24} days of real CAISO day-ahead and real-time PACE prices), for {UNIT_NOTE}. "
                    "Each strategy buys its number day-ahead; any gap is covered (or any surplus sold) at the real-time price.")
        perfect = bill("perfect", S)
        rows = []
        for key, name in STRAT_NAMES.items():
            r = bill(key, S)
            rows.append({"Strategy": name, "key": key, "Power bill": r["bill"], "vs perfect foresight": r["extra"],
                         "vs no forecast": r["bill"] - bill("persistence", S)["bill"], "Hours short %": round(r["short_pct"], 1),
                         "MWh bought short": r["short_mwh"], "Worst hour short (MWh)": r["worst"]})
        T = pd.DataFrame(rows)
        best = T[T.key.str.startswith("maya")].sort_values("Power bill").iloc[0]
        nf, pc = T.set_index("key").loc["persistence"], T.set_index("key").loc["pace"]
        k1, k2, k3 = st.columns(3)
        kpi(k1, "Bill with perfect foresight", money(perfect["bill"]), f"average {money(perfect['bill'] / perfect['mwh'])}/MWh")
        ci_f = DATA / "savings_ci.csv"
        CI = pd.read_csv(ci_f).set_index(["maya", "baseline"]) if ci_f.exists() else None

        def ci_txt(base):
            if CI is None or (best.key, base) not in CI.index:
                return ""
            r = CI.loc[(best.key, base)]
            return (f"90% range {money(r.saved_lo * SCALE)} to {money(r.saved_hi * SCALE)}; "
                    f"ahead in {r.prob_positive * 100:.1f}% of resamples. ")
        kpi(k2, f"{best.Strategy.split(' (')[0]} vs no forecast", money(nf["Power bill"] - best["Power bill"]) + " saved",
            ci_txt("persistence") + f"Short {best['Hours short %']:.1f}% of hours vs {nf['Hours short %']:.0f}%.", C_BUY)
        kpi(k3, f"{best.Strategy.split(' (')[0]} vs PACE's number", money(pc["Power bill"] - best["Power bill"]) + " saved",
            ci_txt("pace") + f"MWh bought short: {fmt(best['MWh bought short'])} vs {fmt(pc['MWh bought short'])}.")
        show = T.drop(columns=["key"]).copy()
        for c in ["Power bill", "vs perfect foresight", "vs no forecast"]:
            show[c] = show[c].map(lambda v: f"${v:,.0f}" if v >= 0 else f"-${-v:,.0f}")
        for c in ["MWh bought short", "Worst hour short (MWh)"]:
            show[c] = show[c].round(1 if SCALE != 1.0 else 0)
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.caption("Negative = cheaper. On ordinary days the strategies differ by cents per MWh; the real gap is in how often each one "
                   "is caught short, which is what decides the bill on a spike day. The days below show that. "
                   "90% ranges come from a day-block bootstrap: 4,000 resamples of whole buy days, so spike days stay intact. "
                   "They are wide because day-ahead prices for PACE only exist since May 2026.")

        st.subheader("Spike days")
        S["spread"] = S.rt - S.da
        top = (S.assign(pos=S.spread.clip(lower=0) * S.actual).groupby("day").agg(max_rt=("rt", "max"), max_spread=("spread", "max"), pos=("pos", "sum"))
               .sort_values("pos", ascending=False).head(8))
        pick = st.selectbox("Day (Mountain time), ranked by how much real-time ran above day-ahead",
                            [f"{d:%a %b %d}  (real-time peak ${r.max_rt:,.0f}/MWh)" for d, r in top.iterrows()])
        dsel = top.index[[f"{d:%a %b %d}  (real-time peak ${r.max_rt:,.0f}/MWh)" for d, r in top.iterrows()].index(pick)]
        Dd = S[S.day == dsel]
        x = local(Dd.index)
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=x, y=Dd.actual, name="Actual need", marker_color="#d9d9d9", secondary_y=False)
        for key in ["persistence", "pace", "maya_ladder"]:
            fig.add_scatter(x=x, y=Dd[key], name=STRAT_NAMES[key].split(" (")[0], line=dict(color=STRAT_COL[key], width=2.4 if key.startswith("maya") else 1.6), secondary_y=False)
        fig.add_scatter(x=x, y=Dd.rt, name="Real-time price", line=dict(color="#d1443a", width=1.4, dash="dot"), secondary_y=True)
        fig.add_scatter(x=x, y=Dd.da, name="Day-ahead price", line=dict(color="#555", width=1.2, dash="dash"), secondary_y=True)
        fig.update_yaxes(title_text="MWh per hour", secondary_y=False)
        fig.update_yaxes(title_text="$/MWh", secondary_y=True)
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        dr = []
        for key, name in STRAT_NAMES.items():
            r = bill(key, Dd)
            dr.append({"Strategy": name, "Bill that day": r["bill"], "vs perfect foresight": r["extra"], "Hours short": int(round(r["short_pct"] * r["hours"] / 100)),
                       "MWh bought short": round(r["short_mwh"], 1)})
        DR = pd.DataFrame(dr)
        for c in ["Bill that day", "vs perfect foresight"]:
            DR[c] = DR[c].map(lambda v: f"${v:,.0f}" if v >= 0 else f"-${-v:,.0f}")
        st.dataframe(DR, hide_index=True, use_container_width=True)

# ============================================================================= RISK
with tab_risk:
    st.markdown("Each day gets a **Green / Yellow / Red** tier from a small classifier that estimates the chance of at least "
                "one hour coming in above the P95 buy. It uses only inputs known at the buy deadline: model uncertainty, "
                "disagreement between the two model halves, forecast temperature and how fast it is changing, cloud, recent "
                "forecast misses, and the calendar. Scored walk-forward: each month by a model trained only on earlier days.")
    rv = load_csv("lag48_risk_validation.csv")
    v = rv[(rv.version == "v2 classifier") & rv.tier.isin(["Green", "Yellow", "Red"])]
    pick = st.radio("Period ", ["2025", "2026", "2025-26"], horizontal=True, index=2)
    v = v[v.period == pick]
    cols = st.columns(3)
    for c, (_, r) in zip(cols, v.iterrows()):
        c.markdown(f"<span style='font-size:1.3rem;color:{TIER_COL[r.tier]}'>&#9679; {r.tier}</span> "
                   f"({r.share_of_days_pct:.0f}% of days)", unsafe_allow_html=True)
        kpi(c, "Days with a short hour at P95", f"{r.days_with_a_short_hour_pct:.0f}%",
            f"average worst hour short: {r.avg_worst_short_MWh:,.1f} MWh")
    auc = rv[(rv.version == "AUC") & (rv.period == pick) & (rv.tier == "v2 classifier")].share_of_days_pct
    st.caption(f"Ranking skill (AUC) {float(auc.iloc[0]):.2f} (0.5 = coin flip). A model-uncertainty-only version scored about 0.5, "
               "because daily recalibration already absorbs it. The tier's value is in moving the buy level, below.")
    st.subheader("Does stepping up on risky days pay?")
    ra = load_csv("lag48_risk_adjusted_buy.csv")
    ra = ra[ra.period == pick].drop(columns=["period"]).round(1 if SCALE == 1.0 else 2)
    st.dataframe(ra.rename(columns={"policy": "Buy rule", "short_hours_pct": "Short hours %", "avg_short_MWh": "Avg short MWh/h",
                                    "avg_surplus_MWh": "Avg surplus MWh/h", "worst_short_MWh": "Worst hour short",
                                    "cost_K19": "Cost index (short = 19x)", "cost_K9": "Cost index (short = 9x)"}).drop(columns=["hours"]),
                 hide_index=True, use_container_width=True)
    st.caption("Demand. Stepping up to P98 on Yellow and Red days cuts short hours by about a quarter and lowers the worst hour, "
               "for about 1% more cost at a 19x ratio. Cost index = average per hour of 19 x short MWh + surplus MWh.")
    st.subheader("Hourly spike risk")
    st.markdown(f"A second model scores each hour for the chance that the real-time price comes in more than \\$25/MWh above its "
                "normal level for that hour. It uses only buy-time inputs: hour, weekday, month, forecast temperature, sun and cloud, "
                "Maya's forecast net load and its ramp, forecast uncertainty, and recent real-time prices (ending two days earlier). "
                "Trained on PACE real-time prices from March 2024, scored walk-forward month by month.")
    sv = load_csv("spike_validation.csv")
    sv = sv[sv.period.isin(["2025", "2026"])]
    cs = st.columns(len(sv))
    for c, (_, r) in zip(cs, sv.iterrows()):
        kpi(c, f"{r.period}: spikes caught in the flagged hours", f"{r.share_of_spikes_caught_pct:.0f}%",
            f"flagged {r.high_risk_share_pct:.0f}% of hours; spike rate {r.spike_rate_high_pct:.1f}% in flagged hours vs "
            f"{r.spike_rate_other_pct:.1f}% in others; AUC {r.auc:.2f}")
    st.caption("Spike risk sets the bid ladder's limit prices and flags hours on the buy plan. Used on its own to add cushion hour by "
               "hour it did not beat the simpler daily rule, because the day-ahead price already prices in most predictable risk.")
    st.subheader("Daily risk tier over time")
    dr = load_csv("backtest_daily_risk.csv")
    dr["day"] = pd.to_datetime(dr.iloc[:, 0])
    fig = go.Figure()
    for tname, o in dr.groupby("tier_v2"):
        fig.add_scatter(x=o.day, y=o.risk_v2, mode="markers", name=tname, marker=dict(color=TIER_COL[tname], size=6))
    sh = dr[dr.label == 1]
    fig.add_scatter(x=sh.day, y=[0.02] * len(sh), mode="markers", name="Day had a short hour", marker=dict(symbol="line-ns-open", color="#333", size=8))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Estimated chance of a short hour", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================= HOW IT WORKS
with tab_how:
    rl = LIVE / "retrain_log.csv"
    mv = ROOT.parent / "pipeline" / "models" / "lag48" / "model_version.txt"
    st.subheader("Model in use")
    st.markdown(f"**{mv.read_text().strip() if mv.exists() else 'v2.0 (trained 2019, 2021-2025)'}**")
    if rl.exists():
        RL = pd.read_csv(rl)
        st.dataframe(RL.rename(columns={"checked_utc": "Checked (UTC)", "holdout_start": "Test from", "holdout_end": "Test to",
                                        "champion_demand_pinball": "Current model error", "challenger_demand_pinball": "Retrained model error",
                                        "demand_gain_pct": "Improvement %", "decision": "Decision"})
                     [["Checked (UTC)", "Test from", "Test to", "Current model error", "Retrained model error", "Improvement %", "Decision"]]
                     .iloc[::-1], hide_index=True, use_container_width=True)
        st.caption("Error = demand pinball loss (average quantile loss, MWh) on held-out days. Lower is better.")
    rb = DATA / "rolling_quarterly_mae.csv"
    if rb.exists():
        st.subheader("Rolling test: every quarter since 2022")
        Q = pd.read_csv(rb)
        fig = go.Figure()
        for col, name, color in [("pace_MAE", "PACE's own forecast", C_BASE), ("maya_v2_MAE", "Maya v2 inputs", "#9a9a9a"), ("maya_v3_MAE", "Maya v3 inputs (in use)", C_BUY)]:
            fig.add_scatter(x=Q.quarter, y=Q[col], name=name, mode="lines+markers", line=dict(color=color, width=3 if "v3" in col else 2))
        fig.add_vrect(x0=-0.5, x1=8.5, fillcolor="#888888", opacity=0.08, line_width=0)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Best-guess demand error (MWh per hour)",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
        T = pd.read_csv(DATA / "rolling_dm_tests.csv")
        T = T[T.period == "forecast weather"][["comparison", "days", "mean_diff", "dm_stat", "p_value"]].copy()
        T["comparison"] = T.comparison.str.replace("maya_v3", "Maya v3").str.replace("maya_v2", "Maya v2")
        T["p_value"] = T.p_value.map(lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}")
        st.dataframe(T.rename(columns={"comparison": "Comparison", "days": "Days", "mean_diff": "Error difference (MWh/h)",
                                       "dm_stat": "Diebold-Mariano statistic", "p_value": "p-value"}).round(3),
                     hide_index=True, use_container_width=True)
        st.caption("Each quarter is forecast by a model trained only on earlier years (2020 left out). Shaded quarters (2022 to "
                   "Q1 2024) use actual weather because no forecast archive exists before February 2024, so they flatter both "
                   "Maya versions; from Q2 2024 on, inputs are the real day-ahead ECMWF forecasts. Gradient boosting best guess "
                   "only. The test table covers the forecast-weather quarters; negative = the first model has lower error.")
    st.divider()
    st.markdown("""
**The question.** Maui Hiva buys power the day before. Buying short means paying same-day prices for the gap, which costs
far more than carrying a small surplus. So Maya v2 answers "how much should we buy?", not "what is the single most likely number?".

**1. A range instead of one number.** Two models (gradient boosting and an LSTM) are trained to predict levels, not averages:
P50 (best guess) up to P98 for demand and net load, and P50 down to P02 for renewable output (the amount you can count on).
P95 means demand should come in above it on only about 5% of hours.

**2. Daily recalibration.** Every day each level is checked against the last 60 finished days and nudged until it really is
short that often (adaptive conformal calibration). This is why the coverage chart sits on target in both the backtest and 2026.

**3. Inputs.** ECMWF day-ahead weather for the Milford point, plus a population-weighted temperature over eight PACE load
centers (Salt Lake City, Ogden, Provo, Logan, Cedar City, Rexburg, Casper, Rock Springs). The ECMWF city forecast runs about
0.9 C colder than the ERA5 reanalysis the model trains on, so it is shifted by its own trailing 60-day bias, which is known at the
deadline. Calendar inputs cover weekday, federal and Utah holidays (Pioneer Day, Christmas Eve, day after Thanksgiving), the days
next to them, and Utah school breaks. Together these cut best-guess error by about 12% (model v3.0, Oct 20, 2026).

**3b. Honest timing.** The buy deadline is taken as 10 am Mountain the day before. Every hour of the buy day then has demand from two
days earlier available, plus the newest reading on the morning of the buy and ECMWF day-ahead weather. The original Maya used
yesterday's same-hour demand, which is not all known at a morning buy; with that, Maya would be roughly 10 to 20% cheaper again
(grey line on the Maya vs PACE tab). If Maui Hiva's real deadline is later in the day, the advantage grows.

**4. Risk tier.** A small classifier flags days likely to have a short hour. On Yellow and Red days the rule buys P98 instead of P95.

**5. Bid ladder (recommended way to buy).** Block 1, the Maya-rule amount, is bid at any price. Block 2, the extra cushion up to
P98 on Green days, is bid at a limit equal to the expected real-time price for that hour: recent normal real-time price plus the
hour's spike risk times a typical spike premium. It is bought only when day-ahead clears at or below that limit. At real May to
Oct 2026 prices this was the best option tested (-\\$0.19/MWh vs perfect foresight, short on 2.7% of hours).

**6. Load shifting.** Expected day-ahead price per hour = the same hour's average day-ahead price over the last 7 days (known
at the deadline). The Shift load tab moves flexible MW out of tomorrow's most expensive expected hours into the cheapest ones.
On May to Oct 2026 prices, moving 1 MW for 4 hours a day saved about \\$108 per day, 92% of the perfect-hindsight value.

**7. Sized to Maui Hiva.** Every MWh and dollar figure can be shown at Maui Hiva's own size (sidebar), assuming its hourly
pattern follows PACE's net load. Replace the 100 MW placeholder with their real average load, or their meter data later.

**8. Live.** A daily job pulls PACE demand from EIA (free API) and ECMWF forecasts from Open-Meteo, rebuilds the Milford renewable
estimate, forecasts the next buy day, recalibrates, and scores yesterday. It runs three times before the deadline (GitHub can
start jobs late); the last forecast before the deadline is locked, so the track record shows only what a buyer would really have had. The track record tab is that job run over every day since
the training data ends.

**9. Week-ahead outlook.** A separate model forecasts daily net-load totals 2 to 7 days out from ECMWF forecasts at those lead
times, with ranges calibrated per lead time. Daily error is about 3%, roughly 40% better than repeating last week.

**10. Monthly retraining.** On the 1st of each month a challenger model is trained on all data up to 45 days ago and compared
with the current model on those last 45 days, which neither has seen. It replaces the current model only if its error is at
least 1% lower on demand and no worse on renewable. Past forecasts are archived as issued and never recomputed, so a new model
cannot rewrite the track record.

**Data and limits**
- PACE demand and PACE's day-ahead forecast: EIA-930, hourly. EIA posts PACE's forecast shortly before the day, so it is a benchmark, not an input.
- Weather: ECMWF IFS 0.25 deg via Open-Meteo, point 38.42 N, 113.02 W, plus eight load-center cities (ERA5 for training and bias checks).
  If the city forecast cannot be fetched on a given day, the daily job falls back to the v2 model rather than skip the buy. Free for non-commercial use; a commercial plan is needed after handoff.
- Milford renewable output is modeled (pvlib tracker model calibrated to EIA-923 monthly plant totals), not metered. Live hours use the
  2025 average calibration until EIA-923 publishes the month.
- Risk tier skill is modest (AUC about 0.65). It is useful for nudging the level, not a guarantee.
- Prices: PacifiCorp has traded in CAISO's Extended Day-Ahead Market (EDAM) since May 1 2026. Day-ahead and 15-minute
  real-time prices for the PACE load point come from CAISO OASIS (free) and update with each daily run.
- At May to Oct 2026 prices, real-time ran slightly above day-ahead on average, so a cushion paid for itself, and the dollar
  differences between strategies were cents per MWh. The cost scenarios in the sidebar show what happens in a stress season.
- Results assume Maui Hiva is a price taker: its purchases are too small to move PACE prices.
""")
