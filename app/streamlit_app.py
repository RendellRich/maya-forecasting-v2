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


bt = load_backtest()
live = load_live()
prices = load_prices()

# ----------------------------------------------------------------------------- sidebar: the client's costs
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
           "Every forecast uses only what is known at the buy deadline (10 am Mountain the day before).")

tab_buy, tab_track, tab_bt, tab_risk, tab_how = st.tabs(
    ["Tomorrow's buy", "Live track record", "Maya vs PACE", "Risk score", "How it works"])

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
        rule = st.radio("Buy rule", ["Maya rule (P95 on Green days, P98 on Yellow or Red days)", "Fixed P90", "Fixed P95", "Fixed P98"],
                        horizontal=True)
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
            if d.recent_err > 350:
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
        st.caption(f"Run {S['run_utc']} UTC. Weather: {S['weather']}. Net load = PACE demand minus Milford-area renewable output.")

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
            f"average worst hour short: {r.avg_worst_short_MWh:.0f} MWh")
    auc = rv[(rv.version == "AUC") & (rv.period == pick) & (rv.tier == "v2 classifier")].share_of_days_pct
    st.caption(f"Ranking skill (AUC) {float(auc.iloc[0]):.2f} (0.5 = coin flip). A model-uncertainty-only version scored about 0.5, "
               "because daily recalibration already absorbs it. The tier's value is in moving the buy level, below.")
    st.subheader("Does stepping up on risky days pay?")
    ra = load_csv("lag48_risk_adjusted_buy.csv")
    ra = ra[ra.period == pick].drop(columns=["period"])
    st.dataframe(ra.rename(columns={"policy": "Buy rule", "short_hours_pct": "Short hours %", "avg_short_MWh": "Avg short MWh/h",
                                    "avg_surplus_MWh": "Avg surplus MWh/h", "worst_short_MWh": "Worst hour short",
                                    "cost_K19": "Cost index (short = 19x)", "cost_K9": "Cost index (short = 9x)"}).drop(columns=["hours"]),
                 hide_index=True, use_container_width=True)
    st.caption("Demand. Stepping up to P98 on Yellow and Red days cuts short hours by about a quarter and lowers the worst hour, "
               "for about 1% more cost at a 19x ratio. Cost index = average per hour of 19 x short MWh + surplus MWh.")
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
    st.markdown("""
**The question.** Maui Hiva buys power the day before. Buying short means paying same-day prices for the gap, which costs
far more than carrying a small surplus. So Maya v2 answers "how much should we buy?", not "what is the single most likely number?".

**1. A range instead of one number.** Two models (gradient boosting and an LSTM) are trained to predict levels, not averages:
P50 (best guess) up to P98 for demand and net load, and P50 down to P02 for renewable output (the amount you can count on).
P95 means demand should come in above it on only about 5% of hours.

**2. Daily recalibration.** Every day each level is checked against the last 60 finished days and nudged until it really is
short that often (adaptive conformal calibration). This is why the coverage chart sits on target in both the backtest and 2026.

**3. Honest inputs.** The buy deadline is taken as 10 am Mountain the day before. Every hour of the buy day then has demand from two
days earlier available, plus the newest reading on the morning of the buy and ECMWF day-ahead weather. The original Maya used
yesterday's same-hour demand, which is not all known at a morning buy; with that, Maya would be roughly 10 to 20% cheaper again
(grey line on the Maya vs PACE tab). If Maui Hiva's real deadline is later in the day, the advantage grows.

**4. Risk tier.** A small classifier flags days likely to have a short hour. On Yellow and Red days the rule buys P98 instead of P95.

**5. Live.** A daily job pulls PACE demand from EIA (free API) and ECMWF forecasts from Open-Meteo, rebuilds the Milford renewable
estimate, forecasts the next buy day, recalibrates, and scores yesterday. The track record tab is that job run over every day since
the training data ends.

**Data and limits**
- PACE demand and PACE's day-ahead forecast: EIA-930, hourly. EIA posts PACE's forecast shortly before the day, so it is a benchmark, not an input.
- Weather: ECMWF IFS 0.25 deg via Open-Meteo, point 38.42 N, 113.02 W. Free for non-commercial use; a commercial plan is needed after handoff.
- Milford renewable output is modeled (pvlib tracker model calibrated to EIA-923 monthly plant totals), not metered. Live hours use the
  2025 average calibration until EIA-923 publishes the month.
- Risk tier skill is modest (AUC about 0.65). It is useful for nudging the level, not a guarantee.
- Prices: PacifiCorp has traded in CAISO's Extended Day-Ahead Market (EDAM) since May 1 2026. Day-ahead and 15-minute
  real-time prices for the PACE load point come from CAISO OASIS (free) and update with each daily run.
- At May to Oct 2026 prices, real-time ran slightly above day-ahead on average, so a cushion paid for itself, and the dollar
  differences between strategies were cents per MWh. The cost scenarios in the sidebar show what happens in a stress season.
- Results assume Maui Hiva is a price taker: its purchases are too small to move PACE prices.
""")
