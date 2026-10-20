"""
One-page client summary (PDF) of Maya v2, built from the current results so the numbers always match the dashboard.
Run by the daily job. Output: app/data/maya_v2_summary.pdf (download button in the app sidebar).
Dollar figures use the dashboard's placeholder size (MW_PLACEHOLDER) until Maui Hiva confirms its load.
"""
import io
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APPDATA, LIVE = os.path.join(ROOT, "app", "data"), os.path.join(ROOT, "live")
OUT = os.path.join(APPDATA, "maya_v2_summary.pdf")
MW_PLACEHOLDER = 100.0
FLEX_MW, SHIFT_H = 5.0, 4
URL_APP, URL_REPO = "maya-v2-buy-ahead.streamlit.app", "github.com/RendellRich/maya-forecasting-v2"
GREEN, ORANGE, GREY, INK = "#0f8a6a", "#eb6834", "#9a9a9a", "#1f1f1f"


def money(x):
    return f"${x:,.0f}"


def numbers():
    bt = pd.read_parquet(os.path.join(APPDATA, "backtest_hourly.parquet"))
    scale = MW_PLACEHOLDER / bt["net_actual"].mean()
    b26 = bt[bt.period == "B forward test 2026"]
    mae_maya = (b26.dem_actual - b26.dem_P50).abs().mean()
    mae_pace = (b26.dem_actual - b26.dem_pace).abs().mean()
    s = pd.read_parquet(os.path.join(APPDATA, "strategies_2026.parquet"))
    s = s[s.da.notna() & s.rt.notna() & s.actual.notna()]
    res = {}
    for k in ["persistence", "pace", "maya_ladder"]:
        g = s.actual - s[k]
        res[k] = dict(bill=float((s.da * s[k] + s.rt * g).sum() * scale), short=float((g > 0).mean() * 100))
    ci = pd.read_csv(os.path.join(APPDATA, "savings_ci.csv")).set_index(["maya", "baseline"])
    sb = pd.read_csv(os.path.join(APPDATA, "shift_backtest.csv"))
    track = pd.read_csv(os.path.join(LIVE, "track_record.csv"))
    summ = json.load(open(os.path.join(LIVE, "summary.json")))
    return dict(scale=scale, mae_maya=mae_maya, mae_pace=mae_pace, res=res, ci=ci, sb=sb, track=track, summ=summ,
                start=s.index.min(), end=s.index.max(), days=len(s) // 24)


def chart(n):
    lab = ["No forecast", "PACE's number", "Maya bid ladder"]
    keys = ["persistence", "pace", "maya_ladder"]
    short = [n["res"][k]["short"] for k in keys]
    base = n["res"]["persistence"]["bill"]
    saved = [base - n["res"][k]["bill"] for k in keys]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.0), dpi=200)
    col = [GREY, ORANGE, GREEN]
    ax[0].barh(lab, short, color=col)
    ax[0].set_title("Hours bought short", fontsize=9, loc="left")
    for i, v in enumerate(short):
        ax[0].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    ax[0].set_xlim(0, max(short) * 1.25)
    ax[1].barh(lab, [x / 1000 for x in saved], color=col)
    ax[1].set_title(f"Saved vs no forecast ($k, {MW_PLACEHOLDER:.0f} MW)", fontsize=9, loc="left")
    for i, v in enumerate(saved):
        ax[1].text(v / 1000 + 2, i, money(v), va="center", fontsize=8)
    ax[1].set_xlim(min(0, min(saved) / 1000), max(saved) / 1000 * 1.35)
    ax[1].set_yticklabels([])
    for a in ax:
        a.tick_params(labelsize=8)
        for sp in ["top", "right"]:
            a.spines[sp].set_visible(False)
        a.invert_yaxis()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def build():
    n = numbers()
    st_t = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=17, leading=20, textColor=INK)
    st_s = ParagraphStyle("s", fontName="Helvetica", fontSize=9, leading=12, textColor="#555555")
    st_h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=2, textColor=INK)
    st_b = ParagraphStyle("b", fontName="Helvetica", fontSize=9, leading=12, textColor=INK)
    st_n = ParagraphStyle("n", fontName="Helvetica", fontSize=7.5, leading=9.5, textColor="#666666")
    ci_np = n["ci"].loc[("maya_ladder", "persistence")]
    ci_pc = n["ci"].loc[("maya_ladder", "pace")]
    sc = n["scale"]
    plan = n["sb"][f"plan_k{SHIFT_H}"] * FLEX_MW
    hind = n["sb"][f"hindsight_k{SHIFT_H}"] * FLEX_MW
    tr = n["track"]
    tr_short = tr.short_hours.sum() / tr.hours.sum() * 100
    tr_pace = tr.pace_short_hours.sum() / tr.hours.sum() * 100
    tr_mae = (tr.maya_P50_MAE * tr.hours).sum() / tr.hours.sum()
    tr_pmae = (tr.pace_MAE * tr.hours).sum() / tr.hours.sum()

    story = [Paragraph("Maya v2: how much power to buy the day before", st_t),
             Paragraph(f"Maui Hiva Innovations | Purdue MS Data Science capstone, Project T | updated {pd.Timestamp.now('UTC'):%B %d, %Y} | "
                       f"live dashboard: {URL_APP}", st_s), Spacer(1, 6),
             Paragraph("What it does", st_h),
             Paragraph("Every morning before the 10 am Mountain day-ahead deadline, Maya forecasts tomorrow's net load for the PacifiCorp East "
                       "area (demand minus Milford, Utah renewable output) as a range, not a single number, and turns it into a buy plan. "
                       "Buying short means paying the real-time price for the gap, so the plan is built to be short rarely and on purpose. "
                       "It also flags hours at risk of a price spike, suggests when to run flexible load, and gives a 7-day outlook of daily "
                       "net load for weekly planning.", st_b),
             Paragraph("Results", st_h)]
    rows = [["", "Maya", "Comparison"],
            ["Best-guess error, 2026 (MWh per hour)", f"{n['mae_maya']:,.0f}", f"PACE's own forecast: {n['mae_pace']:,.0f} ({(n['mae_pace'] - n['mae_maya']) / n['mae_pace'] * 100:.0f}% higher)"],
            [f"Live, {len(tr)} unseen days: hours bought short", f"{tr_short:.1f}%", f"PACE's number: {tr_pace:.0f}%"],
            [f"Live best-guess error (MWh per hour)", f"{tr_mae:,.0f}", f"PACE: {tr_pmae:,.0f}"],
            [f"Saved vs buying with no forecast*", money(ci_np.saved_total * sc), f"90% range {money(ci_np.saved_lo * sc)} to {money(ci_np.saved_hi * sc)}"],
            [f"Saved vs buying PACE's number*", money(ci_pc.saved_total * sc), f"90% range {money(ci_pc.saved_lo * sc)} to {money(ci_pc.saved_hi * sc)}"],
            [f"Load shifting, {FLEX_MW:.0f} MW moved {SHIFT_H} h a day*", money(plan.sum()), f"{plan.sum() / hind.sum() * 100:.0f}% of the perfect-hindsight value; saved money on {(plan > 0).mean() * 100:.0f}% of days"]]
    t = Table(rows, colWidths=[2.55 * inch, 0.95 * inch, 3.7 * inch])
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8.5), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
                           ("FONT", (1, 1), (1, -1), "Helvetica-Bold", 9), ("TEXTCOLOR", (1, 1), (1, -1), colors.HexColor(GREEN)),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#999999")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story += [t, Spacer(1, 4), Image(chart(n), width=7.2 * inch, height=2.0 * inch),
              Paragraph(f"*Real CAISO day-ahead and real-time prices for the PACE area, {n['start']:%b %d} to {n['end']:%b %d, %Y} ({n['days']} days), "
                        f"at a placeholder size of {MW_PLACEHOLDER:.0f} MW until Maui Hiva confirms its load. 90% ranges from a bootstrap over whole days.", st_n),
              Paragraph("How it works", st_h),
              Paragraph("<b>1. A calibrated range.</b> Two models (gradient boosting and an LSTM) forecast levels from P50 (best guess) to P98, "
                        "using ECMWF weather forecasts and only data available at the deadline. Each level is re-checked daily, so P95 really "
                        "is short about 5% of the time.", st_b),
              Paragraph("<b>2. Risk and the bid ladder.</b> A daily Green / Yellow / Red tier sets the always-bought amount (P95, or P98 on risky "
                        "days). An hourly spike-risk model sets a price limit for an extra block up to P98, bought only when day-ahead is cheaper "
                        "than expected real-time.", st_b),
              Paragraph("<b>3. Load shifting.</b> Expected hourly prices (last 7 days of day-ahead results) show which hours to move flexible load "
                        "out of and into.", st_b),
              Paragraph("<b>4. Runs itself.</b> A daily job pulls EIA demand, ECMWF weather and CAISO prices, forecasts, and scores the track "
                        "record. A monthly check retrains the models and switches only if the new one is better on unseen days.", st_b),
              Paragraph("Limits", st_h),
              Paragraph("Day-ahead prices for PACE exist only since May 2026 (CAISO EDAM), so dollar results rest on a short window. Milford "
                        "renewable output is modeled, not metered. Maui Hiva's own load is assumed to follow PACE's pattern. On ordinary days "
                        "the buy strategies differ by cents per MWh; Maya's main value is being short far less often, which is what matters "
                        "on spike days.", st_b),
              Spacer(1, 4), Paragraph(f"Dashboard: {URL_APP}  |  Code and data: {URL_REPO}", st_s)]
    doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.5 * inch, bottomMargin=0.45 * inch,
                            title="Maya v2 summary", author="Project T, Purdue University")
    doc.build(story)
    return OUT


if __name__ == "__main__":
    print(build())
