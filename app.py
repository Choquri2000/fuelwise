"""FuelWise - Streamlit UI. Data stays in memory for the session; nothing is stored or sent anywhere."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

import fuelwise as fw

st.set_page_config(page_title="FuelWise - what does your car really cost?", page_icon="⛽", layout="wide")

st.title("⛽ FuelWise")
st.caption("Upload a fuel log (Fuelly, Drivvo, Fuelio, a spreadsheet) or type fill-ups in. Get your true consumption, "
           "running costs and a heads-up if something changes. **Private by design:** your data is processed in "
           "memory during your session and is never stored. No AI services are called.")

with st.sidebar:
    st.header("Units")
    dist_unit = st.selectbox("Distance unit in your data", list(fw.DIST_UNITS), index=0)
    vol_unit = st.selectbox("Fuel volume unit in your data", list(fw.VOL_UNITS), index=0)
    metric_default = {"L": 0, "US gal": 1, "UK gal": 2}[vol_unit]
    metric = st.selectbox("Show consumption as", fw.CONS_METRICS, index=metric_default)
    currency = st.text_input("Currency symbol", value="€", max_chars=4)
    st.markdown("---")
    st.caption("Everything here is an estimate from the data you give it. Not mechanical or financial advice.")

DEMO_PATH = Path(__file__).with_name("demo_fuel_log.csv")
use_demo = st.checkbox("Use demo data", value=False,
                       help="A synthetic log (km and litres) with a planted consumption jump.")

raw = None
if use_demo:
    raw = fw.make_demo()
    dist_unit, vol_unit = "km", "L"
    st.info("Showing a synthetic demo log (km, litres). Untick the box to use your own data.")
    if DEMO_PATH.exists():
        st.download_button("Download the demo CSV", DEMO_PATH.read_bytes(), "demo_fuel_log.csv", "text/csv")
else:
    tab_up, tab_type = st.tabs(["Upload a CSV", "Type entries"])
    with tab_up:
        up = st.file_uploader("Fuel log CSV", type=["csv", "txt"])
        if up is not None:
            try:
                data = up.getvalue()
                raw = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding_errors="replace")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read that file as CSV: {exc}")
    with tab_type:
        st.caption("Add one row per fill-up. Tick *Full tank* when you filled to the brim. Leave Total cost or "
                   "Price per unit blank if you only know one of them.")
        blank = pd.DataFrame({"Date": pd.Series(dtype="datetime64[ns]"), "Odometer": pd.Series(dtype="float"),
                              "Volume": pd.Series(dtype="float"), "Price per unit": pd.Series(dtype="float"),
                              "Total cost": pd.Series(dtype="float"), "Full tank": pd.Series(dtype="bool")})
        edited = st.data_editor(blank, num_rows="dynamic", use_container_width=True, key="editor",
                                column_config={"Date": st.column_config.DateColumn("Date"),
                                               "Full tank": st.column_config.CheckboxColumn("Full tank", default=True)})
        if raw is None and len(edited.dropna(how="all", subset=["Odometer", "Volume"])) > 0:
            raw = edited

if raw is None:
    st.write("👆 Add data to begin, or tick **Use demo data** to see what you will get.")
    st.stop()

try:
    res = fw.clean_log(raw, None, dist_unit, vol_unit)
except ValueError as exc:
    st.error(str(exc))
    st.write("Columns found in your file:", ", ".join(map(str, raw.columns)))
    st.stop()

with st.expander("How your columns were read", expanded=False):
    st.write({role: str(col) for role, col in res.mapping.items()})
    for n in res.notes:
        st.warning(n)

a = fw.analyse(res.data)
s = a.stats
st.subheader("Summary")
st.success(fw.summary_text(a, metric, dist_unit, vol_unit, currency))

dshort = "mile" if dist_unit == "miles" else "km"
c = st.columns(4)
if "avg_l100" in s:
    c[0].metric("Average consumption", f"{fw.conv_consumption(s['avg_l100'], metric):.1f} {metric}")
    c[1].metric("Latest (rolling 3)", f"{fw.conv_consumption(s['last_l100'], metric):.1f} {metric}")
if "cost_per_km" in s:
    c[2].metric(f"Fuel cost per {dshort}", f"{currency}{fw.cost_per_distance(s['cost_per_km'], dist_unit):.3f}")
if "per_month" in s:
    c[3].metric("Per month / per year", f"{currency}{s['per_month']:,.0f} / {currency}{s['per_year']:,.0f}")

iv = a.intervals
if len(iv):
    st.subheader("Consumption over time")
    chart = pd.DataFrame({
        "Per fill-up": fw.conv_consumption(iv.l_per_100km.where(~iv.suspect), metric),
        "Rolling (3 fill-ups)": fw.conv_consumption(iv.rolling_l100, metric)})
    chart.index = iv.date if iv.date.notna().all() else iv.odo_km / fw.DIST_UNITS[dist_unit]
    st.line_chart(chart)
    st.caption("Calculated full-tank to full-tank: partial fills in between are added to the next full fill-up, so "
               "they do not distort the numbers. " + ("Higher is better for mpg and km/L." if not fw.lower_is_better(metric) else "Lower is better."))

st.subheader("Things that may be worth checking")
if not a.flags:
    st.write("Nothing stands out. 👍")
for f in a.flags:
    (st.warning if f["kind"] in ("rise", "spike") else st.info)(f["text"])

mc = fw.monthly_costs(res.data)
if len(mc):
    st.subheader("Money")
    m1, m2 = st.columns(2)
    with m1:
        st.caption(f"Fuel spend per month ({currency})")
        st.bar_chart(mc.set_index("month")["spent"])
    with m2:
        st.caption(f"Average price per {vol_unit} ({currency})")
        st.line_chart(fw.price_per_volume(mc.set_index("month")["avg_price_per_l"], vol_unit))
    yc = fw.yearly_costs(res.data)
    if len(yc) > 1:
        st.dataframe(yc.rename(columns={"year": "Year", "spent": f"Spent ({currency})", "litres": "Litres"}).round(1),
                     hide_index=True)

with st.expander("Optional: total cost of ownership"):
    st.caption("Enter your own yearly figures. Nothing is looked up. Fuel uses your log's yearly pace when available.")
    k1, k2, k3 = st.columns(3)
    ins = k1.number_input(f"Insurance per year ({currency})", min_value=0.0, value=0.0, step=50.0)
    mnt = k2.number_input(f"Maintenance, tyres, repairs per year ({currency})", min_value=0.0, value=0.0, step=50.0)
    dep = k3.number_input(f"Depreciation per year ({currency})", min_value=0.0, value=0.0, step=100.0)
    if "per_year" in s and s.get("km_per_year"):
        t = fw.tco(s["km_per_year"], s["per_year"], ins, mnt, dep)
        st.write(f"**{currency}{t['per_year']:,.0f} per year** ({currency}{t['per_month']:,.0f} per month, "
                 f"{currency}{t['per_km'] * fw.DIST_UNITS[dist_unit]:.2f} per {dshort}); fuel is "
                 f"{t['fuel_share'] * 100:.0f}% of it.")
    else:
        st.write("Add dates covering at least a month so a yearly fuel pace can be estimated.")

st.subheader("Cleaned data")
show = res.data.copy()
show["date"] = show["date"].dt.strftime("%Y-%m-%d")
show = show.rename(columns={"odo_km": "odometer_km", "litres": "litres", "price_per_l": "price_per_litre",
                            "total": "total_cost", "full": "full_tank", "missed": "missed_fillup",
                            "issue": "ignored_because"})
st.dataframe(show.round(3), hide_index=True)
st.download_button("Download cleaned CSV", show.to_csv(index=False).encode("utf-8"), "fuelwise_cleaned.csv", "text/csv")
st.caption("Cleaned file is always in km and litres. Rows with a reason in *ignored_because* were left out of calculations.")
