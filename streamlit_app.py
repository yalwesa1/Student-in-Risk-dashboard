"""
Student Risk Dashboard - Streamlit app.

Interactive version of student_risk_dashboard.html. Reuses the exact same
synthetic-data generation and rule-based risk scoring from
student_risk_analysis.py, and mirrors the HTML dashboard's structure:

    - Stat tiles (total / high / medium / low)
    - Overall risk distribution
    - Risk level by category (stacked)
    - Overall risk score distribution (histogram with threshold lines)
    - Most common risk factors among at-risk students
    - At-risk student list (filter / search / sort / click-to-expand drivers)

Run with:
    streamlit run streamlit_app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from student_risk_analysis import (
    build_dashboard_data,
    categorize_risk,
    generate_survey_data,
)

# --- Palette (matches the HTML dashboard) -----------------------------------
LEVEL_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}
CATEGORY_COLORS = {"Academic": "#2a78d6", "Financial": "#1baf7a", "Personal": "#eda100"}
LEVELS = ["Low", "Medium", "High"]

level_scale = alt.Scale(domain=LEVELS, range=[LEVEL_COLORS[l] for l in LEVELS])
category_scale = alt.Scale(
    domain=list(CATEGORY_COLORS), range=list(CATEGORY_COLORS.values())
)

st.set_page_config(page_title="Student at Risk Dashboard", page_icon="🎓", layout="wide")

N_STUDENTS = 500
SEED = 42


# --- Data (cached; regenerated only when controls change) -------------------
@st.cache_data(show_spinner="Generating and scoring survey data...")
def load_data(n_students: int, seed: int):
    raw = generate_survey_data(n_students=n_students, seed=seed)
    scored = categorize_risk(raw)
    data = build_dashboard_data(scored)  # reads scored.attrs while still live
    return scored, data


# --- Sidebar ------------------------------------------------------------
with st.sidebar:
    st.header("About this data")
    st.caption(
        "Synthetic onboarding-survey responses, scored with a weighted "
        "rule-based rubric across Academic (40%), Financial (35%) and "
        "Personal (25%) risk dimensions."
    )

scored_df, data = load_data(N_STUDENTS, SEED)
total = data["total_students"]
oc = data["overall_counts"]
med_thr = data["thresholds"]["medium"]
high_thr = data["thresholds"]["high"]

# --- Header -----------------------------------------------------------------
st.title("🎓 Student at Risk Dashboard")
st.subheader("Designed by SUNY Academic Momentum Fellows Program")
st.caption(
    f"Synthetic onboarding-survey data · {total} students scored across "
    "Academic, Financial, and Personal risk dimensions"
)


# --- Stat tiles -------------------------------------------------------------
def tile(col, label, value, sub=None, accent=None):
    border = f"border-left:4px solid {accent};" if accent else ""
    col.markdown(
        f"""
        <div style="background:rgba(128,128,128,0.06);border:1px solid rgba(128,128,128,0.2);
                    {border}border-radius:10px;padding:14px 16px;">
          <div style="font-size:12px;text-transform:uppercase;letter-spacing:.04em;
                      opacity:.7;">{label}</div>
          <div style="font-size:30px;font-weight:700;line-height:1.1;">{value}</div>
          <div style="font-size:12px;opacity:.6;margin-top:2px;">{sub or ""}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


c1, c2, c3, c4 = st.columns(4)
tile(c1, "Total students", total)
tile(c2, "High risk", oc["High"], f"{oc['High']/total*100:.1f}% of students", LEVEL_COLORS["High"])
tile(c3, "Medium risk", oc["Medium"], f"{oc['Medium']/total*100:.1f}% of students", LEVEL_COLORS["Medium"])
tile(c4, "Low risk", oc["Low"], f"{oc['Low']/total*100:.1f}% of students", LEVEL_COLORS["Low"])

st.write("")

# --- Charts row 1: overall distribution + by category -----------------------
left, right = st.columns(2)

with left:
    st.subheader("Overall risk distribution")
    st.caption("Students grouped by combined risk level (40% academic, 35% financial, 25% personal).")
    overall_df = pd.DataFrame(
        {"Level": LEVELS, "Students": [oc[l] for l in LEVELS]}
    )
    overall_df["Level"] = pd.Categorical(overall_df["Level"], categories=LEVELS, ordered=True)
    chart = (
        alt.Chart(overall_df)
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X("Students:Q", title=None),
            y=alt.Y("Level:N", sort=LEVELS, title=None, axis=alt.Axis(labelOverlap=False)),
            color=alt.Color("Level:N", scale=level_scale, legend=None),
            tooltip=["Level", "Students"],
        )
        .properties(height=180)
    )
    labels = chart.mark_text(align="left", dx=4, color="gray").encode(
        text=alt.Text("Students:Q")
    )
    st.altair_chart(chart + labels, use_container_width=True)

with right:
    st.subheader("Risk level by category")
    st.caption("Share of students at each risk level, broken out per dimension.")
    cc = data["category_counts"]
    rows = []
    for cat_key, label in [("academic", "Academic"), ("financial", "Financial"), ("personal", "Personal")]:
        for order_idx, lvl in enumerate(LEVELS):
            rows.append({"Category": label, "Level": lvl, "Students": cc[cat_key][lvl], "order": order_idx})
    cat_df = pd.DataFrame(rows)
    stacked = (
        alt.Chart(cat_df)
        .mark_bar()
        .encode(
            x=alt.X("Students:Q", stack="normalize", title="Share of students", axis=alt.Axis(format="%")),
            y=alt.Y("Category:N", sort=["Academic", "Financial", "Personal"], title=None,
                    axis=alt.Axis(labelOverlap=False, labelPadding=6)),
            color=alt.Color("Level:N", scale=level_scale, sort=LEVELS,
                            legend=alt.Legend(orient="top", title=None)),
            order=alt.Order("order:Q", sort="ascending"),
            tooltip=["Category", "Level", "Students"],
        )
        .properties(height=180)
    )
    st.altair_chart(stacked, use_container_width=True)

# --- Chart row 2: histogram -------------------------------------------------
st.subheader("Overall risk score distribution")
st.caption(f"Dashed lines mark the Medium (≥{med_thr}) and High (≥{high_thr}) thresholds.")
hist_df = pd.DataFrame(data["histogram"])
hist_df["mid"] = (hist_df["start"] + hist_df["end"]) / 2
bars = (
    alt.Chart(hist_df)
    .mark_bar(color="#256abf")
    .encode(
        x=alt.X("start:Q", bin="binned", title="Overall risk score", scale=alt.Scale(domain=[0, 100])),
        x2="end:Q",
        y=alt.Y("count:Q", title="Students"),
        tooltip=[alt.Tooltip("start:Q", title="From"), alt.Tooltip("end:Q", title="To"),
                 alt.Tooltip("count:Q", title="Students")],
    )
)
rules_df = pd.DataFrame(
    {"x": [med_thr, high_thr], "label": [f"Medium ≥{med_thr}", f"High ≥{high_thr}"],
     "color": [LEVEL_COLORS["Medium"], LEVEL_COLORS["High"]]}
)
rules = (
    alt.Chart(rules_df)
    .mark_rule(strokeDash=[5, 4], size=2)
    .encode(x="x:Q", color=alt.Color("color:N", scale=None))
)
rule_text = (
    alt.Chart(rules_df)
    .mark_text(align="left", dx=4, dy=-6, fontSize=11)
    .encode(x="x:Q", text="label:N", color=alt.Color("color:N", scale=None))
)
st.altair_chart((bars + rules + rule_text).properties(height=260), use_container_width=True)

# --- Chart row 3: most common risk factors ----------------------------------
st.subheader("Most common risk factors among at-risk students")
st.caption("How often each survey answer appears among Medium/High-risk students' top drivers.")
freq_df = pd.DataFrame(data["top_driver_freq"])
if not freq_df.empty:
    freq_chart = (
        alt.Chart(freq_df)
        .mark_bar(cornerRadius=4, height=20)
        .encode(
            x=alt.X("count:Q", title="Number of at-risk students"),
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("category:N", scale=category_scale,
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["label", "category", "count"],
        )
        .properties(height=28 * len(freq_df) + 30)
    )
    st.altair_chart(freq_chart, use_container_width=True)

st.divider()

# --- At-risk student list ---------------------------------------------------
st.subheader("At-risk student list")
st.caption("Every Medium or High risk student with the specific survey answers driving that score. "
           "Click a row to see all contributing answers.")

fcol, scol = st.columns([2, 3])
with fcol:
    level_filter = st.radio("Filter", ["All", "High only", "Medium only"], horizontal=True, label_visibility="collapsed")
with scol:
    query = st.text_input("Search", placeholder="Search student ID...", label_visibility="collapsed")

students = data["at_risk_students"]
if level_filter == "High only":
    students = [s for s in students if s["overall_level"] == "High"]
elif level_filter == "Medium only":
    students = [s for s in students if s["overall_level"] == "Medium"]
if query:
    students = [s for s in students if query.lower() in s["id"].lower()]

st.caption(f"Showing {len(students)} of {len(data['at_risk_students'])} at-risk students")

table_df = pd.DataFrame([
    {
        "Student": s["id"],
        "Level": s["overall_level"],
        "Overall": s["overall_score"],
        "Academic": s["academic_score"],
        "Financial": s["financial_score"],
        "Personal": s["personal_score"],
        "Primary": s["primary_category"],
        "Top risk factors": " · ".join(d["detail"] for d in s["drivers"][:3]),
    }
    for s in students
])

if table_df.empty:
    st.info("No students match the current filter.")
else:
    event = st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Overall": st.column_config.ProgressColumn("Overall", min_value=0, max_value=100, format="%.1f"),
            "Academic": st.column_config.NumberColumn(format="%.1f"),
            "Financial": st.column_config.NumberColumn(format="%.1f"),
            "Personal": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    selected = event.selection.rows if event and event.selection else []
    if selected:
        s = students[selected[0]]
        st.markdown(f"#### {s['id']} — all contributing survey answers")
        meta = st.columns(4)
        meta[0].metric("Overall", f"{s['overall_score']:.1f}", s["overall_level"])
        meta[1].metric("Academic", f"{s['academic_score']:.1f}")
        meta[2].metric("Financial", f"{s['financial_score']:.1f}")
        meta[3].metric("Personal", f"{s['personal_score']:.1f}")
        for d in s["drivers"]:
            color = CATEGORY_COLORS.get(d["category"], "#888")
            st.markdown(
                f"<span style='display:inline-block;border-left:3px solid {color};"
                f"padding:2px 10px;margin:3px 0;background:rgba(128,128,128,0.06);"
                f"border-radius:4px;'>{d['detail']}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Select a row above to expand that student's full risk-driver breakdown.")

# --- Downloads --------------------------------------------------------------
st.divider()
d1, d2 = st.columns(2)
d1.download_button(
    "Download full scored dataset (CSV)",
    scored_df.to_csv(index=False).encode("utf-8"),
    file_name="student_risk_scored.csv",
    mime="text/csv",
)
at_risk_export = pd.DataFrame([
    {
        "student_id": s["id"], "overall_risk_score": s["overall_score"],
        "overall_risk_level": s["overall_level"], "primary_risk_category": s["primary_category"],
        "academic_risk_score": s["academic_score"], "financial_risk_score": s["financial_score"],
        "personal_risk_score": s["personal_score"],
        "risk_drivers": "; ".join(d["detail"] for d in s["drivers"]),
    }
    for s in data["at_risk_students"]
])
d2.download_button(
    "Download at-risk list (CSV)",
    at_risk_export.to_csv(index=False).encode("utf-8"),
    file_name="student_risk_at_risk_list.csv",
    mime="text/csv",
)
