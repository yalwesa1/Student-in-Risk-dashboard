"""
Student Dropout/Failure Risk Analysis
Generates synthetic survey responses (14 onboarding survey questions) and
scores each student on three risk dimensions - Academic, Financial, and
Personal/Other - to flag students at elevated risk of failing or dropping out.
"""

import json

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_STUDENTS = 500

# ---------------------------------------------------------------------------
# Survey option lists
# ---------------------------------------------------------------------------

CHALLENGES = [
    "NONE OF THESE", "BALANCING PRIORITIES", "HOME CONFLICTS", "TRANSPORTATION",
    "MENTAL HEALTH", "RELATIONSHIPS", "LACK OF MONEY", "CELL PHONE DISTRACTION",
    "LACK CONFIDENCE", "EASILY DISTRACTED",
]
CHALLENGES_WEIGHTS = [0.25, 0.14, 0.08, 0.08, 0.10, 0.08, 0.12, 0.09, 0.03, 0.03]

ACADEMIC_STRUGGLES = [
    "NONE OF THESE", "PUBLIC SPEAKING", "WRITING", "LECTURE-HEAVY CLASSES",
    "READING-HEAVY CLASSES", "MATH",
]
ACADEMIC_STRUGGLES_WEIGHTS = [0.30, 0.14, 0.16, 0.10, 0.10, 0.20]

AFFORD_ITEMS = ["Food", "Books", "Child Care", "Transportation", "None of these"]
AFFORD_WEIGHTS = [0.12, 0.20, 0.05, 0.13, 0.50]

STRENGTHS = [
    "Asks for help when needed", "Motivated", "Good Listener",
    "Manage Time Well", "Organized", "Good Test Taker",
]
STRUGGLES_11 = [
    "None of these", "Using technology", "Not Having a required laptop",
    "Paying for Textbooks", "Dietary Concerns", "Transportation", "Child Care",
    "Chronic Medical Conditions", "Anxiety", "Food Insecurity", "Depression",
]
STRUGGLES_11_WEIGHTS = [0.28, 0.05, 0.06, 0.14, 0.05, 0.08, 0.04, 0.06, 0.13, 0.06, 0.05]


def _sample_multiselect(rng, options, weights, none_label, min_n=0, max_n=4):
    """Pick a random subset of survey options; picking the 'none' option excludes all others."""
    if rng.random() < weights[options.index(none_label)]:
        return [none_label]
    pool = [o for o in options if o != none_label]
    pool_weights = np.array([w for o, w in zip(options, weights) if o != none_label])
    pool_weights = pool_weights / pool_weights.sum()
    n = rng.integers(1, max_n + 1)
    n = min(n, len(pool))
    choice = rng.choice(pool, size=n, replace=False, p=pool_weights)
    return list(choice)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_survey_data(n_students=N_STUDENTS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_students):
        first_gen = rng.choice(["Yes", "No"], p=[0.35, 0.65])
        challenges = _sample_multiselect(rng, CHALLENGES, CHALLENGES_WEIGHTS, "NONE OF THESE")
        academic_struggles = _sample_multiselect(
            rng, ACADEMIC_STRUGGLES, ACADEMIC_STRUGGLES_WEIGHTS, "NONE OF THESE"
        )
        late_assignments = rng.choice(["Often", "Sometimes", "Rarely"], p=[0.15, 0.35, 0.50])
        afford_struggles = _sample_multiselect(
            rng, AFFORD_ITEMS, AFFORD_WEIGHTS, "None of these", max_n=3
        )

        n_strengths = rng.integers(0, len(STRENGTHS) + 1)
        strengths = list(rng.choice(STRENGTHS, size=n_strengths, replace=False)) if n_strengths else []

        confidence = int(rng.choice([1, 2, 3, 4, 5], p=[0.08, 0.14, 0.28, 0.30, 0.20]))
        study_hours = rng.choice(["0", "1-2h", "3+h"], p=[0.20, 0.50, 0.30])
        commuter_info = rng.choice(
            ["I am not a commuter", "No", "Yes"], p=[0.45, 0.25, 0.30]
        )
        job_hours = rng.choice(
            ["0", "1-5 hours", "5-10 hours", "10-15 hours", "15+ hours"],
            p=[0.30, 0.25, 0.20, 0.15, 0.10],
        )
        struggles_11 = _sample_multiselect(
            rng, STRUGGLES_11, STRUGGLES_11_WEIGHTS, "None of these", max_n=4
        )
        esl = rng.choice(["Yes", "No"], p=[0.15, 0.85])
        accommodations = rng.choice(["Yes", "No"], p=[0.18, 0.82])
        campus_connections = rng.choice(["Yes", "No"], p=[0.55, 0.45])

        rows.append({
            "student_id": f"S{i + 1:04d}",
            "first_gen": first_gen,
            "challenges": "; ".join(challenges),
            "academic_struggles": "; ".join(academic_struggles),
            "late_assignments": late_assignments,
            "afford_struggles": "; ".join(afford_struggles),
            "strengths": "; ".join(strengths) if strengths else "None",
            "confidence_rating": confidence,
            "study_hours": study_hours,
            "commuter_info_interest": commuter_info,
            "job_hours_per_week": job_hours,
            "struggles_11": "; ".join(struggles_11),
            "esl": esl,
            "accommodations_hs": accommodations,
            "campus_connections_confident": campus_connections,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

CHALLENGE_POINTS = {
    "BALANCING PRIORITIES": {"academic": 2},
    "HOME CONFLICTS": {"personal": 2},
    "TRANSPORTATION": {"financial": 2},
    "MENTAL HEALTH": {"personal": 3},
    "RELATIONSHIPS": {"personal": 1},
    "LACK OF MONEY": {"financial": 3},
    "CELL PHONE DISTRACTION": {"academic": 1},
    "LACK CONFIDENCE": {"academic": 2, "personal": 1},
    "EASILY DISTRACTED": {"academic": 1},
}

AFFORD_POINTS = {
    "Food": {"financial": 3, "personal": 1},
    "Books": {"financial": 3, "academic": 1},
    "Child Care": {"financial": 2, "personal": 1},
    "Transportation": {"financial": 2},
}

STRUGGLE_11_POINTS = {
    "Using technology": {"academic": 1},
    "Not Having a required laptop": {"financial": 3, "academic": 1},
    "Paying for Textbooks": {"financial": 3},
    "Dietary Concerns": {"personal": 1},
    "Transportation": {"financial": 2},
    "Child Care": {"financial": 2, "personal": 1},
    "Chronic Medical Conditions": {"personal": 3},
    "Anxiety": {"personal": 3},
    "Food Insecurity": {"financial": 3, "personal": 1},
    "Depression": {"personal": 4},
}

JOB_HOURS_POINTS = {
    "0": {},
    "1-5 hours": {"academic": 0},
    "5-10 hours": {"academic": 1},
    "10-15 hours": {"academic": 2, "financial": 1},
    "15+ hours": {"academic": 3, "financial": 2},
}

STUDY_HOURS_POINTS = {"0": 5, "1-2h": 2, "3+h": 0}
LATE_ASSIGNMENT_POINTS = {"Often": 6, "Sometimes": 3, "Rarely": 0}

# Rough ceilings used to normalize raw points to a 0-100 scale per category.
MAX_POINTS = {"academic": 32, "financial": 24, "personal": 24}


CATEGORY_LABEL = {"academic": "Academic", "financial": "Financial", "personal": "Personal"}


def _split(field):
    if pd.isna(field) or field in ("None", ""):
        return []
    return [v.strip() for v in field.split(";") if v.strip() and v.strip().upper() != "NONE OF THESE"
            and v.strip().lower() != "none of these"]


def _title(text):
    return text[:1].upper() + text[1:].lower() if text.isupper() else text


def score_row_with_drivers(row):
    """Return (points_by_category, drivers) where drivers lists every answer that
    added risk points, so the dashboard can explain *why* a student was flagged."""
    pts = {"academic": 0.0, "financial": 0.0, "personal": 0.0}
    drivers = []

    def add(label, **cats):
        contributed = {c: v for c, v in cats.items() if v}
        if not contributed:
            return
        for c, v in contributed.items():
            pts[c] += v
        primary_cat = max(contributed, key=contributed.get)
        tag = ", ".join(f"+{v:g} {CATEGORY_LABEL[c]}" for c, v in contributed.items())
        drivers.append({
            "label": label,
            "detail": f"{label} ({tag})",
            "points": sum(contributed.values()),
            "primary_category": primary_cat,
        })

    if row["first_gen"] == "Yes":
        add("First-generation college student", academic=2, personal=1)

    for item in _split(row["challenges"]):
        add(_title(item), **CHALLENGE_POINTS.get(item, {}))

    struggles = _split(row["academic_struggles"])
    if struggles:
        add("Struggles in: " + ", ".join(_title(s) for s in struggles), academic=2 * len(struggles))

    add(f"Turns in assignments late ({row['late_assignments'].lower()})",
        academic=LATE_ASSIGNMENT_POINTS[row["late_assignments"]])

    for item in _split(row["afford_struggles"]):
        add(f"Struggles to afford {item.lower()}", **AFFORD_POINTS.get(item, {}))

    n_strengths = 0 if row["strengths"] == "None" else len(row["strengths"].split("; "))

    if row["confidence_rating"] < 5:
        add(f"Low self-reported confidence ({row['confidence_rating']}/5)",
            academic=(5 - row["confidence_rating"]) * 2)

    add(f"Low outside-of-class study time ({row['study_hours']}/day)",
        academic=STUDY_HOURS_POINTS[row["study_hours"]])

    add(f"Plans to work {row['job_hours_per_week']}/week", **JOB_HOURS_POINTS[row["job_hours_per_week"]])

    for item in _split(row["struggles_11"]):
        add(item, **STRUGGLE_11_POINTS.get(item, {}))

    if row["esl"] == "Yes":
        add("English as a second language", personal=1)
    if row["accommodations_hs"] == "Yes":
        add("Had IEP/504 accommodations in high school", personal=1)
    if row["campus_connections_confident"] == "No":
        add("Not confident making campus connections", personal=3)

    # Strengths are protective: they reduce academic risk but are not "drivers".
    pts["academic"] = max(0.0, pts["academic"] - n_strengths)

    drivers.sort(key=lambda d: -d["points"])
    return pts, drivers


def level(score):
    if score >= 50:
        return "High"
    if score >= 28:
        return "Medium"
    return "Low"


def categorize_risk(df):
    academic, financial, personal = [], [], []
    drivers_by_row = []

    for _, row in df.iterrows():
        pts, drivers = score_row_with_drivers(row)
        academic.append(round(min(100, pts["academic"] / MAX_POINTS["academic"] * 100), 1))
        financial.append(round(min(100, pts["financial"] / MAX_POINTS["financial"] * 100), 1))
        personal.append(round(min(100, pts["personal"] / MAX_POINTS["personal"] * 100), 1))
        drivers_by_row.append(drivers)

    df = df.copy()
    df["academic_risk_score"] = academic
    df["financial_risk_score"] = financial
    df["personal_risk_score"] = personal

    df["academic_risk_level"] = df["academic_risk_score"].apply(level)
    df["financial_risk_level"] = df["financial_risk_score"].apply(level)
    df["personal_risk_level"] = df["personal_risk_score"].apply(level)

    df["overall_risk_score"] = round(
        0.40 * df["academic_risk_score"]
        + 0.35 * df["financial_risk_score"]
        + 0.25 * df["personal_risk_score"],
        1,
    )
    df["overall_risk_level"] = df["overall_risk_score"].apply(level)

    cat_scores = df[["academic_risk_score", "financial_risk_score", "personal_risk_score"]]
    df["primary_risk_category"] = cat_scores.idxmax(axis=1).str.replace("_risk_score", "", regex=False).str.capitalize()

    df["risk_drivers"] = ["; ".join(d["detail"] for d in ds[:4]) for ds in drivers_by_row]
    df.attrs["drivers_by_row"] = drivers_by_row  # full detail, used by the dashboard export
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(df):
    print("=" * 60)
    print(f"STUDENT RISK ANALYSIS SUMMARY  (n = {len(df)})")
    print("=" * 60)

    print("\nOverall risk level distribution:")
    print(df["overall_risk_level"].value_counts().reindex(["High", "Medium", "Low"]).to_string())

    print("\nRisk level by category:")
    for cat in ["academic", "financial", "personal"]:
        counts = df[f"{cat}_risk_level"].value_counts().reindex(["High", "Medium", "Low"])
        print(f"  {cat.capitalize():10s}: " + ", ".join(f"{lvl}={n}" for lvl, n in counts.items()))

    print("\nPrimary risk driver among High-risk students:")
    high_risk = df[df["overall_risk_level"] == "High"]
    if len(high_risk):
        print(high_risk["primary_risk_category"].value_counts().to_string())
    else:
        print("  (none)")

    print(f"\nHigh-risk students needing outreach: {len(high_risk)} / {len(df)} "
          f"({len(high_risk) / len(df) * 100:.1f}%)")


def print_at_risk_list(df, max_rows=25):
    at_risk = df[df["overall_risk_level"].isin(["High", "Medium"])].sort_values(
        "overall_risk_score", ascending=False
    )
    print(f"\nAt-risk students (High + Medium): {len(at_risk)}")
    print("-" * 100)
    cols = ["student_id", "overall_risk_score", "overall_risk_level", "primary_risk_category", "risk_drivers"]
    for _, r in at_risk.head(max_rows).iterrows():
        drivers = r["risk_drivers"] if len(r["risk_drivers"]) <= 80 else r["risk_drivers"][:77] + "..."
        print(f"  {r['student_id']}  score={r['overall_risk_score']:5.1f}  [{r['overall_risk_level']:6s}]  "
              f"primary={r['primary_risk_category']:10s}  drivers: {drivers}")
    if len(at_risk) > max_rows:
        print(f"  ... and {len(at_risk) - max_rows} more (see CSV / dashboard for the full list)")


def save_outputs(df, out_dir="."):
    scored_path = f"{out_dir}/student_risk_scored.csv"
    df.to_csv(scored_path, index=False)
    print(f"\nSaved scored dataset to: {scored_path}")

    at_risk_path = f"{out_dir}/student_risk_at_risk_list.csv"
    at_risk_cols = [
        "student_id", "overall_risk_score", "overall_risk_level", "primary_risk_category",
        "academic_risk_score", "financial_risk_score", "personal_risk_score", "risk_drivers",
    ]
    at_risk = df[df["overall_risk_level"].isin(["High", "Medium"])].sort_values(
        "overall_risk_score", ascending=False
    )[at_risk_cols]
    at_risk.to_csv(at_risk_path, index=False)
    print(f"Saved at-risk student list to: {at_risk_path}")
    return scored_path, at_risk_path


# ---------------------------------------------------------------------------
# Dashboard export (charts + sortable at-risk student list, self-contained HTML)
# ---------------------------------------------------------------------------

def build_dashboard_data(df):
    drivers_by_row = df.attrs.get("drivers_by_row", [[] for _ in range(len(df))])
    levels = ["Low", "Medium", "High"]

    overall_counts = {lvl: int((df["overall_risk_level"] == lvl).sum()) for lvl in levels}
    category_counts = {
        cat: {lvl: int((df[f"{cat}_risk_level"] == lvl).sum()) for lvl in levels}
        for cat in ["academic", "financial", "personal"]
    }

    bin_edges = list(range(0, 101, 5))
    hist_counts, _ = np.histogram(df["overall_risk_score"], bins=bin_edges)
    histogram = [
        {"start": bin_edges[i], "end": bin_edges[i + 1], "count": int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]

    at_risk_students = []
    driver_counter = {}
    for pos, (_, row) in enumerate(df.iterrows()):
        if row["overall_risk_level"] not in ("High", "Medium"):
            continue
        top_drivers = drivers_by_row[pos][:6]
        for d in top_drivers:
            key = (d["label"], d["primary_category"])
            driver_counter[key] = driver_counter.get(key, 0) + 1
        at_risk_students.append({
            "id": row["student_id"],
            "overall_score": row["overall_risk_score"],
            "overall_level": row["overall_risk_level"],
            "academic_score": row["academic_risk_score"],
            "financial_score": row["financial_risk_score"],
            "personal_score": row["personal_risk_score"],
            "primary_category": row["primary_risk_category"],
            "drivers": [
                {"label": d["label"], "detail": d["detail"], "points": d["points"], "category": d["primary_category"]}
                for d in top_drivers
            ],
        })
    at_risk_students.sort(key=lambda s: -s["overall_score"])

    top_driver_freq = sorted(
        [{"label": k[0], "category": CATEGORY_LABEL[k[1]], "count": v} for k, v in driver_counter.items()],
        key=lambda x: -x["count"],
    )[:10]

    return {
        "total_students": len(df),
        "thresholds": {"medium": 28, "high": 50},
        "overall_counts": overall_counts,
        "category_counts": category_counts,
        "histogram": histogram,
        "at_risk_students": at_risk_students,
        "top_driver_freq": top_driver_freq,
    }


DASHBOARD_TEMPLATE = r"""
<div class="viz-root">
<style>
.viz-root {
  color-scheme: light;
  --surface-1:      #fcfcfb;
  --page:           #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --gridline:       #e1e0d9;
  --baseline:       #c3c2b7;
  --border:         rgba(11,11,11,0.10);
  --cat-academic:   #2a78d6;
  --cat-financial:  #1baf7a;
  --cat-personal:   #eda100;
  --status-good:    #0ca30c;
  --status-warning: #fab219;
  --status-critical:#d03b3b;
  --seq-100: #cde2fb; --seq-300: #6da7ec; --seq-500: #256abf; --seq-700: #0d366b;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--page);
  color: var(--text-primary);
  padding: 24px;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page:           #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --gridline:       #2c2c2a;
    --baseline:       #383835;
    --border:         rgba(255,255,255,0.10);
    --cat-academic:   #3987e5;
    --cat-financial:  #199e70;
    --cat-personal:   #c98500;
    --status-good:    #0ca30c;
    --status-warning: #fab219;
    --status-critical:#d03b3b;
    --seq-100: #b7d3f6; --seq-300: #5598e7; --seq-500: #1c5cab; --seq-700: #0d366b;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1:      #1a1a19;
  --page:           #0d0d0d;
  --text-primary:   #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted:     #898781;
  --gridline:       #2c2c2a;
  --baseline:       #383835;
  --border:         rgba(255,255,255,0.10);
  --cat-academic:   #3987e5;
  --cat-financial:  #199e70;
  --cat-personal:   #c98500;
  --seq-100: #b7d3f6; --seq-300: #5598e7; --seq-500: #1c5cab; --seq-700: #0d366b;
}

.viz-root * { box-sizing: border-box; }
.viz-page { max-width: 1180px; margin: 0 auto; }
.viz-h1 { font-size: 22px; font-weight: 700; margin: 0 0 4px; }
.viz-sub { color: var(--text-secondary); font-size: 14px; margin: 0 0 24px; }

.viz-tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 28px; }
.viz-tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
.viz-tile-label { font-size: 12px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: .04em; margin-bottom: 6px; }
.viz-tile-value { font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums; }
.viz-tile-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
.viz-tile.high { border-left: 4px solid var(--status-critical); }
.viz-tile.medium { border-left: 4px solid var(--status-warning); }
.viz-tile.low { border-left: 4px solid var(--status-good); }

.viz-card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin-bottom: 20px; }
.viz-card-title { font-size: 15px; font-weight: 600; margin: 0 0 2px; }
.viz-card-desc { font-size: 12.5px; color: var(--text-secondary); margin: 0 0 16px; }

.viz-legend { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 14px; font-size: 12.5px; color: var(--text-secondary); }
.viz-legend-item { display: flex; align-items: center; gap: 6px; }
.viz-swatch { width: 10px; height: 10px; border-radius: 2px; flex: none; }

.viz-hbar-row { display: grid; grid-template-columns: 90px 1fr 110px; align-items: center; gap: 10px; margin-bottom: 10px; font-size: 13px; }
.viz-hbar-track { background: var(--gridline); border-radius: 4px; height: 18px; overflow: hidden; position: relative; }
.viz-hbar-fill { height: 100%; border-radius: 4px; transition: width .2s; }
.viz-hbar-value { font-variant-numeric: tabular-nums; color: var(--text-secondary); font-size: 12.5px; }

.viz-stack-row { display: grid; grid-template-columns: 90px 1fr; align-items: center; gap: 10px; margin-bottom: 12px; font-size: 13px; }
.viz-stack-track { display: flex; height: 22px; border-radius: 5px; overflow: hidden; background: var(--gridline); }
.viz-stack-seg { height: 100%; display: flex; align-items: center; justify-content: center; font-size: 11px; color: #fff; font-weight: 600; }
.viz-stack-seg[data-n="0"] { display: none; }

.viz-hist { display: flex; align-items: flex-end; gap: 2px; height: 160px; padding-top: 20px; position: relative; }
.viz-hist-bar { flex: 1; background: var(--seq-500); border-radius: 3px 3px 0 0; min-height: 1px; position: relative; }
.viz-hist-bar:hover { background: var(--seq-700); }
.viz-hist-axis { display: flex; justify-content: space-between; font-size: 10.5px; color: var(--text-muted); margin-top: 6px; }
.viz-threshold { position: absolute; top: 20px; bottom: 0; width: 0; border-left: 2px dashed var(--status-warning); }
.viz-threshold.high { border-left-color: var(--status-critical); }
.viz-threshold-label { position: absolute; top: 0; font-size: 10px; color: var(--text-secondary); white-space: nowrap; transform: translateX(-50%); }

.viz-driver-row { display: grid; grid-template-columns: 1fr 1fr 40px; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 12.5px; }
.viz-driver-label { color: var(--text-primary); }
.viz-driver-track { background: var(--gridline); border-radius: 4px; height: 14px; }
.viz-driver-fill { height: 100%; border-radius: 4px; }
.viz-driver-count { text-align: right; color: var(--text-secondary); font-variant-numeric: tabular-nums; }

.viz-toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 14px; flex-wrap: wrap; }
.viz-chip { border: 1px solid var(--border); background: transparent; color: var(--text-secondary); border-radius: 999px; padding: 6px 14px; font-size: 12.5px; cursor: pointer; font-family: inherit; }
.viz-chip.active { background: var(--text-primary); color: var(--surface-1); border-color: var(--text-primary); }
.viz-search { border: 1px solid var(--border); background: var(--page); color: var(--text-primary); border-radius: 8px; padding: 7px 12px; font-size: 12.5px; font-family: inherit; margin-left: auto; min-width: 180px; }

.viz-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.viz-table th { text-align: left; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: .03em; font-size: 10.5px; padding: 8px 10px; border-bottom: 1px solid var(--gridline); cursor: pointer; user-select: none; }
.viz-table td { padding: 9px 10px; border-bottom: 1px solid var(--gridline); vertical-align: top; }
.viz-table tr.viz-row:hover { background: color-mix(in srgb, var(--text-primary) 4%, transparent); cursor: pointer; }
.viz-badge { display: inline-flex; align-items: center; gap: 5px; padding: 3px 9px; border-radius: 999px; font-size: 11px; font-weight: 600; }
.viz-badge .dot { width: 7px; height: 7px; border-radius: 50%; }
.viz-badge.high { background: color-mix(in srgb, var(--status-critical) 16%, transparent); color: var(--status-critical); }
.viz-badge.high .dot { background: var(--status-critical); }
.viz-badge.medium { background: color-mix(in srgb, var(--status-warning) 20%, transparent); color: color-mix(in srgb, var(--status-warning) 65%, var(--text-primary)); }
.viz-badge.medium .dot { background: var(--status-warning); }
.viz-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.viz-driver-chip { background: var(--page); border: 1px solid var(--border); border-radius: 6px; padding: 2px 7px; font-size: 11px; color: var(--text-secondary); }
.viz-mini-scores { display: flex; gap: 8px; font-size: 11px; color: var(--text-secondary); }
.viz-mini-scores span b { color: var(--text-primary); font-variant-numeric: tabular-nums; }
.viz-detail { display: none; background: var(--page); }
.viz-detail.open { display: table-row; }
.viz-detail-inner { padding: 14px 16px; }
.viz-detail-title { font-size: 11px; text-transform: uppercase; letter-spacing: .03em; color: var(--text-muted); margin-bottom: 8px; }
.viz-count-note { font-size: 12px; color: var(--text-muted); margin-top: 10px; }
</style>

<div class="viz-page">
  <h1 class="viz-h1">Student Risk Dashboard</h1>
  <p class="viz-sub">Synthetic onboarding-survey data &middot; <span id="meta-n"></span> students scored across Academic, Financial, and Personal risk dimensions</p>

  <div class="viz-tiles" id="tiles"></div>

  <div class="viz-card">
    <p class="viz-card-title">Overall risk distribution</p>
    <p class="viz-card-desc">All students grouped by combined risk level (40% academic, 35% financial, 25% personal).</p>
    <div id="overall-chart"></div>
  </div>

  <div class="viz-card">
    <p class="viz-card-title">Risk level by category</p>
    <p class="viz-card-desc">Share of students at each risk level, broken out per dimension.</p>
    <div class="viz-legend">
      <span class="viz-legend-item"><span class="viz-swatch" style="background:var(--status-good)"></span>Low</span>
      <span class="viz-legend-item"><span class="viz-swatch" style="background:var(--status-warning)"></span>Medium</span>
      <span class="viz-legend-item"><span class="viz-swatch" style="background:var(--status-critical)"></span>High</span>
    </div>
    <div id="category-chart"></div>
  </div>

  <div class="viz-card">
    <p class="viz-card-title">Overall risk score distribution</p>
    <p class="viz-card-desc">Dashed lines mark the Medium (&ge;28) and High (&ge;50) thresholds.</p>
    <div id="hist-chart"></div>
  </div>

  <div class="viz-card">
    <p class="viz-card-title">Most common risk factors among at-risk students</p>
    <p class="viz-card-desc">How often each survey answer appears among Medium/High-risk students' top drivers.</p>
    <div class="viz-legend">
      <span class="viz-legend-item"><span class="viz-swatch" style="background:var(--cat-academic)"></span>Academic</span>
      <span class="viz-legend-item"><span class="viz-swatch" style="background:var(--cat-financial)"></span>Financial</span>
      <span class="viz-legend-item"><span class="viz-swatch" style="background:var(--cat-personal)"></span>Personal</span>
    </div>
    <div id="driver-chart"></div>
  </div>

  <div class="viz-card">
    <p class="viz-card-title">At-risk student list</p>
    <p class="viz-card-desc">Every Medium or High risk student with the specific survey answers driving that score. Click a row to expand.</p>
    <div class="viz-toolbar">
      <button class="viz-chip active" data-filter="all">All</button>
      <button class="viz-chip" data-filter="High">High only</button>
      <button class="viz-chip" data-filter="Medium">Medium only</button>
      <input class="viz-search" id="search" placeholder="Search student ID..." />
    </div>
    <table class="viz-table">
      <thead>
        <tr>
          <th data-sort="id">Student</th>
          <th data-sort="overall_score">Overall</th>
          <th data-sort="academic_score">Academic</th>
          <th data-sort="financial_score">Financial</th>
          <th data-sort="personal_score">Personal</th>
          <th data-sort="primary_category">Primary</th>
          <th>Top risk factors</th>
        </tr>
      </thead>
      <tbody id="table-body"></tbody>
    </table>
    <p class="viz-count-note" id="count-note"></p>
  </div>
</div>
</div>

<script>
(function () {
  const DATA = __STUDENT_DATA_JSON__;
  const root = document.currentScript.previousElementSibling;

  const statusColor = { Low: "var(--status-good)", Medium: "var(--status-warning)", High: "var(--status-critical)" };
  const catColor = { Academic: "var(--cat-academic)", Financial: "var(--cat-financial)", Personal: "var(--cat-personal)" };
  const catKeyColor = { academic: "var(--cat-academic)", financial: "var(--cat-financial)", personal: "var(--cat-personal)" };

  document.getElementById("meta-n").textContent = DATA.total_students;

  // --- Stat tiles ---------------------------------------------------------
  const tiles = document.getElementById("tiles");
  const total = DATA.total_students;
  const oc = DATA.overall_counts;
  const tileDefs = [
    { label: "Total students", value: total, cls: "" },
    { label: "High risk", value: oc.High, cls: "high", sub: (oc.High / total * 100).toFixed(1) + "% of students" },
    { label: "Medium risk", value: oc.Medium, cls: "medium", sub: (oc.Medium / total * 100).toFixed(1) + "% of students" },
    { label: "Low risk", value: oc.Low, cls: "low", sub: (oc.Low / total * 100).toFixed(1) + "% of students" },
  ];
  tiles.innerHTML = tileDefs.map(t => `
    <div class="viz-tile ${t.cls}">
      <div class="viz-tile-label">${t.label}</div>
      <div class="viz-tile-value">${t.value}</div>
      ${t.sub ? `<div class="viz-tile-sub">${t.sub}</div>` : ""}
    </div>`).join("");

  // --- Overall risk distribution (horizontal bars) ------------------------
  const overallChart = document.getElementById("overall-chart");
  const maxOverall = Math.max(oc.Low, oc.Medium, oc.High, 1);
  overallChart.innerHTML = ["Low", "Medium", "High"].map(lvl => `
    <div class="viz-hbar-row">
      <span>${lvl}</span>
      <div class="viz-hbar-track"><div class="viz-hbar-fill" style="width:${oc[lvl] / maxOverall * 100}%; background:${statusColor[lvl]}"></div></div>
      <span class="viz-hbar-value">${oc[lvl]} (${(oc[lvl] / total * 100).toFixed(1)}%)</span>
    </div>`).join("");

  // --- Risk level by category (100% stacked bars) --------------------------
  const categoryChart = document.getElementById("category-chart");
  const catLabels = { academic: "Academic", financial: "Financial", personal: "Personal" };
  categoryChart.innerHTML = Object.keys(DATA.category_counts).map(cat => {
    const c = DATA.category_counts[cat];
    const catTotal = c.Low + c.Medium + c.High;
    const seg = lvl => `<div class="viz-stack-seg" data-n="${c[lvl]}" style="width:${c[lvl] / catTotal * 100}%; background:${statusColor[lvl]}" title="${lvl}: ${c[lvl]} students">${c[lvl] > catTotal * 0.08 ? c[lvl] : ""}</div>`;
    return `
    <div class="viz-stack-row">
      <span>${catLabels[cat]}</span>
      <div class="viz-stack-track">${seg("Low")}${seg("Medium")}${seg("High")}</div>
    </div>`;
  }).join("");

  // --- Histogram ------------------------------------------------------------
  const histChart = document.getElementById("hist-chart");
  const maxHist = Math.max(...DATA.histogram.map(b => b.count), 1);
  const bars = DATA.histogram.map(b => `<div class="viz-hist-bar" style="height:${b.count / maxHist * 100}%" title="${b.start}-${b.end}: ${b.count} students"></div>`).join("");
  const medPct = DATA.thresholds.medium;
  const highPct = DATA.thresholds.high;
  histChart.innerHTML = `
    <div class="viz-hist">
      ${bars}
      <div class="viz-threshold" style="left:${medPct}%"><span class="viz-threshold-label">Medium &ge;${medPct}</span></div>
      <div class="viz-threshold high" style="left:${highPct}%"><span class="viz-threshold-label">High &ge;${highPct}</span></div>
    </div>
    <div class="viz-hist-axis"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>`;

  // --- Top risk factors -------------------------------------------------
  const driverChart = document.getElementById("driver-chart");
  const maxDriver = Math.max(...DATA.top_driver_freq.map(d => d.count), 1);
  driverChart.innerHTML = DATA.top_driver_freq.map(d => `
    <div class="viz-driver-row">
      <span class="viz-driver-label">${d.label}</span>
      <div class="viz-driver-track"><div class="viz-driver-fill" style="width:${d.count / maxDriver * 100}%; background:${catColor[d.category]}"></div></div>
      <span class="viz-driver-count">${d.count}</span>
    </div>`).join("");

  // --- At-risk student table ------------------------------------------------
  let sortKey = "overall_score", sortDir = -1, filter = "all", query = "";
  const tbody = document.getElementById("table-body");
  const countNote = document.getElementById("count-note");

  function scoreBadge(lvl) {
    const cls = lvl === "High" ? "high" : "medium";
    return `<span class="viz-badge ${cls}"><span class="dot"></span>${lvl}</span>`;
  }

  function render() {
    let rows = DATA.at_risk_students.filter(s => filter === "all" || s.overall_level === filter);
    if (query) rows = rows.filter(s => s.id.toLowerCase().includes(query.toLowerCase()));
    rows = rows.slice().sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      if (typeof av === "string") return av.localeCompare(bv) * sortDir;
      return (av - bv) * sortDir;
    });

    tbody.innerHTML = rows.map((s, i) => {
      const chips = s.drivers.slice(0, 3).map(d => `<span class="viz-driver-chip">${d.detail}</span>`).join("");
      const fullDrivers = s.drivers.map(d => `<span class="viz-driver-chip" style="border-color:${catKeyColor[d.category.toLowerCase()] || catKeyColor[d.category]}">${d.detail}</span>`).join(" ");
      return `
      <tr class="viz-row" data-idx="${i}">
        <td><b>${s.id}</b></td>
        <td>${scoreBadge(s.overall_level)}<div class="viz-tile-sub" style="margin-top:3px">${s.overall_score.toFixed(1)}</div></td>
        <td>${s.academic_score.toFixed(1)}</td>
        <td>${s.financial_score.toFixed(1)}</td>
        <td>${s.personal_score.toFixed(1)}</td>
        <td>${s.primary_category}</td>
        <td><div class="viz-chips">${chips}</div></td>
      </tr>
      <tr class="viz-detail" data-detail="${i}">
        <td colspan="7">
          <div class="viz-detail-inner">
            <div class="viz-detail-title">All contributing survey answers</div>
            <div class="viz-chips">${fullDrivers}</div>
          </div>
        </td>
      </tr>`;
    }).join("");

    countNote.textContent = `Showing ${rows.length} of ${DATA.at_risk_students.length} at-risk students`;

    tbody.querySelectorAll(".viz-row").forEach(tr => {
      tr.addEventListener("click", () => {
        const detail = tbody.querySelector(`.viz-detail[data-detail="${tr.dataset.idx}"]`);
        detail.classList.toggle("open");
      });
    });
  }

  document.querySelectorAll(".viz-chip[data-filter]").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".viz-chip[data-filter]").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      filter = btn.dataset.filter;
      render();
    });
  });
  document.getElementById("search").addEventListener("input", e => { query = e.target.value; render(); });
  document.querySelectorAll(".viz-table th[data-sort]").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      sortDir = (sortKey === key) ? -sortDir : -1;
      sortKey = key;
      render();
    });
  });

  render();
})();
</script>
"""


def render_dashboard_html(dashboard_data, out_path="student_risk_dashboard.html"):
    html = DASHBOARD_TEMPLATE.replace("__STUDENT_DATA_JSON__", json.dumps(dashboard_data))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved interactive dashboard to: {out_path}")
    return out_path


def main():
    raw_df = generate_survey_data()
    scored_df = categorize_risk(raw_df)
    print_summary(scored_df)
    print_at_risk_list(scored_df)
    save_outputs(scored_df)
    dashboard_data = build_dashboard_data(scored_df)
    render_dashboard_html(dashboard_data)
    return scored_df


if __name__ == "__main__":
    main()
