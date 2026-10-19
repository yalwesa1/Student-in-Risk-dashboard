# Student Risk Dashboard

Identifies students at elevated risk of failing or dropping out — for academic,
financial, or personal reasons — based on a 14-question onboarding survey.
Includes a synthetic-data generator, a rule-based risk-scoring engine, and an
interactive Streamlit dashboard.

_Designed by SUNY Academic Momentum Fellows Program_

## What it does

1. **Generates synthetic survey data** for 500 students matching the 14
   onboarding survey questions (first-gen status, challenges, academic
   struggles, financial affordability, study habits, confidence, etc.).
2. **Scores each student** on three risk dimensions using a weighted,
   transparent point rubric:
   - **Academic** (40% of overall score)
   - **Financial** (35%)
   - **Personal** (25%)

   Each survey answer that adds risk points is tracked as a named "driver,"
   so every score traces back to specific answers.
3. **Classifies** each student as Low / Medium / High risk per dimension and
   overall (thresholds: High ≥ 50, Medium ≥ 28).
4. **Visualizes results** in an interactive dashboard: risk distributions,
   category breakdowns, a score histogram, the most common risk factors
   among at-risk students, and a searchable/sortable list of at-risk
   students with full driver detail per student.

## Files

| File | Purpose |
|---|---|
| `student_risk_analysis.py` | Data generation, scoring engine, CSV export, static HTML dashboard export |
| `streamlit_app.py` | Interactive Streamlit version of the dashboard |
| `student_risk_scored.csv` | Full scored dataset (generated) |
| `student_risk_at_risk_list.csv` | Medium/High risk students only (generated) |
| `student_risk_dashboard.html` | Self-contained static dashboard (generated) |

## Running it

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Or to just regenerate the data and static HTML dashboard:

```bash
python student_risk_analysis.py
```

## Note on the data

All survey responses are **synthetically generated** (`numpy` random sampling
with realistic weighted distributions) for demonstration purposes — this is
not real student data. The risk-scoring weights are a starting rubric based
on domain judgment, not a model validated against real outcome data; treat
classifications as a triage aid, not a certified prediction.
