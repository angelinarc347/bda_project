"""
Heart Disease Risk Predictor — Production App
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense · Heart Risk Analyzer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --red:       #e84545;
    --red-soft:  rgba(232,69,69,.12);
    --amber:     #f0a500;
    --amber-soft:rgba(240,165,0,.12);
    --green:     #3fb950;
    --green-soft:rgba(63,185,80,.12);
    --blue:      #388bfd;
    --blue-soft: rgba(56,139,253,.10);
  }

  html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--text); }
  .main { background:var(--bg); }
  .block-container { padding: 2rem 2.5rem 4rem; max-width:1200px; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
  }
  section[data-testid="stSidebar"] .stMarkdown h1,
  section[data-testid="stSidebar"] .stMarkdown h2,
  section[data-testid="stSidebar"] .stMarkdown h3 { color:var(--text); }

  /* Headers */
  h1 { font-family:'DM Serif Display',serif; font-size:2.4rem !important; letter-spacing:-.5px; }
  h2 { font-family:'DM Serif Display',serif; font-size:1.6rem !important; color:var(--text); }
  h3 { font-family:'DM Sans',sans-serif; font-size:1.05rem !important; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }

  /* Cards */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .card-red    { border-left: 3px solid var(--red);   background: var(--red-soft);   }
  .card-amber  { border-left: 3px solid var(--amber); background: var(--amber-soft); }
  .card-green  { border-left: 3px solid var(--green); background: var(--green-soft); }
  .card-blue   { border-left: 3px solid var(--blue);  background: var(--blue-soft);  }

  /* Risk badge */
  .risk-badge {
    display:inline-block; padding:.35rem 1.1rem; border-radius:999px;
    font-weight:600; font-size:.9rem; letter-spacing:.03em;
  }
  .risk-high   { background:var(--red-soft);   color:var(--red);   border:1px solid var(--red);   }
  .risk-mod    { background:var(--amber-soft); color:var(--amber); border:1px solid var(--amber); }
  .risk-low    { background:var(--green-soft); color:var(--green); border:1px solid var(--green); }

  /* Metric tiles */
  .metric-tile {
    background:var(--surface); border:1px solid var(--border); border-radius:10px;
    padding:1rem 1.2rem; text-align:center;
  }
  .metric-tile .val { font-size:1.8rem; font-weight:700; font-family:'DM Serif Display',serif; }
  .metric-tile .lbl { font-size:.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; }

  /* Factor rows */
  .factor-row {
    display:flex; align-items:center; gap:.75rem;
    padding:.6rem .8rem; border-radius:8px; margin-bottom:.4rem;
    background:var(--surface); border:1px solid var(--border);
  }
  .factor-icon { font-size:1.2rem; }
  .factor-name { flex:1; font-size:.9rem; font-weight:500; }
  .factor-bar  { height:6px; border-radius:999px; background:var(--border); flex:2; }
  .factor-fill { height:6px; border-radius:999px; }
  .factor-val  { font-size:.8rem; color:var(--muted); width:3rem; text-align:right; }

  /* Buttons */
  .stButton > button {
    background: var(--red) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: .75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: opacity .2s;
  }
  .stButton > button:hover { opacity:.85 !important; }

  /* Divider */
  hr { border-color: var(--border) !important; }

  /* Slider / select labels */
  label { color: var(--text) !important; font-size: .9rem !important; }
  .stSlider [data-testid="stTickBar"] { display:none; }

  /* Tab bar */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 1px solid var(--border);
    background: transparent !important;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 0 !important;
    font-size: .9rem; font-weight: 500;
    padding: .6rem 1.4rem;
    border-bottom: 2px solid transparent;
  }
  .stTabs [aria-selected="true"] {
    color: var(--text) !important;
    border-bottom: 2px solid var(--red) !important;
  }

  /* History table */
  .history-row {
    display:flex; align-items:center; justify-content:space-between;
    padding:.7rem 1rem; border-radius:8px; margin-bottom:.35rem;
    background:var(--surface); border:1px solid var(--border);
    font-size:.88rem;
  }

  /* Caption override */
  .stCaption { color: var(--muted) !important; }

  /* Remove default Streamlit metric styling */
  [data-testid="stMetricValue"] { font-family:'DM Serif Display',serif; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model    = pickle.load(open("model.pkl",    "rb"))
        scaler   = pickle.load(open("scaler.pkl",   "rb"))
        metadata = pickle.load(open("metadata.pkl", "rb"))
        return model, scaler, metadata
    except FileNotFoundError as e:
        st.error(f"❌ Missing artifact: {e}. Run `python model.py` first.")
        st.stop()

model, scaler, meta = load_artifacts()
FEATURES    = meta["feature_names"]
LABELS      = meta["feature_labels"]
IMPORTANCES = meta["feature_importance"]
METRICS     = meta["metrics"]

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar — input form ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 CardioSense")
    st.markdown("<p style='color:#8b949e;font-size:.85rem;margin-top:-.5rem'>Heart Risk Analyzer v2.0</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 👤 Basic Info")
    age = st.slider("Age", 20, 100, 52)
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
    sex_val = 1 if sex == "Male" else 0

    st.markdown("### 💓 Symptoms")
    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-Anginal Pain": 2,
        "Asymptomatic": 3,
    }
    cp = st.selectbox("Chest Pain Type", list(cp_map.keys()))
    cp_val = cp_map[cp]

    exang = st.radio("Exercise-Induced Angina?", ["No", "Yes"], horizontal=True)
    exang_val = 1 if exang == "Yes" else 0

    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1,
                        help="ST depression induced by exercise relative to rest.")

    st.markdown("### 🩺 Clinical Measurements")
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
    chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    thalach  = st.slider("Max Heart Rate Achieved", 60, 220, 150)

    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"], horizontal=True)
    fbs_val = 1 if fbs == "Yes" else 0

    restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "LV Hypertrophy": 2}
    restecg     = st.selectbox("Resting ECG", list(restecg_map.keys()))
    restecg_val = restecg_map[restecg]

    st.markdown("### 📊 Advanced")
    with st.expander("Show advanced fields"):
        slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        slope     = st.selectbox("ST Slope", list(slope_map.keys()))
        slope_val = slope_map[slope]

        ca  = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3])

        thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
        thal     = st.selectbox("Thalassemia", list(thal_map.keys()))
        thal_val = thal_map[thal]

    st.markdown("---")
    run = st.button("🔍 Analyze Risk")

# ── Assemble input vector ─────────────────────────────────────────────────────
input_vec = np.array([[
    age, sex_val, cp_val, trestbps, chol, fbs_val,
    restecg_val, thalach, exang_val, oldpeak,
    slope_val, ca, thal_val
]])

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown("# 🫀 CardioSense")
    st.markdown("<p style='color:#8b949e;margin-top:-.8rem;'>AI-powered cardiovascular risk assessment · Powered by Random Forest</p>", unsafe_allow_html=True)
with col_badge:
    st.markdown(f"""
    <div style='text-align:right;margin-top:1.2rem'>
      <span class='risk-badge risk-low' style='font-size:.75rem'>
        Model AUC {METRICS['roc_auc']:.3f}
      </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Risk Assessment", "📊 Model Intelligence", "🕓 History"])

# ════════════════════════════════════════════════════════════════════
# TAB 1 · Risk Assessment
# ════════════════════════════════════════════════════════════════════
with tab1:
    if not run and not st.session_state.history:
        st.markdown("""
        <div class='card card-blue' style='text-align:center;padding:3rem 2rem'>
          <div style='font-size:3rem'>🫀</div>
          <h2 style='margin:.5rem 0'>Ready to Analyze</h2>
          <p style='color:#8b949e'>Fill in the patient details in the sidebar and click <strong>Analyze Risk</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Run prediction
        input_scaled = scaler.transform(input_vec)
        prob         = model.predict_proba(input_scaled)[0][1]

        # Persist to history
        if run:
            st.session_state.history.append({
                "time":  datetime.now().strftime("%H:%M:%S"),
                "age":   age,
                "sex":   sex,
                "prob":  prob,
                "input": input_vec.copy(),
            })

        # Use latest result
        latest = st.session_state.history[-1]
        prob   = latest["prob"]

        # ── Risk tier ─────────────────────────────────────────────
        if prob >= 0.60:
            tier, badge_cls, tier_icon = "High Risk",     "risk-high",  "⚠️"
            accent = "#e84545"
        elif prob >= 0.35:
            tier, badge_cls, tier_icon = "Moderate Risk", "risk-mod",   "⚡"
            accent = "#f0a500"
        else:
            tier, badge_cls, tier_icon = "Low Risk",      "risk-low",   "✅"
            accent = "#3fb950"

        # ── Gauge + summary ───────────────────────────────────────
        col_gauge, col_summary = st.columns([1, 1.6], gap="large")

        with col_gauge:
            st.markdown("### Risk Score")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 42, "color": "#e6edf3", "family": "DM Serif Display"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#30363d",
                             "tickfont": {"color": "#8b949e", "size": 11}},
                    "bar": {"color": accent, "thickness": 0.25},
                    "bgcolor": "#161b22",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  35], "color": "rgba(63,185,80,.15)"},
                        {"range": [35, 60], "color": "rgba(240,165,0,.15)"},
                        {"range": [60,100], "color": "rgba(232,69,69,.15)"},
                    ],
                    "threshold": {"line": {"color": accent, "width": 3}, "value": prob * 100},
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                height=260,
                font=dict(color="#e6edf3"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(f"""
            <div style='text-align:center;margin-top:-.5rem'>
              <span class='risk-badge {badge_cls}'>{tier_icon} {tier}</span>
            </div>
            """, unsafe_allow_html=True)

        with col_summary:
            st.markdown("### Patient Summary")
            # Key metrics grid
            mc = st.columns(3)
            metrics_data = [
                ("Age",         f"{latest['age']} yrs", ""),
                ("Sex",         latest["sex"],           ""),
                ("Max HR",      f"{thalach} bpm",        ""),
                ("BP",          f"{trestbps} mmHg",      ""),
                ("Cholesterol", f"{chol} mg/dl",         ""),
                ("Oldpeak",     f"{oldpeak}",             ""),
            ]
            for i, (lbl, val, _) in enumerate(metrics_data):
                with mc[i % 3]:
                    st.markdown(f"""
                    <div class='metric-tile' style='margin-bottom:.6rem'>
                      <div class='val'>{val}</div>
                      <div class='lbl'>{lbl}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Risk card
            if prob >= 0.60:
                msg = "Significant cardiovascular markers detected. Prompt clinical follow-up is recommended."
                card_cls = "card-red"
            elif prob >= 0.35:
                msg = "Some risk indicators present. Lifestyle modifications and regular monitoring advised."
                card_cls = "card-amber"
            else:
                msg = "Low cardiovascular risk profile. Maintain current healthy habits."
                card_cls = "card-green"

            st.markdown(f"""
            <div class='card {card_cls}' style='margin-top:.5rem'>
              <strong>{tier_icon} {tier}</strong>
              <p style='margin:.4rem 0 0;font-size:.9rem;color:#c9d1d9'>{msg}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Feature contribution ──────────────────────────────────
        st.markdown("### 🔬 Key Risk Drivers for This Patient")

        # Compute per-feature risk contribution = input × global importance (normalized display)
        input_df  = pd.DataFrame(input_vec, columns=FEATURES)
        imp_vals  = np.array([IMPORTANCES[f] for f in FEATURES])
        # Normalize input for contribution display
        inp_norm  = scaler.transform(input_vec)[0]          # z-scores
        contrib   = np.abs(inp_norm) * imp_vals              # weighted absolute z
        contrib   = contrib / contrib.max()                  # scale 0-1

        # Sort descending
        order = np.argsort(contrib)[::-1][:8]

        icons = {
            "age": "🎂", "sex": "⚧", "cp": "💢", "trestbps": "🩸",
            "chol": "🧪", "fbs": "🍬", "restecg": "📈", "thalach": "💓",
            "exang": "🏃", "oldpeak": "📉", "slope": "📐", "ca": "🔬", "thal": "🧬",
        }

        for idx in order:
            fname = FEATURES[idx]
            label = LABELS[fname]
            icon  = icons.get(fname, "•")
            c_val = contrib[idx]
            bar_color = "#e84545" if c_val > 0.65 else "#f0a500" if c_val > 0.35 else "#3fb950"
            raw_val = input_vec[0][idx]
            st.markdown(f"""
            <div class='factor-row'>
              <span class='factor-icon'>{icon}</span>
              <span class='factor-name'>{label}</span>
              <div class='factor-bar'>
                <div class='factor-fill' style='width:{int(c_val*100)}%;background:{bar_color}'></div>
              </div>
              <span class='factor-val'>{c_val:.0%}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Clinical recommendations ──────────────────────────────
        st.markdown("### 📋 Clinical Recommendations")

        recs = []
        if prob >= 0.60:
            recs += [
                ("🏥", "Urgent referral", "Schedule a cardiology consultation promptly. Consider stress testing and echocardiogram."),
                ("💊", "Medication review", "Evaluate need for antiplatelet therapy, statins, or beta-blockers with a physician."),
                ("📉", "Lifestyle overhaul", "Immediately adopt a low-sodium, heart-healthy diet and supervised exercise program."),
                ("🚫", "Avoid triggers", "Abstain from smoking, excessive alcohol, and high-intensity unsupervised exercise."),
            ]
        elif prob >= 0.35:
            recs += [
                ("🩺", "Schedule a check-up", "Book a preventive cardiology appointment within the next 1–3 months."),
                ("🥗", "Diet optimization", "Reduce saturated fats, increase omega-3 intake, limit sodium below 2,300 mg/day."),
                ("🏃", "Graded exercise", "Begin with 150 min/week moderate-intensity aerobic activity as tolerated."),
                ("📊", "Monitor vitals", "Track blood pressure and cholesterol regularly. Aim BP < 130/80 mmHg."),
            ]
        else:
            recs += [
                ("✅", "Stay on track", "Your current health profile is favorable. Continue preventive habits."),
                ("🥗", "Maintain diet", "Sustain a balanced diet rich in fruits, vegetables, whole grains, and lean protein."),
                ("🏃", "Stay active", "Keep up with at least 150 min/week of moderate exercise."),
                ("🔁", "Annual screening", "Recheck lipid panel and blood pressure annually as preventive monitoring."),
            ]

        # Additional context-specific tips
        if chol > 240:
            recs.append(("🧪", "High cholesterol detected", f"Serum cholesterol {chol} mg/dl exceeds 240 mg/dl threshold. Discuss statin therapy with a physician."))
        if trestbps > 140:
            recs.append(("🩸", "Elevated blood pressure", f"Resting BP {trestbps} mmHg is above the 140 mmHg threshold. Lifestyle and pharmacological management warranted."))
        if exang_val == 1:
            recs.append(("💢", "Exercise-induced angina", "Exercise-induced chest pain is a significant marker. Avoid high-intensity exertion without physician clearance."))
        if ca > 1:
            recs.append(("🔬", "Vessel obstruction", f"{ca} major vessels flagged. This strongly correlates with atherosclerotic disease. Further imaging is advised."))

        rec_cols = st.columns(2)
        for i, (icon, title, body) in enumerate(recs):
            with rec_cols[i % 2]:
                st.markdown(f"""
                <div class='card' style='margin-bottom:.75rem'>
                  <div style='font-size:1.4rem'>{icon}</div>
                  <strong style='font-size:.95rem'>{title}</strong>
                  <p style='font-size:.85rem;color:#8b949e;margin:.3rem 0 0'>{body}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <p style='color:#8b949e;font-size:.8rem;margin-top:1rem'>
        ⚕️ <em>This tool is for informational purposes only and does not constitute medical advice.
        Always consult a qualified healthcare professional for diagnosis and treatment.</em>
        </p>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 · Model Intelligence
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Performance")

    perf_cols = st.columns(5)
    perf_data = [
        ("Accuracy",  f"{METRICS['accuracy']:.1%}"),
        ("ROC-AUC",   f"{METRICS['roc_auc']:.3f}"),
        ("F1 Score",  f"{METRICS['f1']:.3f}"),
        ("Precision", f"{METRICS['precision']:.1%}"),
        ("Recall",    f"{METRICS['recall']:.1%}"),
    ]
    for col, (label, val) in zip(perf_cols, perf_data):
        with col:
            st.markdown(f"""
            <div class='metric-tile'>
              <div class='val'>{val}</div>
              <div class='lbl'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='card card-blue' style='margin-top:1rem'>
      <strong>5-Fold Cross-Validation AUC</strong>
      <p style='font-size:1.4rem;font-family:DM Serif Display,serif;margin:.2rem 0'>
        {METRICS['cv_auc_mean']:.4f} <span style='font-size:1rem;color:#8b949e'>± {METRICS['cv_auc_std']:.4f}</span>
      </p>
      <p style='color:#8b949e;font-size:.85rem;margin:0'>Algorithm: Random Forest · 300 estimators · Balanced class weights</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Global Feature Importance")

    imp_df = (
        pd.DataFrame({"Feature": list(LABELS.values()),
                      "Importance": [IMPORTANCES[k] for k in FEATURES]})
        .sort_values("Importance", ascending=True)
    )

    fig_imp = go.Figure(go.Bar(
        x=imp_df["Importance"],
        y=imp_df["Feature"],
        orientation="h",
        marker=dict(
            color=imp_df["Importance"],
            colorscale=[[0, "#388bfd"], [0.5, "#f0a500"], [1, "#e84545"]],
            showscale=False,
        ),
        text=[f"{v:.3f}" for v in imp_df["Importance"]],
        textposition="outside",
        textfont=dict(color="#8b949e", size=11),
    ))
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=60, t=10, b=10),
        height=420,
        xaxis=dict(showgrid=True, gridcolor="#30363d", color="#8b949e", showline=False),
        yaxis=dict(color="#e6edf3", tickfont=dict(size=12)),
        bargap=0.3,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.markdown("### Feature Reference Guide")

    guide = [
        ("age",      "Patient age in years. Risk increases significantly after 45 (M) / 55 (F)."),
        ("sex",      "Biological sex. Males carry higher baseline risk before age 65."),
        ("cp",       "Chest pain type. Asymptomatic (3) paradoxically correlates with highest disease risk."),
        ("trestbps", "Resting blood pressure. Normal < 120/80 mmHg; ≥ 140 indicates Stage 2 hypertension."),
        ("chol",     "Serum cholesterol. Optimal < 200 mg/dl; ≥ 240 is high and warrants treatment."),
        ("fbs",      "Fasting blood sugar > 120 mg/dl suggests impaired glucose tolerance or diabetes."),
        ("restecg",  "Resting ECG. LV hypertrophy and ST-T abnormalities are independent risk markers."),
        ("thalach",  "Max heart rate during stress test. Lower max HR can signal reduced cardiac reserve."),
        ("exang",    "Chest pain triggered by exercise — a classic sign of obstructive coronary disease."),
        ("oldpeak",  "ST depression during exercise. Values > 2 are clinically significant."),
        ("slope",    "Slope of peak ST segment. Downsloping is the most ominous pattern."),
        ("ca",       "Number of major coronary vessels colored by fluoroscopy (0–3). Higher = worse."),
        ("thal",     "Thalassemia: Normal < Fixed < Reversible defect in terms of disease association."),
    ]

    g1, g2 = st.columns(2)
    for i, (key, desc) in enumerate(guide):
        with (g1 if i % 2 == 0 else g2):
            st.markdown(f"""
            <div class='card' style='padding:1rem;margin-bottom:.6rem'>
              <strong style='font-size:.85rem'>{LABELS[key]}</strong>
              <p style='font-size:.82rem;color:#8b949e;margin:.2rem 0 0'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 3 · History
# ════════════════════════════════════════════════════════════════════
with tab3:
    if not st.session_state.history:
        st.markdown("""
        <div class='card' style='text-align:center;padding:2.5rem'>
          <div style='font-size:2rem'>🕓</div>
          <p style='color:#8b949e'>No analyses run yet this session.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"### Session History  <span style='color:#8b949e;font-size:.9rem;font-weight:400'>({len(st.session_state.history)} records)</span>", unsafe_allow_html=True)

        # Chart: probability over time
        hist_df = pd.DataFrame([
            {"#": i+1, "Time": h["time"], "Risk %": round(h["prob"]*100, 1),
             "Age": h["age"], "Sex": h["sex"]}
            for i, h in enumerate(st.session_state.history)
        ])

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist_df["#"], y=hist_df["Risk %"],
            mode="lines+markers",
            line=dict(color="#e84545", width=2),
            marker=dict(size=8, color="#e84545"),
            fill="tozeroy", fillcolor="rgba(232,69,69,.08)",
            hovertemplate="<b>Run %{x}</b><br>Risk: %{y:.1f}%<extra></extra>",
        ))
        fig_hist.add_hline(y=60, line_dash="dot", line_color="rgba(232,69,69,.4)", annotation_text="High risk", annotation_font_color="#8b949e")
        fig_hist.add_hline(y=35, line_dash="dot", line_color="rgba(240,165,0,.4)", annotation_text="Moderate risk", annotation_font_color="#8b949e")
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=20, b=10), height=220,
            xaxis=dict(showgrid=False, color="#8b949e", title="Run #"),
            yaxis=dict(showgrid=True, gridcolor="#30363d", color="#8b949e", title="Risk %", range=[0, 105]),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Rows
        for i, h in enumerate(reversed(st.session_state.history)):
            prob = h["prob"]
            if prob >= 0.60:   badge = f"<span class='risk-badge risk-high'>High</span>"
            elif prob >= 0.35: badge = f"<span class='risk-badge risk-mod'>Moderate</span>"
            else:              badge = f"<span class='risk-badge risk-low'>Low</span>"

            st.markdown(f"""
            <div class='history-row'>
              <span style='color:#8b949e'>#{len(st.session_state.history)-i}</span>
              <span>{h['time']}</span>
              <span>Age {h['age']} · {h['sex']}</span>
              <span style='font-weight:600'>{prob*100:.1f}%</span>
              {badge}
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑 Clear History", key="clear_hist"):
            st.session_state.history = []
            st.rerun()
