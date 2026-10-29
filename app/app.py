"""
CardioSense AI - Streamlit dashboard for cardiovascular disease risk prediction (educational use only).
Run locally:  streamlit run app/app.py
All numbers shown here are read from files written by train_models.py / clustering.py.
The layout is responsive: it adapts to phones, tablets and desktops (see the CSS block below).
"""
import json
from html import escape
from pathlib import Path
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
NAVY, TEAL, CORAL, MUTED = "#0B2545", "#0E9AA7", "#E4572E", "#8DA2B5"
st.set_page_config(page_title="CardioSense AI — Cardiovascular Risk Prediction & Analytics", page_icon="🫀", layout="wide")

# ---- Styling. Desktop first; the @media blocks adapt it for tablets (<=768px) and phones (<=640px).
st.markdown(f"""<style>
.block-container {{max-width: 1100px; padding-top: 1.2rem; padding-bottom: 3rem;}}
header[data-testid="stHeader"] {{background: transparent;}}
section[data-testid="stSidebar"] {{background: {NAVY};}}
section[data-testid="stSidebar"] * {{color: #E6EEF5 !important;}}
.hero {{background: linear-gradient(120deg, {NAVY} 0%, #134074 60%, {TEAL} 130%); color: #fff; border-radius: 16px;
        padding: 1.4rem 1.6rem; margin-bottom: .8rem;}}
.hero h1 {{margin: 0; font-size: 1.9rem; color: #fff; letter-spacing: .2px;}}
.hero p {{margin: .3rem 0 0; color: #CFE6EA; font-size: .95rem;}}
.disclaimer {{background: #FFF6E5; border-left: 4px solid #E8A33D; color: #5B4300; padding: .6rem .9rem; border-radius: 8px;
              font-size: .85rem; margin-bottom: 1rem;}}
.kpi-grid {{display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .75rem; margin-bottom: 1rem;}}
.kpi {{background: #fff; border-radius: 14px; padding: .9rem 1.1rem; border: 1px solid #DDE7EE; border-top: 4px solid {TEAL};
       box-shadow: 0 1px 4px rgba(11,37,69,.06); min-height: 5.6rem;}}
.kpi .l {{font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; color: #5B7186;}}
.kpi .v {{font-size: 1.2rem; font-weight: 700; color: {NAVY}; line-height: 1.3; overflow-wrap: anywhere;}}
.kpi .s {{font-size: .78rem; color: #5B7186;}}
.panel {{background: #fff; border: 1px solid #DDE7EE; border-radius: 14px; padding: 1rem 1.2rem;
         box-shadow: 0 1px 4px rgba(11,37,69,.06);}}
.result {{border-radius: 16px; padding: 1.3rem 1.4rem; color: #fff; text-align: center;}}
.result .big {{font-size: 3rem; font-weight: 800; line-height: 1.1;}}
.result .tag {{font-size: .85rem; letter-spacing: .12em; text-transform: uppercase; opacity: .9;}}
.bar {{height: 10px; border-radius: 6px; background: rgba(255,255,255,.3); margin-top: .8rem; overflow: hidden;}}
.bar > div {{height: 100%; background: #fff;}}
.note {{font-size: .82rem; color: #5B7186;}}
div[data-testid="stCaptionContainer"] {{font-size: .82rem;}}
.table-wrap {{overflow-x: auto; margin-bottom: .6rem;}}
.mini-table {{width: 100%; border-collapse: collapse; font-size: .85rem; background: #fff;}}
.mini-table th {{background: #EAF2F7; color: {NAVY}; text-align: right; padding: .45rem .5rem; font-weight: 600;}}
.mini-table td {{text-align: right; padding: .45rem .5rem; border-top: 1px solid #E3EBF1; color: {NAVY};}}
.mini-table th:first-child, .mini-table td:first-child {{text-align: left; overflow-wrap: anywhere; min-width: 5.5rem;}}
.mini-table tr.hl td {{background: #E3F4F6; font-weight: 700;}}
h2, h3 {{color: {NAVY};}}
div[data-testid="stMetric"] {{background: #fff; border: 1px solid #DDE7EE; border-radius: 12px; padding: .6rem .8rem;}}
button[kind="primary"] {{background: {TEAL}; border: 0;}}

/* Tablets and phones: larger touch targets (at least ~44px) */
@media (max-width: 768px), (pointer: coarse) {{
  div[data-testid="stNumberInputContainer"], div[data-testid="stSelectbox"] div[role="group"] {{min-height: 46px;}}
  div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] input {{min-height: 44px; font-size: 1rem;}}
  div[data-testid="stNumberInput"] button {{min-width: 46px; min-height: 46px;}}
  .stButton > button {{min-height: 50px; font-size: 1.05rem;}}
  div[data-testid="stCheckbox"] label, div[data-testid="stRadio"] label {{min-height: 44px; align-items: center;}}
  div[data-testid="stSlider"] div[role="group"], div[data-testid="stSlider"] div[role="group"] > div {{min-height: 46px;}}
  div[data-testid="stSlider"] div[role="group"] > div > div[data-rac] {{width: 26px; height: 26px;}}
  div[data-testid="stTab"] {{min-height: 46px;}}
}}
/* Phones */
@media (max-width: 640px) {{
  .block-container {{padding: 3.4rem 1rem 3rem 1rem;}}
  .hero {{padding: 1rem 1.1rem;}} .hero h1 {{font-size: 1.45rem;}} .hero p {{font-size: .9rem;}}
  .kpi-grid {{grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .5rem;}}
  .kpi {{padding: .65rem .75rem; min-height: 5.2rem;}} .kpi .v {{font-size: 1.05rem;}}
  .result .big {{font-size: 2.4rem;}}
  .mini-table {{font-size: .78rem;}} .mini-table th, .mini-table td {{padding: .4rem .3rem;}}
  div[role="tablist"] {{flex-wrap: wrap; gap: .3rem; overflow: visible;}}
  div[data-testid="stTab"] {{padding: 0 .8rem; border-radius: 10px; background: #EAF2F7;}}
  div[data-testid="stTab"][aria-selected="true"] {{background: {TEAL};}}
  div[data-testid="stTab"][aria-selected="true"], div[data-testid="stTab"][aria-selected="true"] * {{color: #fff !important;}}
  div[data-testid="stTab"] .react-aria-SelectionIndicator {{display: none;}}
}}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    # joblib/pickle: safe here because the file is produced locally by train_models.py, never from user input
    return joblib.load(ROOT / "models" / "best_model.joblib")


@st.cache_data
def load_csv(name, **kw):
    p = OUT / name
    return pd.read_csv(p, **kw) if p.exists() else None


def show_img(name, caption):
    p = OUT / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


def kpi_grid(items):
    """Metric cards: 4 in a row on desktop/tablet, 2 x 2 on phones (pure CSS grid)."""
    cards = "".join(f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
                    for l, v, s in items)
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def html_table(df, highlight=None):
    """Small table that wraps text and fits any screen. `highlight` = value of the first column to emphasise."""
    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    rows = "".join(f'<tr{" class=hl" if highlight is not None and r.iloc[0] == highlight else ""}>'
                   + "".join(f"<td>{escape(str(v))}</td>" for v in r) + "</tr>" for _, r in df.iterrows())
    st.markdown(f'<div class="table-wrap"><table class="mini-table"><tr>{head}</tr>{rows}</table></div>', unsafe_allow_html=True)


bundle = load_model()
model, features = bundle["model"], bundle["features"]
metrics = json.loads((OUT / "metrics.json").read_text()) if (OUT / "metrics.json").exists() else None
res = load_csv("model_comparison.csv")

# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.markdown("### 🫀 CardioSense AI")
    st.caption("Cardiovascular Risk Prediction & Analytics")
    st.markdown(f"**Final model**  \n{bundle['name']}")
    st.markdown("**Selection rule**  \nHighest validation ROC-AUC")
    st.markdown(f"**Training data**  \n{bundle.get('n_train', 'n/a'):,} cleaned records (70/15/15 split)")
    st.divider()
    st.markdown("**Educational use only.** Not a medical device. Not for diagnosis or treatment decisions. "
                "Consult a qualified doctor for health concerns.")
    st.divider()
    st.caption("Tabs: Predict · Threshold analysis · Model performance · Data insights")

# ------------------------------------------------------------------ Header, disclaimer + KPIs
st.markdown('<div class="hero"><h1>CardioSense AI</h1>'
            '<p><b>Cardiovascular Risk Prediction &amp; Analytics.</b> Estimate cardiovascular disease risk from 11 routine '
            'health measurements and explore how the model behaves.</p></div>'
            '<div class="disclaimer"><b>Educational use only.</b> Not a medical device; not for diagnosis or treatment decisions. '
            'Consult a qualified doctor for health concerns.</div>', unsafe_allow_html=True)
if metrics:
    t = metrics["test"]
    kpi_grid([("Final model", metrics["final_model"], "picked on validation"),
              ("Test ROC-AUC", f"{t['ROC_AUC']:.3f}", "held-out test set"),
              ("Test accuracy", f"{t['Accuracy']:.1%}", f"precision {t['Precision']:.1%}"),
              ("Test recall", f"{t['Recall']:.1%}", f"F1 {t['F1']:.3f}")])

tab_pred, tab_thr, tab_perf, tab_data = st.tabs(["Predict", "Threshold analysis", "Model performance", "Data insights"])

# ------------------------------------------------------------------ Predict
with tab_pred:
    form, panel = st.columns([3, 2], gap="large")
    with form:
        st.markdown("#### Patient measurements")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", 18, 100, 50)
            gender = st.selectbox("Gender", ["Female", "Male"])
            height = st.number_input("Height (cm)", 130, 210, 165)
            weight = st.number_input("Weight (kg)", 35.0, 200.0, 70.0)
            ap_hi = st.number_input("Systolic BP (ap_hi, mmHg)", 70, 250, 120)
            ap_lo = st.number_input("Diastolic BP (ap_lo, mmHg)", 40, 200, 80)
        with col2:
            lv = {1: "Normal", 2: "Above normal", 3: "Well above normal"}
            chol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lv.get)
            gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lv.get)
            smoke = st.checkbox("Smoker")
            alco = st.checkbox("Drinks alcohol")
            active = st.checkbox("Physically active", value=True)
        go = st.button("Predict risk", type="primary", width="stretch")

    with panel:
        st.markdown("#### Prediction")
        bmi = weight / (height / 100) ** 2
        if not go:
            st.markdown(f'<div class="panel note">Enter the measurements and press <b>Predict risk</b>. '
                        f'Current BMI: <b>{bmi:.1f}</b>.</div>', unsafe_allow_html=True)
        elif ap_hi <= ap_lo:
            st.error("Systolic pressure must be higher than diastolic pressure.")
        elif not 12 <= bmi <= 60:
            st.error(f"BMI {bmi:.1f} is outside the range the model was trained on (12-60). Check height and weight.")
        else:
            row = pd.DataFrame([{
                "age_years": age, "gender": 1 if gender == "Female" else 2,
                "height": height, "weight": weight, "bmi": bmi,
                "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": chol, "gluc": gluc,
                "smoke": int(smoke), "alco": int(alco), "active": int(active)}])[features]
            p = float(model.predict_proba(row)[0, 1])
            band, color = (("High estimated risk", CORAL) if p >= 0.6 else
                           ("Moderate estimated risk", "#E8A33D") if p >= 0.4 else ("Low estimated risk", TEAL))
            st.markdown(f'<div class="result" style="background:{color}"><div class="tag">{band}</div>'
                        f'<div class="big">{p:.1%}</div><div class="tag">probability of cardiovascular disease</div>'
                        f'<div class="bar"><div style="width:{min(max(p, 0), 1) * 100:.0f}%"></div></div></div>',
                        unsafe_allow_html=True)
            st.markdown(f'<p class="note">BMI used by the model: {bmi:.1f}. Risk bands (0.4 / 0.6) are illustrative, '
                        f'not clinical cut-offs. {"Please consult a doctor." if p >= 0.6 else ""}</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ Threshold analysis
with tab_thr:
    st.subheader("Decision-threshold analysis")
    st.write("The model outputs a probability. The cut-off above which a patient is flagged as *disease* trades "
             "**recall** (catching sick patients) against **precision** (avoiding false alarms).")
    st.caption("The Predict tab labels a probability below 0.4 as Low, 0.4 to 0.6 as Moderate and 0.6 or above as High "
               "estimated risk. These two cut-offs are fixed, illustrative labels, not clinical thresholds and not "
               "tuned on data; they are separate from the decision threshold chosen below, which only affects the "
               "metrics on this tab.")
    pr = load_csv("best_model_predictions.csv")
    if pr is None:
        st.info("Run `python train_models.py` first to create outputs/best_model_predictions.csv.")
    else:
        split = st.radio("Data split", ["validation", "test"], horizontal=True,
                         help="Use validation to choose a threshold; test is the untouched final check.")
        thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
        d = pr[pr["split"] == split]
        y, pb = d["y_true"].to_numpy(), d["proba"].to_numpy()

        def at(t):
            pred = pb >= t
            tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
            fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            return dict(tp=tp, fp=fp, fn=fn, tn=tn, Accuracy=(tp + tn) / len(y), Precision=prec, Recall=rec, F1=f1)

        m = at(thr)
        kpi_grid([(k, f"{m[k]:.3f}", f"{split} set, cut-off {thr:.2f}") for k in ["Accuracy", "Precision", "Recall", "F1"]])
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.caption("Confusion matrix at this threshold (rows = actual, columns = predicted)")
            html_table(pd.DataFrame([["No disease", m["tn"], m["fp"]], ["Disease", m["fn"], m["tp"]]],
                                    columns=["Actual", "Pred. no", "Pred. yes"]))
        with right:
            ts = np.round(np.arange(0.05, 0.96, 0.01), 2)
            curve = pd.DataFrame([{"threshold": t, **{k: at(t)[k] for k in ["Precision", "Recall", "F1"]}} for t in ts])
            long = curve.melt("threshold", var_name="Metric", value_name="Score")
            line = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
                x=alt.X("threshold:Q", title="Decision threshold"), y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Metric:N", scale=alt.Scale(domain=["Precision", "Recall", "F1"], range=[NAVY, TEAL, CORAL]),
                                legend=alt.Legend(orient="bottom", direction="horizontal", title=None, labelFontSize=11,
                                                  columnPadding=6, symbolSize=50, labelExpr="datum.label == 'Precision' ? 'Prec.' : datum.label")))
            rule = alt.Chart(pd.DataFrame({"threshold": [thr]})).mark_rule(color=MUTED, strokeDash=[5, 4]).encode(x="threshold:Q")
            st.altair_chart((line + rule).properties(height=280), width="stretch")
        best_f1_t = float(curve.loc[curve["F1"].idxmax(), "threshold"])
        st.caption(f"On the {split} set, F1 is highest at threshold {best_f1_t:.2f}. "
                   "Lowering the threshold raises recall but lowers precision.")

# ------------------------------------------------------------------ Performance
with tab_perf:
    if res is not None:
        st.subheader("Model comparison")
        st.caption("Same 70/15/15 split for every model, ordered by **validation** ROC-AUC. The final model is the first "
                   "row that outputs probabilities. Test columns are reported once, after selection.")
        bars = alt.Chart(res).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("Model:N", sort="-x", title=None, axis=alt.Axis(labelLimit=150, labelFontSize=11)),
            x=alt.X("Test_ROC_AUC:Q", scale=alt.Scale(domain=[0.75, 0.82], clamp=True), title="Test ROC-AUC"),
            color=alt.condition(alt.datum.Model == bundle["name"], alt.value(TEAL), alt.value("#B7C7D6")),
            tooltip=["Model", alt.Tooltip("Test_ROC_AUC:Q", format=".4f"), alt.Tooltip("Test_Accuracy:Q", format=".4f")])
        st.altair_chart(bars.properties(height=300, autosize=alt.AutoSizeParams(type="fit", contains="padding")), width="stretch")
        short = res[["Model", "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC"]].copy()
        for c in short.columns[1:]:
            short[c] = short[c].map("{:.4f}".format)
        short.columns = ["Model", "Acc.", "Prec.", "Recall", "F1", "AUC"]
        html_table(short, highlight=bundle["name"])
        with st.expander("All columns (training and validation metrics, timing)"):
            st.dataframe(res.style.format({c: "{:.4f}" for c in res.columns if c != "Model"}), width="stretch", hide_index=True)
    st.info("Decision Tree uses `criterion='entropy'` (information gain, the idea behind ID3). scikit-learn builds "
            "CART-style binary trees, so this is entropy-based but not an exact ID3 implementation. The SVM is scored "
            "with decision-function margins (no probabilities), so it is compared but not deployable.")
    st.caption("Tip: tap or click an image to enlarge it.")
    a, b = st.columns(2)
    with a:
        show_img("roc_curves.png", "ROC curves (test set)")
        show_img("feature_importance.png", "Permutation importance (validation set)")
    with b:
        show_img("confusion_matrix.png", "Confusion matrix - final model (test)")
        show_img("model_comparison.png", "Metric comparison (test set)")
    show_img("confusion_matrices_all.png", "Confusion matrices - all models (test set)")
    leak = load_csv("leakage_check.csv")
    if leak is not None:
        with st.expander("Data-leakage and feature-order checks"):
            st.dataframe(leak.astype(str), width="stretch", hide_index=True)

# ------------------------------------------------------------------ Data insights
with tab_data:
    show_img("eda.png", "Age, blood pressure and BMI vs disease")
    a, b = st.columns([3, 2], gap="large")
    with a:
        show_img("correlation.png", "Correlation matrix")
    with b:
        rep = load_csv("cleaning_report.csv", index_col=0)
        if rep is not None:
            st.subheader("Data cleaning summary")
            st.dataframe(rep, width="stretch")
    prof = load_csv("kmeans_profile.csv", index_col=0)
    if prof is not None:
        with st.expander("Exploratory K-Means clustering (unsupervised, separate from prediction)"):
            st.write("Patients were grouped by K-Means on standardised features, without using the disease label. "
                     "The label is only used afterwards to describe each cluster. This is exploratory and is not "
                     "used by the predictor.")
            show_img("kmeans_clusters.png", "Silhouette score by k, and clusters in PCA space")
            st.dataframe(prof, width="stretch")
