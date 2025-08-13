# app.py
# Streamlit dashboard for economists — simple, visual, and jargon-light

import streamlit as st
import pandas as pd
import numpy as np

# Matplotlib/Seaborn figures are supported natively by Streamlit
import matplotlib.pyplot as plt
import seaborn as sns

# ========= APP CONFIG (must be first Streamlit call) =========
st.set_page_config(page_title="Crisis Early-Warning", layout="wide")

# ========= YOUR MODELING CODE =========
from econ_model import (
    make_features, BASE_FEATURES,
    train_on_A_test_on_B, loco,
    plot_confusion_matrix_custom, plot_timeline_probs, plot_pr_curve,
    plot_reliability, plot_logreg_coefficients, plot_feature_importance,
    events_from_predictions, estimate_lead_times, plot_risk_strip,
    transfer_matrix, plot_transfer_heatmap,
    probs_from_simulations, plot_fan_chart, forecast_table, BASE_VARS,
    summarize_forecast, prettify_forecast_table
)

# ========= DATA LOADING =========
@st.cache_data
def load_data():
    idn = pd.read_csv("indonesia_processed.csv")
    mys = pd.read_csv("malaysia_processed.csv")
    phl = pd.read_csv("philippines_processed.csv")
    tha = pd.read_csv("thailand_processed.csv")
    return {"IDN": idn, "MYS": mys, "PHL": phl, "THA": tha}

def best_models_by_country(data_dict):
    # Run LOCO once, pick best per country, return both the small table and artifacts
    tbl, arts = loco(data_dict, BASE_FEATURES, beta=2.0, top_k_per_country=1)
    best = (tbl.sort_values(["test_country","PR_AUC","recall_1"], ascending=[True,False,False])
              .groupby("test_country").first().reset_index())
    best_art = {}
    for _, r in best.iterrows():
        c = r["test_country"]; m = r["model"]
        best_art[c] = arts[c][m]  # contains fitted pipeline & feature_cols
    return best, best_art

DATA = load_data()
COUNTRY_NAMES = {"IDN":"Indonesia","MYS":"Malaysia","PHL":"Philippines","THA":"Thailand"}

# ========= SMALL HELPERS (display) =========
def kpi_cards(container, pr_auc, hit_rate, fa_share, widths=(1,1,1)):
    c1, c2, c3 = container.columns(widths)
    c1.metric("Overall signal (PR-AUC)", f"{pr_auc:.3f}",
              help="Area under the Precision-Recall curve. Higher is better.")
    c2.metric("Hit rate on crises", f"{hit_rate:.0%}",
              help="Of the crisis years, how many did the model catch?")
    c3.metric("False-alarm share", f"{fa_share:.0%}",
              help="Of all flagged years, how many were not crises? Lower is better.")

def head_note(txt):
    st.caption(txt)

def compute_simple_kpis(row):
    tp, fp, fn = int(row["tp"]), int(row["fp"]), int(row["fn"])
    pr_auc = float(row["PR_AUC"])
    hit_rate = 0.0 if (tp+fn)==0 else tp/(tp+fn)
    fa_share = 0.0 if (tp+fp)==0 else fp/(fp+tp)  # corrected denominator
    return pr_auc, hit_rate, fa_share

def executive_summary_block(country_b, art_entry, best_row):
    """Plain-language bullet points economists can use quickly."""
    years = art_entry["year"]; y_true = art_entry["y_true"]
    y_pred = art_entry["y_pred"]; y_prob = art_entry["y_prob"]
    thr = art_entry["threshold"]

    ev = events_from_predictions(years, y_true, y_pred)
    leads = estimate_lead_times(years, y_true, y_prob, thr)

    tp, fp, fn = int(best_row["tp"]), int(best_row["fp"]), int(best_row["fn"])
    bullets = []
    bullets.append(f"**Crisis years**: {', '.join(map(str, ev['crisis_years'])) or '—'}")
    if ev["tp_years"]:
        bullets.append(f"**Detected crises**: {', '.join(map(str, ev['tp_years']))} "
                       f"({' & '.join([str(x)+'y lead' for x in leads]) if leads else 'lead time: n/a'})")
    else:
        bullets.append("**Detected crises**: none")
    bullets.append(f"**Missed crises**: {', '.join(map(str, ev['missed_years'])) or 'none'}")
    bullets.append(f"**False alarms**: {', '.join(map(str, ev['false_alarm_years'])) or 'none'}")

    st.markdown("### Executive summary")
    for b in bullets:
        st.markdown(f"- {b}")

    # extra small KPI row
    c1, c2, c3 = st.columns(3)
    c1.metric("Crisis years", f"{sum(y_true)}")
    c2.metric("Caught (TP)", f"{tp}")
    c3.metric("Missed (FN) / False (FP)", f"{fn} / {fp}")

# ========= PAGE: A → B =========
def page_train_A_to_B():
    st.title("Train on A → Test on B")
    head_note("Compare two countries: learn patterns on Country A, assess on Country B.")

    # Sidebar controls
    st.sidebar.header("Settings")
    a = st.sidebar.selectbox("Train on (A)", list(COUNTRY_NAMES.keys()), format_func=lambda k: COUNTRY_NAMES[k], index=0)
    b = st.sidebar.selectbox("Test on (B)",  list(COUNTRY_NAMES.keys()), format_func=lambda k: COUNTRY_NAMES[k], index=1)
    beta = st.sidebar.slider("Threshold preference (β for Fβ)", 0.5, 4.0, 2.0, 0.5,
                             help="Higher β favors recall (catching crises) over precision.")
    n_splits = st.sidebar.slider("Time folds (walk-forward)", 2, 6, 4)
    min_years = st.sidebar.slider("Earliest train start (min years)", 4, 12, 8)

    if a == b:
        st.info("Pick different countries for A and B.")
        return

    # Run
    with st.spinner("Training and evaluating…"):
        tbl, art = train_on_A_test_on_B(
            DATA[a], DATA[b], BASE_FEATURES,
            beta=float(beta), n_splits=int(n_splits),
            min_train_years=int(min_years)
        )

    # Pick best row and its artifacts
    best_row = tbl.iloc[0]
    model_name = best_row["model"]
    st.subheader(f"Best model on {COUNTRY_NAMES[b]} (trained on {COUNTRY_NAMES[a]}): **{model_name}**")
    art_a = art[model_name]

    # KPIs
    pr_auc, hit_rate, fa_share = compute_simple_kpis(best_row)
    kpi_cards(st, pr_auc, hit_rate, fa_share)

    # Executive summary (plain text KPIs)
    executive_summary_block(COUNTRY_NAMES[b], art_a, best_row)

    # Risk strip (very visual)
    fig, ax = plt.subplots(figsize=(9, 1.3))
    plot_risk_strip(art_a["year"], art_a["y_prob"], art_a["threshold"], ax=ax)
    st.pyplot(fig)

    # Confusion matrix & timeline
    st.markdown("#### What it means in years")
    col1, col2 = st.columns([1.1,1.0])

    with col1:
        st.caption("Predicted crisis probability (dashed = decision threshold; red shading = actual crisis years)")
        fig, ax = plt.subplots(figsize=(9,3.2))
        plot_timeline_probs(
            art_a["year"], art_a["y_true"], art_a["y_prob"], art_a["threshold"],
            f"{COUNTRY_NAMES[b]} — Crisis probability ({model_name})", ax=ax
        )
        st.pyplot(fig)

    with col2:
        st.caption("Confusion matrix (counts)")
        fig, ax = plt.subplots(figsize=(4.2,3.6))
        plot_confusion_matrix_custom(
            art_a["y_true"], art_a["y_pred"],
            f"{COUNTRY_NAMES[b]} — Confusion ({model_name})", ax=ax
        )
        st.pyplot(fig)

    # Optional diagnostics
    with st.expander("More diagnostics (optional)"):
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(4.8,3.8))
            plot_pr_curve(
                art_a["y_true"], art_a["y_prob"],
                f"{COUNTRY_NAMES[b]} — PR curve ({model_name})", ax=ax
            )
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(4.8,3.8))
            plot_reliability(
                art_a["y_true"], art_a["y_prob"],
                title=f"{COUNTRY_NAMES[b]} — Calibration ({model_name})", ax=ax
            )
            st.pyplot(fig)

        # Drivers (if available)
        st.caption("Economic drivers behind the signal")
        c3, c4 = st.columns(2)
        with c3:
            fig, ax = plt.subplots(figsize=(6,3.5))
            ax_or_none = plot_logreg_coefficients(
                art_a["best"], art_a["feature_cols"],
                title=f"{COUNTRY_NAMES[b]} — Drivers (LogReg)", ax=ax
            )
            if ax_or_none is not None:
                st.pyplot(fig)
            else:
                st.info("Logistic coefficients are not available for this model.")
        with c4:
            fig, ax = plt.subplots(figsize=(6,3.5))
            ax_or_none = plot_feature_importance(
                art_a["best"], art_a["feature_cols"],
                title=f"{COUNTRY_NAMES[b]} — Drivers (Tree)", ax=ax
            )
            if ax_or_none is not None:
                st.pyplot(fig)
            else:
                st.info("Tree feature importances are not available for this model.")

# ========= PAGE: LOCO =========
def page_loco():
    st.title("Leave-One-Country-Out (LOCO)")
    head_note("Train on three countries, test on the held-out country. Shows the best model per country.")

    with st.spinner("Running LOCO…"):
        tbl, arts = loco(DATA, BASE_FEATURES, beta=2.0, top_k_per_country=1)

    best_per = (tbl.sort_values(["test_country","PR_AUC","recall_1"], ascending=[True,False,False])
                   .groupby("test_country").first().reset_index())

    st.markdown("#### Country snapshots")
    for _, r in best_per.iterrows():
        c = r["test_country"]; model_name = r["model"]
        pr_auc, hit_rate, fa_share = compute_simple_kpis(r)

        with st.container():
            st.write(f"**{COUNTRY_NAMES[c]}** — best model: **{model_name}**")
            kpi_cards(st, pr_auc, hit_rate, fa_share)
            art = arts[c][model_name]

            # Executive summary per country
            executive_summary_block(COUNTRY_NAMES[c], art, r)

            # Risk strip
            fig, ax = plt.subplots(figsize=(9, 1.3))
            plot_risk_strip(art["year"], art["y_prob"], art["threshold"], ax=ax)
            st.pyplot(fig)

            col1, col2 = st.columns([1.1,1.0])
            with col1:
                fig, ax = plt.subplots(figsize=(9,3.2))
                plot_timeline_probs(
                    art["year"], art["y_true"], art["y_prob"], art["threshold"],
                    f"{COUNTRY_NAMES[c]} — Crisis probability ({model_name})", ax=ax
                )
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(4.2,3.6))
                plot_confusion_matrix_custom(
                    art["y_true"], art["y_pred"],
                    f"{COUNTRY_NAMES[c]} — Confusion ({model_name})", ax=ax
                )
                st.pyplot(fig)

            st.divider()
    with st.expander("Cross-country transfer map (PR-AUC)"):
        st.caption("How well patterns trained on A transfer to B (higher is better).")
        with st.spinner("Computing A → B performance matrix…"):
            M = transfer_matrix(DATA, BASE_FEATURES, beta=2.0, n_splits=4, min_train_years=8)
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        plot_transfer_heatmap(M, COUNTRY_NAMES, ax=ax)
        st.pyplot(fig)

def page_forecast():
    st.title("Forecast 2021–2025 (probabilities)")
    head_note("Use the best cross-country model per market (from LOCO), run a VAR on macro drivers, and show crisis probabilities for the next 5 years under baseline and stress scenarios.")

    # Controls
    st.sidebar.header("Forecast settings")
    sel_country = st.sidebar.selectbox("Country", list(COUNTRY_NAMES.keys()),
                                       format_func=lambda k: COUNTRY_NAMES[k], index=0)
    sims = st.sidebar.slider("Monte Carlo simulations", 100, 2000, 500, 100)
    steps = st.sidebar.slider("Years ahead", 3, 6, 5)
    seed = st.sidebar.number_input("Random seed", value=123, step=1)

    # Get best model per country (cached)
    with st.spinner("Selecting best models (LOCO)…"):
        best_tbl, best_art = best_models_by_country(DATA)

    # Wrap artifact to the format expected by probs_from_simulations
    best_entry = {"art": best_art[sel_country]}

    with st.spinner("Simulating macro paths (VAR) and scoring probabilities…"):
        results = probs_from_simulations(
            country_code=sel_country,
            hist_df=DATA[sel_country],
            best_entry=best_entry,
            steps=int(steps),
            sims=int(sims),
            scenarios=('baseline','mild','severe'),
            seed=int(seed)
        )

     # Top strip with a quick read
    st.subheader(f"{COUNTRY_NAMES[sel_country]} — 5-year crisis probability bands")
    st.caption("Shaded area shows the 5–95% range across simulations; lines show mean and median. Values are probabilities, not certainties.")

    # Charts in tabs (clearer, bigger)
    tabs = st.tabs(["Baseline", "Mild shock", "Severe shock"])
    for tab, sc in zip(tabs, ['baseline','mild','severe']):
        with tab:
            fig, ax = plt.subplots(figsize=(7.2, 3.8))
            plot_fan_chart(sel_country, results, scenario=sc, ax=ax)
            st.pyplot(fig)

    st.markdown("#### Key takeaways")
    take = summarize_forecast(results)
    st.dataframe(
        take.style.format({'Peak prob (%)':'{:.1f}', 'Avg over horizon (%)':'{:.1f}'}),
        use_container_width=True
    )

    # Numbers behind the charts — compact & readable
    st.markdown("#### Numbers behind the charts")
    tidy = forecast_table(results)
    st.dataframe(
        prettify_forecast_table(tidy),
        use_container_width=True
    )

    # Little “what to watch” note
    st.info(
        "Interpretation tip: **Baseline** uses recent dynamics; **Mild/Severe** apply stylized macro shocks "
        "(GDP down, CPI up). Redder/upper bands suggest higher crisis risk. Use alongside judgment and other indicators."
    )


# ========= APP LAYOUT =========
with st.sidebar:
    st.markdown("### Crisis Early-Warning Dashboard")
    page = st.radio("Choose view", ["Train A → B", "LOCO", "Forecast 2021–2025"])

if page == "Train A → B":
    page_train_A_to_B()
elif page == "LOCO":
    page_loco()
else:
    page_forecast()