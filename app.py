"""CarePulse AI - 30-Day Hospital Readmission Prediction & Smart Intervention Platform.

Main Streamlit Application.
Product Tagline: "Predict Earlier. Care Smarter. Prevent Readmissions."
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    COLORS,
    DEFAULT_DATA_PATH,
    MODEL_BUNDLE_PATH,
    DEFAULT_WEEKLY_CAPACITY,
    RISK_THRESHOLDS,
    INTERVENTION_TYPES,
    LAB_REFERENCE_RANGES,
    get_risk_category,
)
from src.data_generator import generate_synthetic_cohort
from src.data_manager import load_dataset, clean_and_deduplicate, audit_dataset
from src.preprocessor import patient_grouped_split
from src.models import (
    load_model_bundle,
    predict_patient_risk,
    ModelTrainer,
)
from src.explainer import ClinicalExplainer
from src.intervention_engine import recommend_interventions
from src.call_list import (
    generate_weekly_call_list,
    record_supervisor_override,
    get_override_history,
    record_contact_outcome,
    load_contact_outcomes,
    export_call_list_csv,
)
from src.evaluator import compute_evaluation_metrics
from src.fairness import audit_fairness_subgroups
from src.simulator import simulate_outreach_impact
from src.ui_components import (
    inject_custom_css,
    render_metric_card,
    render_risk_pill,
    get_plotly_theme,
)

# Set page configuration
st.set_page_config(
    page_title="CarePulse AI | Clinical Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom clinical design system
inject_custom_css()


@st.cache_resource
def get_cached_bundle():
    """Loads trained model bundle if available, or automatically trains baseline bundle if absent."""
    if MODEL_BUNDLE_PATH.exists():
        try:
            return load_model_bundle(MODEL_BUNDLE_PATH)
        except Exception:
            pass
    try:
        # Fallback: Auto-initialize baseline model bundle using demo dataset
        clean_df, _ = get_cached_dataset()
        train_df, val_df, _ = patient_grouped_split(clean_df, test_size=0.15, val_size=0.15, random_state=42)
        trainer = ModelTrainer(random_state=42)
        bundle = trainer.train_and_select(train_df, val_df, calibration_method="sigmoid")
        trainer.save_bundle(bundle, path=MODEL_BUNDLE_PATH)
        return bundle
    except Exception:
        return None


@st.cache_data
def get_cached_dataset():
    """Loads or generates dataset."""
    if not DEFAULT_DATA_PATH.exists():
        df = generate_synthetic_cohort(num_records=25000, output_path=str(DEFAULT_DATA_PATH))
    else:
        df, _ = load_dataset(str(DEFAULT_DATA_PATH))
    clean_df = clean_and_deduplicate(df)
    audit = audit_dataset(clean_df)
    return clean_df, audit


# Initialize Session State
if "df" not in st.session_state:
    df, audit = get_cached_dataset()
    st.session_state["df"] = df
    st.session_state["audit"] = audit

if "bundle" not in st.session_state:
    st.session_state["bundle"] = get_cached_bundle()

if "risks" not in st.session_state and st.session_state["bundle"] is not None:
    st.session_state["risks"] = predict_patient_risk(st.session_state["df"], st.session_state["bundle"])
elif "risks" not in st.session_state:
    st.session_state["risks"] = None

if "selected_patient_id" not in st.session_state:
    st.session_state["selected_patient_id"] = None

if "weekly_capacity" not in st.session_state:
    st.session_state["weekly_capacity"] = DEFAULT_WEEKLY_CAPACITY


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 12px 4px 20px 4px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="background-color: {COLORS['primary_teal']}; width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px;">
                    🩺
                </div>
                <div>
                    <h2 style="margin: 0; font-size: 1.35rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em;">CarePulse AI</h2>
                    <span style="font-size: 0.76rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;">Clinical Intelligence</span>
                </div>
            </div>
            <p style="margin: 10px 0 0 0; font-size: 0.78rem; color: #CBD5E1; font-style: italic;">
                "Predict Earlier. Care Smarter. Prevent Readmissions."
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        "📊 Overview",
        "🔍 Patient Risk Explorer",
        "📋 Weekly Call List",
        "👤 Patient Detail",
        "🎯 Intervention Planner",
        "📈 Model Performance",
        "⚖️ Fairness & Bias Audit",
        "🔮 Impact Simulator",
        "🗄️ Data Management",
        "⚙️ Settings & About",
    ]

    # Pre-select Patient Detail if user clicked a patient
    default_idx = 0
    if st.session_state["selected_patient_id"] is not None:
        default_idx = 3

    page_selection = st.radio("Navigation", pages, index=default_idx, label_visibility="collapsed")

    # Bottom status card
    st.markdown("---")
    model_status_text = "Trained & Calibrated" if st.session_state["bundle"] is not None else "Model Not Trained"
    model_color = COLORS["primary_teal"] if st.session_state["bundle"] is not None else COLORS["warning_amber"]
    
    st.markdown(
        f"""
        <div style="font-size: 0.78rem; line-height: 1.5; color: #94A3B8; padding: 4px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span>Environment:</span>
                <span style="color: #FFFFFF; font-weight: 600;">Prototype v1.0 (Local)</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span>Model:</span>
                <span style="color: {model_color}; font-weight: 600;">{model_status_text}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Dataset:</span>
                <span style="color: #FFFFFF; font-weight: 600;">{len(st.session_state['df']):,} Records</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# TOP HEADER BAR
# ==========================================
st.markdown(
    f"""
    <div class="cp-top-bar">
        <div>
            <div class="cp-breadcrumb">CarePulse AI &rsaquo; {page_selection}</div>
            <div class="cp-page-title">{page_selection}</div>
        </div>
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="background-color: {COLORS['soft_mint']}; color: {COLORS['deep_teal']}; font-size: 0.76rem; font-weight: 600; padding: 6px 12px; border-radius: 20px; border: 1px solid #B8EBD9;">
                ● Synthetic Cohort (Academic Demo)
            </div>
            <div style="font-size: 0.8rem; color: {COLORS['secondary_text']};">
                Discharge Cohort: <strong>{len(st.session_state['df']):,} Encounters</strong>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Helper: check model readiness
def ensure_model_ready():
    if st.session_state["bundle"] is None:
        st.warning("⚠️ Machine learning model is not yet trained. Go to **Data Management** or **Settings & About** to train the model.")
        if st.button("🚀 Train Model Now with Calibrated XGBoost", type="primary"):
            with st.spinner("Partitioning patient-grouped cohorts and training candidate models..."):
                train_df, val_df, _ = patient_grouped_split(st.session_state["df"], test_size=0.15, val_size=0.15)
                trainer = ModelTrainer(random_state=42)
                bundle = trainer.train_and_select(train_df, val_df, calibration_method="sigmoid")
                trainer.save_bundle(bundle)
                st.session_state["bundle"] = bundle
                st.session_state["risks"] = predict_patient_risk(st.session_state["df"], bundle)
                st.success("Model trained, calibrated, and persisted successfully!")
                st.rerun()
        return False
    return True


# ==========================================
# PAGE 1: OVERVIEW DASHBOARD
# ==========================================
if page_selection == "📊 Overview":
    st.markdown("### Good morning, Care Team")
    st.markdown(
        f"<span style='color: {COLORS['secondary_text']}; font-size: 0.95rem;'>"
        "Identify high-risk post-discharge patients who may need proactive outreach. "
        "Weekly nurse outreach capacity is constrained to <strong>50 patients</strong>.</span>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if ensure_model_ready():
        df = st.session_state["df"]
        risks = st.session_state["risks"]

        # Summary KPIs
        total_records = len(df)
        readmitted_count = int(df["readmitted_30d"].sum()) if "readmitted_30d" in df.columns else 0
        readmission_rate = (readmitted_count / total_records) if total_records > 0 else 0.0
        high_risk_count = int(np.sum(risks >= RISK_THRESHOLDS["moderate"]))
        weekly_capacity = st.session_state["weekly_capacity"]

        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        with kpi_c1:
            render_metric_card("Total Discharges", f"{total_records:,}", "Active monitored cohort", COLORS["deep_navy"])
        with kpi_c2:
            render_metric_card("Observed Readmission Rate", f"{readmission_rate:.1%}", f"{readmitted_count:,} 30-day readmissions", COLORS["primary_teal"])
        with kpi_c3:
            render_metric_card("High / Very High Risk", f"{high_risk_count:,}", f"{(high_risk_count / total_records):.1%} of patient cohort", COLORS["warning_amber"])
        with kpi_c4:
            render_metric_card("Weekly Outreach Target", f"{weekly_capacity} pts", "Prioritized Monday Call List", COLORS["deep_teal"])

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

        # Charts Row
        col_c1, col_c2 = st.columns([6, 5])
        with col_c1:
            st.markdown(
                f"""
                <div class="cp-card">
                    <div class="cp-card-header">
                        <span>Readmission Risk Distribution</span>
                        <span style="font-size: 0.8rem; font-weight: 500; color: {COLORS['secondary_text']};">Calibrated Probabilities</span>
                    </div>
                """,
                unsafe_allow_html=True
            )
            # Histogram of risks
            hist_fig = px.histogram(
                x=risks,
                nbins=35,
                labels={"x": "Calibrated 30-Day Readmission Risk", "y": "Number of Patients"},
                color_discrete_sequence=[COLORS["primary_teal"]],
            )
            # Add vertical line for capacity cutoff (top 50)
            sorted_risks = np.sort(risks)[::-1]
            top_cutoff = sorted_risks[min(weekly_capacity - 1, len(sorted_risks) - 1)]
            hist_fig.add_vline(
                x=top_cutoff,
                line_dash="dash",
                line_color=COLORS["critical_red"],
                annotation_text=f"Top {weekly_capacity} Cutoff ({top_cutoff:.1%})",
                annotation_position="top right"
            )
            hist_fig.update_layout(get_plotly_theme(), height=320)
            st.plotly_chart(hist_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_c2:
            st.markdown(
                f"""
                <div class="cp-card">
                    <div class="cp-card-header">
                        <span>Patients by Risk Category</span>
                        <span style="font-size: 0.8rem; font-weight: 500; color: {COLORS['secondary_text']};">Clinical Stratification</span>
                    </div>
                """,
                unsafe_allow_html=True
            )
            categories = [get_risk_category(p) for p in risks]
            cat_counts = pd.Series(categories).value_counts().reindex(["Low", "Moderate", "High", "Very High"]).fillna(0)
            
            bar_colors = [COLORS["soft_mint"], COLORS["pale_blue"], COLORS["warning_amber"], COLORS["critical_red"]]
            donut_fig = go.Figure(data=[go.Pie(
                labels=cat_counts.index,
                values=cat_counts.values,
                hole=0.55,
                marker=dict(colors=[COLORS["deep_teal"], "#2563EB", COLORS["warning_amber"], COLORS["critical_red"]]),
                textinfo="label+percent",
            )])
            donut_fig.update_layout(get_plotly_theme(), height=320, showlegend=False)
            st.plotly_chart(donut_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Priority Attention Panel (Top 5 highest risk eligible patients)
        st.markdown(
            f"""
            <div class="cp-card">
                <div class="cp-card-header">
                    <span>Priority Attention: Highest-Risk Eligible Discharges</span>
                    <span style="font-size: 0.8rem; font-weight: 500; color: {COLORS['secondary_text']};">Top 5 of 50 Weekly Target</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        call_list = generate_weekly_call_list(df, risks, capacity=weekly_capacity)
        top_5 = call_list.head(5)

        for _, row in top_5.iterrows():
            st.markdown(
                f"""
                <div style="background-color: {COLORS['warm_off_white']}; border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <span style="font-size: 0.85rem; font-weight: 700; color: {COLORS['deep_navy']};">Rank #{row['priority_rank']} — Patient {row['patient_id']}</span>
                        <span style="margin-left: 10px;">{render_risk_pill(row['risk_category'])}</span>
                        <div style="font-size: 0.82rem; color: {COLORS['secondary_text']}; margin-top: 4px;">
                            Diagnosis: <strong>{row['primary_diagnosis']}</strong> | Age: {row['age']} | Prior Admissions (6m): {row['prior_admissions_6m']}
                        </div>
                    </div>
                    <div style="flex: 1; padding: 0 16px;">
                        <span style="font-size: 0.8rem; font-weight: 600; color: {COLORS['deep_teal']};">Recommended Action:</span>
                        <div style="font-size: 0.82rem; color: {COLORS['main_text']};">{row['recommended_action']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.4rem; font-weight: 700; color: {COLORS['critical_red']};">{row['calibrated_risk']:.1%}</div>
                        <div style="font-size: 0.72rem; color: {COLORS['secondary_text']};">Readmission Risk</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        col_b1, col_b2 = st.columns([3, 7])
        with col_b1:
            if st.button("📋 View Full Weekly Call List", type="primary"):
                st.session_state["current_page"] = "📋 Weekly Call List"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 2: PATIENT RISK EXPLORER
# ==========================================
elif page_selection == "🔍 Patient Risk Explorer":
    st.markdown("### Searchable Patient Cohort")
    st.markdown(
        f"<span style='color: {COLORS['secondary_text']}; font-size: 0.95rem;'>"
        "Filter and inspect individual patients across clinical, demographic, and risk attributes.</span>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if ensure_model_ready():
        df = st.session_state["df"].copy()
        risks = st.session_state["risks"]
        df["calibrated_risk"] = risks
        df["risk_category"] = [get_risk_category(p) for p in risks]

        # Filter bar
        with st.expander("Filter & Search Controls", expanded=True):
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            with f_col1:
                search_id = st.text_input("Search Patient ID", placeholder="e.g. PT-100234").strip()
            with f_col2:
                selected_risk_cats = st.multiselect("Risk Category", ["Low", "Moderate", "High", "Very High"], default=["High", "Very High"])
            with f_col3:
                all_diags = sorted(df["primary_diagnosis"].dropna().unique())
                selected_diags = st.multiselect("Primary Diagnosis", all_diags)
            with f_col4:
                selected_payers = st.multiselect("Payer Type", sorted(df["payer_type"].dropna().unique()))

        # Apply filters
        filtered_df = df.copy()
        if search_id:
            filtered_df = filtered_df[filtered_df["patient_id"].str.contains(search_id, case=False, na=False)]
        if selected_risk_cats:
            filtered_df = filtered_df[filtered_df["risk_category"].isin(selected_risk_cats)]
        if selected_diags:
            filtered_df = filtered_df[filtered_df["primary_diagnosis"].isin(selected_diags)]
        if selected_payers:
            filtered_df = filtered_df[filtered_df["payer_type"].isin(selected_payers)]

        # Sort by calibrated risk descending
        filtered_df = filtered_df.sort_values(by="calibrated_risk", ascending=False).reset_index(drop=True)
        filtered_df["priority_rank"] = np.arange(1, len(filtered_df) + 1)

        st.markdown(f"Found **{len(filtered_df):,}** patients matching filters.")

        # Pagination controls
        page_size = 20
        total_pages = max(1, (len(filtered_df) - 1) // page_size + 1)
        p_c1, p_c2 = st.columns([2, 8])
        with p_c1:
            current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))
        display_page_df = filtered_df.iloc[start_idx:end_idx]

        # Display table
        view_cols = [
            "priority_rank", "patient_id", "age", "sex", "primary_diagnosis",
            "calibrated_risk", "risk_category", "prior_admissions_6m", "medication_count", "followup_scheduled"
        ]
        
        # Format table for clean display
        show_table = display_page_df[view_cols].copy()
        show_table["calibrated_risk"] = show_table["calibrated_risk"].apply(lambda p: f"{p:.1%}")

        st.dataframe(
            show_table,
            column_config={
                "priority_rank": "Rank",
                "patient_id": "Patient ID",
                "age": "Age",
                "sex": "Sex",
                "primary_diagnosis": "Primary Diagnosis",
                "calibrated_risk": "Readmission Risk",
                "risk_category": "Risk Tier",
                "prior_admissions_6m": "Prior 6m",
                "medication_count": "Meds",
                "followup_scheduled": "Follow-up Appt",
            },
            hide_index=True,
            use_container_width=True,
        )

        # Quick patient selection for deep-dive
        st.markdown("#### Patient Deep-Dive")
        select_col1, select_col2 = st.columns([3, 7])
        with select_col1:
            patient_ids_on_page = display_page_df["patient_id"].tolist()
            chosen_pid = st.selectbox("Select Patient to Inspect", patient_ids_on_page if patient_ids_on_page else ["None"])
            if st.button("Open Patient Detail Screen", type="primary") and chosen_pid != "None":
                st.session_state["selected_patient_id"] = chosen_pid
                st.rerun()


# ==========================================
# PAGE 3: PATIENT DETAIL SCREEN
# ==========================================
elif page_selection == "👤 Patient Detail":
    if ensure_model_ready():
        df = st.session_state["df"]
        risks = st.session_state["risks"]
        
        # If no patient selected yet, pick top ranked patient
        if st.session_state["selected_patient_id"] is None:
            top_call = generate_weekly_call_list(df, risks, capacity=1)
            st.session_state["selected_patient_id"] = top_call.iloc[0]["patient_id"]

        patient_id = st.session_state["selected_patient_id"]
        pt_matches = df[df["patient_id"] == patient_id]

        if pt_matches.empty:
            st.error(f"Patient ID '{patient_id}' not found.")
        else:
            pt_record = pt_matches.iloc[0]
            pt_idx = pt_matches.index[0]
            pt_risk = risks[pt_idx]
            pt_category = get_risk_category(pt_risk)

            # Patient header banner
            st.markdown(
                f"""
                <div class="cp-card" style="border-left: 6px solid {COLORS['primary_teal']};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 0.8rem; color: {COLORS['secondary_text']}; text-transform: uppercase; font-weight: 600;">Patient Profile</span>
                            <h2 style="margin: 0; color: {COLORS['deep_navy']};">{patient_id} <span style="font-size: 1rem; color: {COLORS['secondary_text']};">({pt_record['sex']}, {pt_record['age']} yrs)</span></h2>
                            <div style="font-size: 0.85rem; color: {COLORS['secondary_text']}; margin-top: 4px;">
                                Encounter ID: <strong>{pt_record['encounter_id']}</strong> | Payer: <strong>{pt_record['payer_type']}</strong> | Discharge Date: <strong>{pt_record.get('discharge_date', 'N/A')}</strong>
                            </div>
                        </div>
                        <div style="text-align: right; background-color: {COLORS['warm_off_white']}; padding: 12px 20px; border-radius: 10px; border: 1px solid {COLORS['border']};">
                            <div style="font-size: 2.2rem; font-weight: 700; color: {COLORS['critical_red'] if pt_risk >= 0.3 else COLORS['primary_teal']}; line-height: 1;">
                                {pt_risk:.1%}
                            </div>
                            <div style="margin-top: 4px;">{render_risk_pill(pt_category)}</div>
                            <div style="font-size: 0.72rem; color: {COLORS['secondary_text']}; margin-top: 4px;">Calibrated 30-Day Risk</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Clinical Features Grid
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Primary Diagnosis", pt_record["primary_diagnosis"], f"Length of Stay: {pt_record['length_of_stay']} days", COLORS["midnight_blue"])
            with c2:
                render_metric_card("Prior Admissions", f"{pt_record['prior_admissions_6m']} (6m)", f"{pt_record['prior_admissions']} total in 12m", COLORS["deep_navy"])
            with c3:
                render_metric_card("Active Medications", f"{pt_record['medication_count']} meds", "Discharge medication list", COLORS["warning_amber"] if pt_record['medication_count'] >= 10 else COLORS["deep_teal"])
            with c4:
                followup_disp = pt_record["followup_scheduled"] if pd.notna(pt_record["followup_scheduled"]) else "Unrecorded"
                render_metric_card("Follow-up Scheduled", str(followup_disp), f"Discharge: {pt_record['discharge_disposition']}", COLORS["primary_teal"] if followup_disp == "Yes" else COLORS["critical_red"])

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

            # Two columns: Explainable AI & Intervention Pathways
            xai_col, rec_col = st.columns([6, 5])

            with xai_col:
                st.markdown(
                    f"""
                    <div class="cp-card">
                        <div class="cp-card-header">
                            <span>Explainable AI: Key Risk Factor Contributions</span>
                            <span style="font-size: 0.8rem; font-weight: 500; color: {COLORS['secondary_text']};">Local Attributions</span>
                        </div>
                    """,
                    unsafe_allow_html=True
                )
                explainer = ClinicalExplainer(st.session_state["bundle"])
                explanation = explainer.explain_patient(pt_record, top_k=6)
                
                # Plotly horizontal waterfall / bar of top factors
                factors = explanation["top_factors"]
                feat_names = [f["feature"].replace("num__", "").replace("cat__", "").replace("_", " ").title() for f in factors][::-1]
                feat_impacts = [f["importance"] for f in factors][::-1]
                colors_bar = [COLORS["critical_red"] if imp > 0 else COLORS["primary_teal"] for imp in feat_impacts]

                bar_fig = go.Figure(go.Bar(
                    x=feat_impacts,
                    y=feat_names,
                    orientation="h",
                    marker_color=colors_bar,
                    text=[f"{imp:+.3f}" for imp in feat_impacts],
                    textposition="auto",
                ))
                bar_fig.update_layout(
                    get_plotly_theme(),
                    height=280,
                    xaxis_title="Contribution to Readmission Log-Odds (+ increases, - decreases)",
                    margin=dict(l=150, r=20, t=10, b=30),
                )
                st.plotly_chart(bar_fig, use_container_width=True)

                st.markdown("##### Clinical Factor Narratives")
                for item in explanation["clinical_statements"]:
                    badge_color = COLORS["critical_red"] if item["direction"] == "increases" else COLORS["deep_teal"]
                    st.markdown(
                        f"""
                        <div style="padding: 8px 12px; margin-bottom: 8px; border-left: 3px solid {badge_color}; background-color: {COLORS['warm_off_white']}; border-radius: 4px;">
                            <strong>{item['title']}</strong><br/>
                            <span style="font-size: 0.82rem; color: {COLORS['main_text']};">{item['text']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("</div>", unsafe_allow_html=True)

            with rec_col:
                st.markdown(
                    f"""
                    <div class="cp-card">
                        <div class="cp-card-header">
                            <span>Recommended Care Pathways</span>
                            <span style="font-size: 0.8rem; font-weight: 500; color: {COLORS['secondary_text']};">Intervention Engine</span>
                        </div>
                    """,
                    unsafe_allow_html=True
                )
                recs = recommend_interventions(pt_record)
                if not recs:
                    st.info("No urgent high-intensity interventions triggered. Standard routine follow-up call indicated.")
                else:
                    for rec in recs:
                        p_color = COLORS["critical_red"] if rec.priority == "High" else COLORS["warning_amber"]
                        st.markdown(
                            f"""
                            <div style="background-color: {COLORS['warm_off_white']}; border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                    <span style="font-size: 0.9rem; font-weight: 700; color: {COLORS['deep_navy']};">{rec.title}</span>
                                    <span style="font-size: 0.75rem; font-weight: 600; color: {p_color}; background-color: #FFFFFF; padding: 2px 8px; border-radius: 12px; border: 1px solid {p_color};">{rec.priority} Priority</span>
                                </div>
                                <div style="font-size: 0.8rem; color: {COLORS['secondary_text']}; margin-bottom: 4px;">
                                    <strong>Evidence:</strong> {rec.supporting_evidence}
                                </div>
                                <div style="font-size: 0.8rem; color: {COLORS['main_text']}; margin-bottom: 6px;">
                                    <strong>Rationale:</strong> {rec.reason}
                                </div>
                                <div style="font-size: 0.74rem; color: {COLORS['secondary_text']}; display: flex; justify-content: space-between;">
                                    <span>Assumed RRR: <strong>{rec.relative_risk_reduction:.0%}</strong></span>
                                    <span>{'⚠️ Clinician Review Required' if rec.clinician_review_required else 'Care Coordinator Action'}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.markdown(
                    f"""
                    <div class="cp-demo-banner" style="margin-top: 14px;">
                        <strong>Clinical Decision Support Notice:</strong> Recommendations are generated by an interpretable rules engine and must be paired with clinical evaluation. Do not assume causal guarantees.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 4: WEEKLY CALL LIST (CORE FEATURE)
# ==========================================
elif page_selection == "📋 Weekly Call List":
    st.markdown("### Monday Morning Prioritized Outreach Queue")
    st.markdown(
        f"<span style='color: {COLORS['secondary_text']}; font-size: 0.95rem;'>"
        "Care-coordination team has a finite capacity of <strong>50 patients per week</strong>. "
        "Patients are ranked by calibrated readmission risk with patient-level deduplication and supervisor override auditing.</span>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if ensure_model_ready():
        df = st.session_state["df"]
        risks = st.session_state["risks"]
        capacity = st.session_state["weekly_capacity"]

        call_list = generate_weekly_call_list(df, risks, capacity=capacity)

        # Capacity & Summary Bar
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("Outreach Slots Filled", f"{len(call_list)} / {capacity}", "Weekly contact capacity limit", COLORS["deep_navy"])
        with m2:
            avg_risk = call_list["calibrated_risk"].mean()
            render_metric_card("Cohort Mean Risk", f"{avg_risk:.1%}", "Average probability of readmission", COLORS["critical_red"])
        with m3:
            pending_count = sum(call_list["contact_status"] == "Pending Outreach")
            render_metric_card("Pending Calls", f"{pending_count}", "Awaiting nurse outreach", COLORS["warning_amber"])
        with m4:
            contacted_count = len(call_list) - pending_count
            render_metric_card("Contact Attempts", f"{contacted_count}", "Completed or in-progress", COLORS["primary_teal"])

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Action Buttons: Export CSV & Print View
        col_act1, col_act2, col_act3 = st.columns([3, 3, 4])
        with col_act1:
            csv_data = call_list.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Monday Call List (CSV)",
                data=csv_data,
                file_name=f"carepulse_call_list_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary",
            )
        with col_act2:
            show_print_mode = st.checkbox("🖨️ Clean Printable Layout", value=False)

        # Table Display
        st.markdown("#### Prioritized Patient Queue")
        
        display_cols = [
            "priority_rank", "patient_id", "calibrated_risk", "risk_category",
            "primary_diagnosis", "age", "recommended_action", "contact_status"
        ]
        
        table_format = call_list[display_cols].copy()
        table_format["calibrated_risk"] = table_format["calibrated_risk"].apply(lambda p: f"{p:.1%}")

        st.dataframe(
            table_format,
            column_config={
                "priority_rank": "Rank",
                "patient_id": "Patient ID",
                "calibrated_risk": "Risk",
                "risk_category": "Tier",
                "primary_diagnosis": "Diagnosis",
                "age": "Age",
                "recommended_action": "Recommended Action",
                "contact_status": "Contact Status",
            },
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Tabbed workflows: Log Contact Attempt & Supervisor Overrides
        tab_log, tab_override, tab_audit = st.tabs(["📞 Log Contact Outcome", "⚖️ Supervisor Override", "📜 Override Audit Trail"])

        with tab_log:
            st.markdown("##### Record Outreach Call Outcome")
            l_c1, l_c2, l_c3 = st.columns(3)
            with l_c1:
                call_pids = call_list["patient_id"].tolist()
                log_pid = st.selectbox("Select Patient", call_pids)
            with l_c2:
                outcome_options = [
                    "Contacted - Follow-up Booked",
                    "Contacted - Medication Reconciled",
                    "Contacted - Home Visit Scheduled",
                    "Left Voicemail",
                    "Unable to Reach (No Answer)",
                    "Patient Refused Outreach",
                ]
                log_status = st.selectbox("Outreach Outcome", outcome_options)
            with l_c3:
                log_notes = st.text_input("Care Manager Notes", placeholder="e.g. Scheduled Dr. Adams for next Tuesday")

            if st.button("Save Contact Outcome", type="primary"):
                record_contact_outcome(log_pid, log_status, log_notes)
                st.success(f"Outcome saved for patient {log_pid}!")
                st.rerun()

        with tab_override:
            st.markdown("##### Clinical Supervisor Override")
            st.markdown(
                f"<span style='font-size: 0.85rem; color: {COLORS['secondary_text']};'>"
                "Supervisors can manually force-include or force-exclude a patient with documented rationale. "
                "All overrides are recorded in the permanent audit trail.</span>",
                unsafe_allow_html=True
            )
            o_c1, o_c2, o_c3 = st.columns(3)
            with o_c1:
                override_pid = st.text_input("Patient ID to Override", placeholder="e.g. PT-100452")
            with o_c2:
                override_action = st.selectbox("Action", ["force_include", "force_exclude"], format_func=lambda x: "Force Include in Top-50" if x == "force_include" else "Force Exclude from Call List")
            with o_c3:
                override_reason = st.text_input("Clinical Justification", placeholder="e.g. Patient referred by attending physician")

            if st.button("Submit Authorized Override"):
                if override_pid and override_reason:
                    record_supervisor_override(override_pid.strip(), override_action, override_reason.strip())
                    st.success(f"Override registered for {override_pid}!")
                    st.rerun()
                else:
                    st.error("Please supply both a Patient ID and clinical justification.")

        with tab_audit:
            st.markdown("##### Override Audit Trail")
            history = get_override_history()
            if not history:
                st.info("No supervisor overrides currently logged.")
            else:
                st.dataframe(pd.DataFrame(history), use_container_width=True)


# ==========================================
# PAGE 5: INTERVENTION PLANNER
# ==========================================
elif page_selection == "🎯 Intervention Planner":
    st.markdown("### Care Pathway Capacity & Resource Allocation")
    st.markdown(
        f"<span style='color: {COLORS['secondary_text']}; font-size: 0.95rem;'>"
        "Review aggregated intervention pathways recommended for the prioritized outreach cohort. "
        "Align nursing, pharmacy, and clinic staff with anticipated patient needs.</span>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if ensure_model_ready():
        df = st.session_state["df"]
        risks = st.session_state["risks"]
        capacity = st.session_state["weekly_capacity"]
        call_list = generate_weekly_call_list(df, risks, capacity=capacity)

        # Aggregate counts by primary recommended intervention
        counts = call_list["recommended_action"].value_counts().reset_index()
        counts.columns = ["Intervention", "Count"]

        c_col1, c_col2 = st.columns([5, 5])
        with c_col1:
            st.markdown(
                f"""
                <div class="cp-card">
                    <div class="cp-card-header">
                        <span>Staff Resource Allocation (Top {capacity} Cohort)</span>
                    </div>
                """,
                unsafe_allow_html=True
            )
            fig_bar = px.bar(
                counts,
                x="Count",
                y="Intervention",
                orientation="h",
                color_discrete_sequence=[COLORS["deep_teal"]],
                text="Count",
            )
            fig_bar.update_layout(get_plotly_theme(), height=300, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_col2:
            st.markdown(
                f"""
                <div class="cp-card">
                    <div class="cp-card-header">
                        <span>Care Pathway Breakdown</span>
                    </div>
                """,
                unsafe_allow_html=True
            )
            for _, row in counts.iterrows():
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; padding: 10px 14px; background-color: {COLORS['warm_off_white']}; border-radius: 6px; margin-bottom: 8px; border: 1px solid {COLORS['border']};">
                        <span style="font-size: 0.85rem; font-weight: 600; color: {COLORS['deep_navy']};">{row['Intervention']}</span>
                        <span style="font-size: 0.85rem; font-weight: 700; color: {COLORS['primary_teal']};">{row['Count']} Patients ({(row['Count'] / len(call_list)):.0%})</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Patient Intervention List
        st.markdown(
            f"""
            <div class="cp-card">
                <div class="cp-card-header">
                    <span>Patient Intervention Assignment</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        st.dataframe(
            call_list[["priority_rank", "patient_id", "primary_diagnosis", "calibrated_risk", "recommended_action", "primary_risk_reason"]],
            column_config={
                "priority_rank": "Rank",
                "patient_id": "Patient ID",
                "primary_diagnosis": "Diagnosis",
                "calibrated_risk": "Calibrated Risk",
                "recommended_action": "Primary Pathway",
                "primary_risk_reason": "Clinical Evidence",
            },
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 6: MODEL PERFORMANCE & EVALUATION
# ==========================================
elif page_selection == "📈 Model Performance":
    st.markdown("### Algorithmic Evaluation & Calibration")
    st.markdown(
        f"<span style='color: {COLORS['secondary_text']}; font-size: 0.95rem;'>"
        "Evaluated on the patient-grouped held-out test split. "
        "Strictly computes AUC-PR (Average Precision), Recall@50, Precision@50, and Expected Calibration Error (ECE).</span>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if ensure_model_ready():
        bundle = st.session_state["bundle"]
        df = st.session_state["df"]

        # Prepare held-out test cohort
        _, _, test_df = patient_grouped_split(df, test_size=0.15, val_size=0.15, random_state=42)
        test_probs = predict_patient_risk(test_df, bundle)
        y_test = test_df["readmitted_30d"].values

        eval_metrics = compute_evaluation_metrics(
            y_true=y_test,
            y_prob=test_probs,
            threshold=0.30,
            capacity=st.session_state["weekly_capacity"]
        )

        # Metric cards
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            render_metric_card("AUC-PR (Avg Precision)", f"{eval_metrics['auc_pr']:.4f}", f"Prevalence baseline: {eval_metrics['prevalence']:.1%}", COLORS["deep_navy"])
        with e2:
            render_metric_card("Recall@50", f"{eval_metrics['recall_at_capacity']:.1%}", f"{eval_metrics['top_k_true_positives']} of {eval_metrics['n_positives']} readmissions", COLORS["primary_teal"])
        with e3:
            render_metric_card("Precision@50", f"{eval_metrics['precision_at_capacity']:.1%}", f"{eval_metrics['top_k_true_positives']} of {eval_metrics['capacity_k']} in top tier", COLORS["deep_teal"])
        with e4:
            render_metric_card("Calibration Error (ECE)", f"{eval_metrics['expected_calibration_error']:.4f}", f"Brier Score: {eval_metrics['brier_score']:.4f}", COLORS["warning_amber"])

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Charts: PR Curve & Reliability Calibration Curve
        pr_col, cal_col = st.columns(2)

        with pr_col:
            st.markdown(
                f"""
                <div class="cp-card">
                    <div class="cp-card-header">
                        <span>Precision-Recall Curve</span>
                        <span style="font-size: 0.8rem; font-weight: 500; color: {COLORS['secondary_text']};">AUC-PR = {eval_metrics['auc_pr']:.4f}</span>
                    </div>
                """,
                unsafe_allow_html=True
            )
            pr_data = eval_metrics["pr_curve_data"]
            pr_fig = go.Figure()
            pr_fig.add_trace(go.Scatter(
                x=pr_data["recall"],
                y=pr_data["precision"],
                mode="lines",
                name="CarePulse Classifier",
                line=dict(color=COLORS["primary_teal"], width=2.5),
            ))
            # Baseline prevalence horizontal line
            pr_fig.add_hline(
                y=eval_metrics["prevalence"],
                line_dash="dot",
                line_color=COLORS["secondary_text"],
                annotation_text=f"Baseline ({eval_metrics['prevalence']:.1%})"
            )
            pr_fig.update_layout(
                get_plotly_theme(),
                height=300,
                xaxis_title="Recall",
                yaxis_title="Precision",
                yaxis=dict(range=[0, 1]),
                xaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(pr_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with cal_col:
            st.markdown(
                f"""
                <div class="cp-card">
                    <div class="cp-card-header">
                        <span>Reliability Diagram (Calibration)</span>
                        <span style="font-size: 0.8rem; font-weight: 500; color: {COLORS['secondary_text']};">ECE = {eval_metrics['expected_calibration_error']:.4f}</span>
                    </div>
                """,
                unsafe_allow_html=True
            )
            bins_df = pd.DataFrame(eval_metrics["calibration_bins"])
            bins_filtered = bins_df[bins_df["count"] > 0]
            
            cal_fig = go.Figure()
            # Perfectly calibrated diagonal
            cal_fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Perfect Calibration",
                line=dict(color=COLORS["secondary_text"], dash="dash")
            ))
            cal_fig.add_trace(go.Scatter(
                x=bins_filtered["avg_confidence"],
                y=bins_filtered["observed_accuracy"],
                mode="lines+markers",
                name="Calibrated Model",
                line=dict(color=COLORS["deep_navy"], width=2),
                marker=dict(size=8, color=COLORS["critical_red"])
            ))
            cal_fig.update_layout(
                get_plotly_theme(),
                height=300,
                xaxis_title="Mean Predicted Risk",
                yaxis_title="Observed Fraction Positive",
                yaxis=dict(range=[0, 1]),
                xaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(cal_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Candidate Model Comparison Table
        st.markdown(
            f"""
            <div class="cp-card">
                <div class="cp-card-header">
                    <span>Candidate Model Selection Benchmarks (Validation Split)</span>
                    <span style="font-size: 0.8rem; font-weight: 500; color: {COLORS['secondary_text']};">Selected: {bundle.get('model_name', 'Active Model')}</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        candidates = bundle.get("candidate_results", {})
        if candidates:
            cand_df = pd.DataFrame(candidates).T.reset_index()
            cand_df.columns = ["Architecture", "Val AUC-PR", "Val ROC-AUC", "Val Brier Score", "Fit Time (s)"]
            st.dataframe(cand_df, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 7: FAIRNESS & BIAS AUDIT
# ==========================================
elif page_selection == "⚖️ Fairness & Bias Audit":
    st.markdown("### Algorithmic Equity & Subgroup Parity Audit")
    st.markdown(
        f"<span style='color: {COLORS['secondary_text']}; font-size: 0.95rem;'>"
        "Quantifies predictive performance across age brackets, sex, and insurance payer cohorts. "
        "Discloses demographic representation in the top-50 prioritized call list and actionable bias mitigation recommendations.</span>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if ensure_model_ready():
        df = st.session_state["df"]
        risks = st.session_state["risks"]
        capacity = st.session_state["weekly_capacity"]

        call_list = generate_weekly_call_list(df, risks, capacity=capacity)
        top_50_pids = call_list["patient_id"].tolist()

        audit_results = audit_fairness_subgroups(df, risks, top_50_patient_ids=top_50_pids)
        subgroup_audits = audit_results["subgroup_audits"]
        mitigation = audit_results["mitigation_report"]

        # Subgroup Tables
        for cat_name, records in subgroup_audits.items():
            st.markdown(
                f"""
                <div class="cp-card">
                    <div class="cp-card-header">
                        <span>Demographic Category: {cat_name}</span>
                    </div>
                """,
                unsafe_allow_html=True
            )
            rec_df = pd.DataFrame(records)
            st.dataframe(
                rec_df,
                column_config={
                    "subgroup": "Cohort",
                    "sample_size": "Sample Size (N)",
                    "population_share_pct": "% of Total Population",
                    "top50_share_pct": "% of Top-50 Call List",
                    "prevalence_pct": "Readmission Rate",
                    "auc_pr": "AUC-PR",
                    "ece": "Calibration Error (ECE)",
                    "brier_score": "Brier Score",
                    "missing_lab_pct": "Lab Missingness %",
                    "status": "Audit Status",
                },
                hide_index=True,
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Mitigation Report
        st.markdown(
            f"""
            <div class="cp-card">
                <div class="cp-card-header">
                    <span>Clinical Bias Mitigation & Data Quality Findings</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("##### Disparity Observations:")
        for disp in mitigation["disparities_detected"]:
            st.markdown(f"- ⚠️ {disp}")

        st.markdown("##### Recommended Mitigations:")
        for mit in mitigation["recommended_mitigations"]:
            st.markdown(f"- ✅ {mit}")

        st.markdown(
            f"""
            <div class="cp-demo-banner" style="margin-top: 14px;">
                <strong>Regulatory & Ethics Guidance:</strong> Fairness cannot be established by viewing a single parity metric in isolation. Clinical care teams must retain human review for patient prioritization.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 8: IMPACT SIMULATOR
# ==========================================
elif page_selection == "🔮 Impact Simulator":
    st.markdown("### Interactive Outreach Capacity & Impact Simulator")
    st.markdown(
        f"<span style='color: {COLORS['secondary_text']}; font-size: 0.95rem;'>"
        "Explore 'What could better follow-up achieve?' under configurable outreach capacity and relative risk reduction (RRR) assumptions.</span>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if ensure_model_ready():
        df = st.session_state["df"]
        risks = st.session_state["risks"]

        # Interactive controls
        with st.expander("Simulation Parameters", expanded=True):
            sim_c1, sim_c2, sim_c3 = st.columns(3)
            with sim_c1:
                sim_capacity = st.slider("Weekly Contact Capacity", min_value=20, max_value=200, value=50, step=10)
            with sim_c2:
                sim_adherence = st.slider("Patient Outreach Adherence", min_value=0.40, max_value=1.0, value=0.75, step=0.05, format="%.0f%%")
            with sim_c3:
                sim_mode = st.radio("Relative Risk Reduction (RRR) Model", ["Intervention-Specific RRR", "Uniform RRR Override"])
                uniform_rrr = None
                if sim_mode == "Uniform RRR Override":
                    uniform_rrr = st.slider("Uniform Assumed RRR", min_value=0.05, max_value=0.50, value=0.25, step=0.05, format="%.0f%%")

        # Run simulation
        selected_cohort = generate_weekly_call_list(df, risks, capacity=sim_capacity)
        sim_results = simulate_outreach_impact(
            selected_patients_df=selected_cohort,
            calibrated_risks=selected_cohort["calibrated_risk"].values,
            global_rrr_override=uniform_rrr,
            adherence_rate=sim_adherence,
            use_intervention_specific_rrr=(sim_mode == "Intervention-Specific RRR")
        )

        # Simulation KPIs
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            render_metric_card("Selected Patient Cohort", f"{sim_results['n_patients']} pts", "High-risk outreach queue", COLORS["deep_navy"])
        with s2:
            render_metric_card("Expected Readmissions", f"{sim_results['expected_without_intervention']:.1f}", "Without targeted intervention", COLORS["midnight_blue"])
        with s3:
            render_metric_card("Estimated Readmissions Averted", f"{sim_results['estimated_averted_readmissions']:.1f}", f"{sim_results['relative_cohort_reduction_pct']:.1f}% relative reduction", COLORS["primary_teal"])
        with s4:
            render_metric_card("Net Expected Readmissions", f"{sim_results['net_readmissions_after_intervention']:.1f}", "Projected with proactive support", COLORS["deep_teal"])

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Sensitivity Matrix
        st.markdown(
            f"""
            <div class="cp-card">
                <div class="cp-card-header">
                    <span>Sensitivity Analysis: Readmissions Averted Across Parameter Variations</span>
                    <span style="font-size: 0.8rem; font-weight: 500; color: {COLORS['secondary_text']};">Target Cohort = {sim_capacity} patients</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        sens_df = pd.DataFrame(sim_results["sensitivity_table"])
        st.dataframe(
            sens_df,
            column_config={
                "assumed_rrr_pct": "Assumed RRR (%)",
                "adh_50pct": "50% Adherence",
                "adh_65pct": "65% Adherence",
                "adh_80pct": "80% Adherence",
                "adh_95pct": "95% Adherence",
            },
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("##### Simulation Modeling Assumptions:")
        for asm in sim_results["assumptions"]:
            st.markdown(f"- ℹ️ {asm}")

        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 9: DATA MANAGEMENT
# ==========================================
elif page_selection == "🗄️ Data Management":
    st.markdown("### Clinical Data Management & Ingestion Audit")
    st.markdown(
        f"<span style='color: {COLORS['secondary_text']}; font-size: 0.95rem;'>"
        "Inspect ingested dataset schema, laboratory missingness, duplicate encounter detection, and patient longitudinal records.</span>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    df = st.session_state["df"]
    audit = st.session_state["audit"]

    # Data Quality Cards
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        render_metric_card("Total Discharge Records", f"{audit.total_records:,}", "Analyzed encounter volume", COLORS["deep_navy"])
    with q2:
        render_metric_card("Unique Patients", f"{audit.unique_patients:,}", f"{audit.total_records - audit.unique_patients:,} repeat encounters", COLORS["midnight_blue"])
    with q3:
        render_metric_card("Duplicate Encounters", f"{audit.duplicate_encounters}", "Deduplicated by encounter ID", COLORS["primary_teal"] if audit.duplicate_encounters == 0 else COLORS["warning_amber"])
    with q4:
        readm_rate = audit.target_distribution.get("readmission_rate", 0.0)
        render_metric_card("Cohort Readmission Rate", f"{readm_rate:.1%}", "Observed 30-day outcomes", COLORS["deep_teal"])

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Missingness Table
    col_d1, col_d2 = st.columns([5, 5])
    with col_d1:
        st.markdown(
            f"""
            <div class="cp-card">
                <div class="cp-card-header">
                    <span>Laboratory & Feature Missingness Audit</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        miss_df = pd.DataFrame({
            "Feature": list(audit.missing_counts.keys()),
            "Missing Count": list(audit.missing_counts.values()),
            "Missing %": list(audit.missing_percentages.values()),
        }).sort_values(by="Missing %", ascending=False).reset_index(drop=True)
        st.dataframe(miss_df, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_d2:
        st.markdown(
            f"""
            <div class="cp-card">
                <div class="cp-card-header">
                    <span>Data Actions & Synthetic Generator</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("Regenerate synthetic 25,000 hospital discharge records with controlled seeds and clinical disease interactions.")
        if st.button("🔄 Regenerate 25,000 Synthetic Discharge Records", type="secondary"):
            with st.spinner("Generating 25,000 synthetic patient records..."):
                new_df = generate_synthetic_cohort(num_records=25000, output_path=str(DEFAULT_DATA_PATH))
                clean_new = clean_and_deduplicate(new_df)
                st.session_state["df"] = clean_new
                st.session_state["audit"] = audit_dataset(clean_new)
                # Invalidate cached model risks
                if st.session_state["bundle"] is not None:
                    st.session_state["risks"] = predict_patient_risk(clean_new, st.session_state["bundle"])
                st.success("New 25,000 record synthetic dataset generated!")
                st.rerun()

        # CSV uploader
        st.markdown("---")
        st.markdown("##### Upload Custom Clinical CSV Dataset")
        uploaded_file = st.file_uploader("Select discharge records CSV", type=["csv"])
        if uploaded_file is not None:
            user_df = pd.read_csv(uploaded_file)
            user_clean = clean_and_deduplicate(user_df)
            user_audit = audit_dataset(user_clean)
            st.session_state["df"] = user_clean
            st.session_state["audit"] = user_audit
            if st.session_state["bundle"] is not None:
                st.session_state["risks"] = predict_patient_risk(user_clean, st.session_state["bundle"])
            st.success(f"Loaded {len(user_clean)} records from custom dataset!")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 10: SETTINGS & ABOUT
# ==========================================
elif page_selection == "⚙️ Settings & About":
    st.markdown("### Platform Settings & Clinical Disclaimers")
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    s_c1, s_c2 = st.columns(2)

    with s_c1:
        st.markdown(
            f"""
            <div class="cp-card">
                <div class="cp-card-header">
                    <span>Operational Configuration</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        new_capacity = st.number_input("Weekly Nurse Outreach Capacity", min_value=10, max_value=250, value=st.session_state["weekly_capacity"], step=5)
        if new_capacity != st.session_state["weekly_capacity"]:
            st.session_state["weekly_capacity"] = new_capacity
            st.success(f"Weekly capacity updated to {new_capacity} patients!")

        st.markdown("##### Model Retraining:")
        calib_choice = st.selectbox("Probability Calibration Method", ["sigmoid", "isotonic"], format_func=lambda x: "Sigmoid (Platt Scaling)" if x == "sigmoid" else "Isotonic Regression")
        
        if st.button("🚀 Retrain Model on Current Dataset", type="primary"):
            with st.spinner("Retraining candidate models (Logistic Regression, Random Forest, Gradient Boosting)..."):
                train_df, val_df, _ = patient_grouped_split(st.session_state["df"], test_size=0.15, val_size=0.15)
                trainer = ModelTrainer(random_state=42)
                bundle = trainer.train_and_select(train_df, val_df, calibration_method=calib_choice)
                trainer.save_bundle(bundle)
                st.session_state["bundle"] = bundle
                st.session_state["risks"] = predict_patient_risk(st.session_state["df"], bundle)
                st.success(f"Model retrained and calibrated via {calib_choice}!")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with s_c2:
        st.markdown(
            f"""
            <div class="cp-card">
                <div class="cp-card-header">
                    <span>About CarePulse AI</span>
                </div>
                <p><strong>CarePulse AI</strong> is a healthcare decision-support platform designed for hospital care-coordination teams.</p>
                <p><strong>Core Capabilities:</strong></p>
                <ul>
                    <li>Machine learning calibrated 30-day post-discharge readmission risk prediction.</li>
                    <li>Explainable AI factor decomposition with plain-language clinical insights.</li>
                    <li>Weekly Top-50 outreach prioritization queue with supervisor audit trails.</li>
                    <li>Fairness and algorithmic bias audit across age, sex, and payer cohorts.</li>
                    <li>Impact and capacity simulator for care-coordination planning.</li>
                </ul>
                <div class="cp-demo-banner">
                    <strong>Medical Disclaimer:</strong> CarePulse AI is an academic prototype for demonstration purposes. It does not provide direct medical advice, prescribe clinical regimens, or replace the judgment of licensed physicians and care coordinators.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
