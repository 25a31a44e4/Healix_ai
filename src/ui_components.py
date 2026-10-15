"""UI Components, Styling, and Plotly Theme Engine for CarePulse AI.

Implements the clinical intelligence aesthetic:
- Deep Navy (#142D4E), Midnight Blue (#1B3B5F), Primary Teal (#16A6A1)
- Soft Mint (#DDF5EC), Pale Blue (#EAF3FC), Warm Off-White (#F7F9FC)
- Responsive metric cards, risk pills, top bar, and consistent Plotly styling
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from src.config import COLORS


def inject_custom_css():
    """Injects cohesive clinical SaaS styling into Streamlit."""
    custom_css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: {COLORS['main_text']};
            background-color: {COLORS['warm_off_white']};
        }}

        /* App Background */
        .stApp {{
            background-color: {COLORS['warm_off_white']};
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: {COLORS['deep_navy']} !important;
            border-right: 1px solid {COLORS['midnight_blue']};
        }}
        
        section[data-testid="stSidebar"] * {{
            color: #E2E8F0 !important;
        }}
        
        section[data-testid="stSidebar"] .stRadio label {{
            color: #CBD5E1 !important;
            font-size: 0.95rem;
            padding: 8px 12px;
            border-radius: 8px;
            transition: all 0.2s ease;
        }}

        /* Top Bar Container */
        .cp-top-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 24px;
            background-color: {COLORS['pure_white']};
            border-radius: 12px;
            border: 1px solid {COLORS['border']};
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }}

        .cp-breadcrumb {{
            font-size: 0.85rem;
            color: {COLORS['secondary_text']};
            font-weight: 500;
        }}

        .cp-page-title {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {COLORS['deep_navy']};
            margin: 0;
        }}

        /* Clinical Metric Cards */
        .cp-metric-card {{
            background-color: {COLORS['pure_white']};
            border-radius: 12px;
            border: 1px solid {COLORS['border']};
            padding: 20px 24px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
            border-top: 4px solid {COLORS['primary_teal']};
            height: 100%;
        }}

        .cp-metric-label {{
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {COLORS['secondary_text']};
            margin-bottom: 6px;
        }}

        .cp-metric-value {{
            font-size: 2.1rem;
            font-weight: 700;
            color: {COLORS['deep_navy']};
            line-height: 1.1;
        }}

        .cp-metric-subtext {{
            font-size: 0.82rem;
            color: {COLORS['secondary_text']};
            margin-top: 6px;
        }}

        /* Risk Badge Pills */
        .cp-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            text-align: center;
        }}

        .cp-badge-low {{
            background-color: {COLORS['risk_low_bg']};
            color: {COLORS['risk_low_text']};
            border: 1px solid #C4EEDC;
        }}

        .cp-badge-mod {{
            background-color: {COLORS['risk_mod_bg']};
            color: {COLORS['risk_mod_text']};
            border: 1px solid #D0E4FA;
        }}

        .cp-badge-high {{
            background-color: {COLORS['risk_high_bg']};
            color: {COLORS['risk_high_text']};
            border: 1px solid #FBE1A8;
        }}

        .cp-badge-vhigh {{
            background-color: {COLORS['risk_vhigh_bg']};
            color: {COLORS['risk_vhigh_text']};
            border: 1px solid #F8CCD1;
        }}

        /* Content Container Cards */
        .cp-card {{
            background-color: {COLORS['pure_white']};
            border-radius: 12px;
            border: 1px solid {COLORS['border']};
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        }}

        .cp-card-header {{
            font-size: 1.15rem;
            font-weight: 700;
            color: {COLORS['deep_navy']};
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        /* Demo Notice Banner */
        .cp-demo-banner {{
            background-color: {COLORS['pale_blue']};
            border-left: 4px solid {COLORS['primary_teal']};
            padding: 10px 16px;
            border-radius: 6px;
            font-size: 0.82rem;
            color: {COLORS['midnight_blue']};
            margin-bottom: 20px;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = "", accent_color: str = COLORS["primary_teal"]):
    """Renders an aesthetic clinical KPI card."""
    st.markdown(
        f"""
        <div class="cp-metric-card" style="border-top-color: {accent_color};">
            <div class="cp-metric-label">{label}</div>
            <div class="cp-metric-value">{value}</div>
            <div class="cp-metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_risk_pill(risk_category: str) -> str:
    """Returns HTML for an accessible risk category badge."""
    cat = risk_category.lower()
    if "low" in cat:
        return f'<span class="cp-badge cp-badge-low">{risk_category}</span>'
    elif "mod" in cat:
        return f'<span class="cp-badge cp-badge-mod">{risk_category}</span>'
    elif "very" in cat or "vhigh" in cat:
        return f'<span class="cp-badge cp-badge-vhigh">{risk_category}</span>'
    else:
        return f'<span class="cp-badge cp-badge-high">{risk_category}</span>'


def get_plotly_theme():
    """Returns standardized layout dict for Plotly charts matching design system."""
    return dict(
        font=dict(family="Inter, sans-serif", color=COLORS["main_text"], size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=30, t=40, b=40),
        xaxis=dict(
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            showline=True,
            linecolor=COLORS["border"],
            tickfont=dict(size=11, color=COLORS["secondary_text"]),
        ),
        yaxis=dict(
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            showline=True,
            linecolor=COLORS["border"],
            tickfont=dict(size=11, color=COLORS["secondary_text"]),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=COLORS["main_text"]),
        ),
    )
