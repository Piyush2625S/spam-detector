# ============================================================
# FILE: app/app.py
# PURPOSE: Streamlit UI — Spam Detector
# MEMBER: Member 4
# REQUIRES: Flask API running on http://localhost:5000
# ============================================================

import streamlit as st
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import time
import base64
from pathlib import Path

# ============================================================
# PAGE CONFIG — must be first Streamlit call
# ============================================================

st.set_page_config(
    page_title="NLPShield — Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DESIGN SYSTEM — Light Theme
# Palette:
#   Background  : #FFFFFF  (clean white)
#   Surface     : #F8F9FA  (light surface)
#   Surface2    : #F0F2F5  (card layer)
#   Border      : #E2E6EA  (subtle border)
#   Border2     : #CDD3DB  (active border)
#   Primary     : #E8920A  (amber — darker for contrast on white)
#   Spam        : #D93025  (threat red)
#   Ham         : #0F9D58  (clear green)
#   Text 1      : #1A1A2E  (primary text)
#   Text 2      : #555F6D  (secondary text)
#   Text 3      : #8A9299  (muted)
# Display Font : Fraunces — editorial, analytical authority
# Body Font    : IBM Plex Mono — terminal precision
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #FFFFFF;
    --surface:   #F8F9FA;
    --surface2:  #F0F2F5;
    --border:    #E2E6EA;
    --border2:   #CDD3DB;
    --amber:     #E8920A;
    --amber-dim: #7A5212;
    --amber-glow:rgba(232,146,10,0.15);
    --spam:      #D93025;
    --spam-dim:  #FDECEA;
    --ham:       #0F9D58;
    --ham-dim:   #E6F4EA;
    --text1:     #1A1A2E;
    --text2:     #555F6D;
    --text3:     #8A9299;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
    --display:   'Fraunces', serif;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg);
}

/* ── Page background ── */
.main, .stApp {
    background-color: var(--bg);
}
.block-container {
    padding: 1.8rem 2.8rem 5rem;
    max-width: 1200px;
}

/* ── Hide chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #F4F6F8 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    font-family: var(--mono) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid var(--border);
    padding: 0;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    color: var(--text3);
    font-family: var(--mono) !important;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 10px 24px 10px;
    margin-bottom: -1px;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: var(--amber) !important;
    border-bottom: 2px solid var(--amber) !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text2) !important;
}
/* Remove tab highlight bar */
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    color: var(--text1) !important;
    font-family: var(--mono) !important;
    font-size: 12.5px !important;
    line-height: 1.8 !important;
    padding: 16px 18px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    caret-color: var(--amber);
    letter-spacing: 0.02em;
}
.stTextArea textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px var(--amber-glow) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder {
    color: var(--text3) !important;
    font-style: italic;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: var(--amber) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 12px 28px !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 2px 8px rgba(232,146,10,0.25) !important;
    position: relative !important;
}
.stButton > button[kind="primary"]:hover {
    background: #D4820A !important;
    box-shadow: 0 4px 16px rgba(232,146,10,0.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
    box-shadow: none !important;
}

/* ── Secondary buttons ── */
.stButton > button:not([kind="primary"]) {
    background: var(--surface) !important;
    color: var(--text2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    font-family: var(--mono) !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 8px 12px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: var(--surface2) !important;
    color: var(--text1) !important;
    border-color: var(--border2) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    color: var(--text2) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
.streamlit-expanderContent {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-bottom-left-radius: 2px !important;
    border-bottom-right-radius: 2px !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    overflow: hidden !important;
}
.stDataFrame [data-testid="stDataFrameResizable"] {
    background: var(--surface) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 20px 0 !important;
}

/* ── Alerts ── */
.stWarning, .stError, .stInfo, .stSuccess {
    border-radius: 2px !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
}

/* ── Spinner ── */
.stSpinner > div > div {
    border-top-color: var(--amber) !important;
}

/* ── Custom scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb {
    background: var(--border2);
    border-radius: 0;
}
::-webkit-scrollbar-thumb:hover { background: var(--text3); }

/* ── Result reveal animation ── */
@keyframes revealUp {
    from {
        opacity: 0;
        transform: translateY(14px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
@keyframes scanLine {
    0%   { transform: translateY(-100%); opacity: 0.3; }
    100% { transform: translateY(400px); opacity: 0; }
}
@keyframes pulseAmber {
    0%, 100% { box-shadow: 0 0 0 0 rgba(232,146,10,0.0); }
    50%       { box-shadow: 0 0 16px 4px rgba(232,146,10,0.15); }
}
@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 0 0 rgba(217,48,37,0.0); }
    50%       { box-shadow: 0 0 20px 4px rgba(217,48,37,0.2); }
}
@keyframes pulseGreen {
    0%, 100% { box-shadow: 0 0 0 0 rgba(15,157,88,0.0); }
    50%       { box-shadow: 0 0 16px 4px rgba(15,157,88,0.15); }
}

.result-reveal {
    animation: revealUp 0.4s cubic-bezier(0.22, 1, 0.36, 1) forwards;
}

/* ── Pipeline steps ── */
.pipeline-step {
    transition: border-color 0.2s ease, background 0.2s ease;
}
.pipeline-step:hover {
    border-color: var(--amber) !important;
    background: var(--surface2) !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS & HELPERS
# ============================================================

API_URL = "http://localhost:5000"

# Shared card style — used everywhere for consistency
def card(content, border_color="#E2E6EA", padding="20px 22px"):
    return f"""
    <div style='background:var(--surface2); border:1px solid {border_color};
                border-radius:3px; padding:{padding};
                margin-bottom:12px;'>
        {content}
    </div>
    """

def stat_card(value, label, color="#E8920A", sub=None):
    sub_html = f"<p style='font-size:10px;color:var(--text3);margin:5px 0 0;font-family:var(--mono);letter-spacing:0.06em;'>{sub}</p>" if sub else ""
    return f"""
    <div style='background:var(--surface); border:1px solid var(--border);
                border-top: 2px solid {color};
                border-radius:0 0 3px 3px; padding:16px 18px 14px;'>
        <p style='font-family:"Fraunces",serif;
                  font-size:28px; font-weight:600; color:{color};
                  margin:0; line-height:1; letter-spacing:-0.02em;'>
            {value}
        </p>
        <p style='font-size:9px; font-weight:600; color:var(--text3);
                  letter-spacing:0.14em; text-transform:uppercase;
                  margin:8px 0 0; font-family:var(--mono);'>
            {label}
        </p>
        {sub_html}
    </div>
    """

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        d = r.json()
        return d.get('status') == 'healthy', d.get('model_type', 'Unknown')
    except:
        return False, 'Offline'

def predict_message(message):
    try:
        r = requests.post(f"{API_URL}/predict",
                          json={"message": message}, timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def load_image_b64(path):
    """Load image as base64 — works on ALL Streamlit versions."""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = Path(path).suffix.lower().replace('.', '')
        if ext == 'jpg': ext = 'jpeg'
        return f"data:image/{ext};base64,{data}"
    except:
        return None

def confidence_bar_chart(label, confidence):
    """
    Clean confidence bar for light theme.
    """
    fig, ax = plt.subplots(figsize=(9, 0.65))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    fill_color = '#D93025' if label == 'spam' else '#0F9D58'
    track_color = '#E8ECF0'

    # Track
    ax.barh(0, 100, color=track_color, height=0.45,
            linewidth=0, zorder=1)
    # Fill
    ax.barh(0, confidence, color=fill_color, height=0.45,
            linewidth=0, zorder=2)

    # Label at the right edge
    ax.text(min(confidence - 1, 97), 0,
            f'{confidence:.1f}%',
            va='center', ha='right',
            color='white', fontsize=11,
            fontfamily='monospace',
            fontweight='bold', zorder=3)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 0.55)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


# ============================================================
# SIDEBAR
# ============================================================

api_ok, model_type = check_api()

with st.sidebar:

    # Logo / Brand
    st.markdown("""
    <div style='padding:28px 0 28px;'>
        <p style='font-family:"IBM Plex Mono",monospace;
                  font-size:9px; color:#8A9299;
                  letter-spacing:0.18em; margin:0 0 12px; text-transform:uppercase;'>
            System
        </p>
        <p style='font-family:"Fraunces",serif;
                  font-size:22px; font-weight:600;
                  color:#1A1A2E; margin:0; line-height:1.1; letter-spacing:-0.02em;'>
            NLPShield
        </p>
        <p style='font-family:"IBM Plex Mono",monospace;
                  font-size:9px; color:#E8920A;
                  letter-spacing:0.14em; margin:4px 0 0; text-transform:uppercase;'>
            Spam Classifier v1.0
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # API status
    dot_color   = '#0F9D58' if api_ok else '#D93025'
    status_text = 'Connected' if api_ok else 'Disconnected'
    status_bg   = '#E6F4EA' if api_ok else '#FDECEA'
    status_bd   = '#A8D5B5' if api_ok else '#F5C6C3'

    st.markdown(f"""
    <p style='font-family:"IBM Plex Mono",monospace; font-size:9px;
              color:#8A9299; letter-spacing:0.18em;
              margin-bottom:10px; text-transform:uppercase;'>API Status</p>
    <div style='background:{status_bg}; border:1px solid {status_bd};
                border-radius:2px; padding:12px 14px;'>
        <div style='display:flex; align-items:center; gap:9px;'>
            <span style='width:6px; height:6px; border-radius:50%;
                         background:{dot_color}; flex-shrink:0;
                         box-shadow:0 0 8px {dot_color}66;
                         display:inline-block;'></span>
            <span style='font-family:"IBM Plex Mono",monospace;
                         font-size:11px; font-weight:600;
                         color:{dot_color}; letter-spacing:0.06em;'>{status_text}</span>
        </div>
        <p style='font-family:"IBM Plex Mono",monospace;
                  font-size:10px; color:#555F6D;
                  margin:9px 0 0; line-height:1.7;'>
            localhost:5000<br>
            model: <span style='color:#8A9299;'>{model_type}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not api_ok:
        st.markdown("""
        <div style='background:#FDECEA; border:1px solid #F5C6C3;
                    border-radius:2px; padding:10px 14px; margin-top:8px;'>
            <p style='font-family:"IBM Plex Mono",monospace;
                      font-size:10px; color:#D93025; margin:0; line-height:1.7;'>
                $ python app/flask_api.py
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # Project metadata
    st.markdown("""
    <p style='font-family:"IBM Plex Mono",monospace; font-size:9px;
              color:#8A9299; letter-spacing:0.18em;
              margin-bottom:10px; text-transform:uppercase;'>Project</p>
    """, unsafe_allow_html=True)

    rows = [
        ("Dataset",   "UCI SMS Spam"),
        ("Records",   "5,169 messages"),
        ("Split",     "80% train / 20% test"),
        ("Accuracy",  "~98%"),
        ("Algorithm", "Naive Bayes / LR"),
        ("Features",  "TF-IDF, n-grams"),
    ]
    for k, v in rows:
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between;
                    align-items:baseline; padding:7px 0;
                    border-bottom:1px solid #E2E6EA;'>
            <span style='font-size:10px; color:#555F6D;
                         font-family:"IBM Plex Mono",monospace;
                         letter-spacing:0.04em;'>{k}</span>
            <span style='font-size:10px; color:#1A1A2E;
                         font-family:"IBM Plex Mono",monospace;
                         text-align:right;'>
                {v}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # Tech chips
    st.markdown("""
    <p style='font-family:"IBM Plex Mono",monospace; font-size:9px;
              color:#8A9299; letter-spacing:0.18em;
              margin-bottom:10px; text-transform:uppercase;'>Tech Stack</p>
    """, unsafe_allow_html=True)

    for tech in ['Python 3.10', 'scikit-learn', 'NLTK',
                 'TF-IDF Vectorizer', 'Flask REST API', 'Streamlit']:
        st.markdown(f"""
        <span style='display:inline-block;
                     font-family:"IBM Plex Mono",monospace;
                     font-size:9px; color:#555F6D;
                     background:#F0F2F5; border:1px solid #E2E6EA;
                     border-radius:1px; padding:3px 8px;
                     margin:2px 2px 2px 0; line-height:1.8;
                     letter-spacing:0.05em;'>
            {tech}
        </span>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# MAIN — Header
# ============================================================

st.markdown("""
<div style='padding:8px 0 32px; border-bottom:1px solid #E2E6EA; margin-bottom:28px;'>
    <div style='display:flex; align-items:flex-end;
                justify-content:space-between; flex-wrap:wrap; gap:16px;'>
        <div>
            <p style='font-family:"IBM Plex Mono",monospace;
                      font-size:9px; color:#E8920A;
                      letter-spacing:0.2em; margin:0 0 8px;
                      text-transform:uppercase;'>
                Signal Analysis / NLP
            </p>
            <h1 style='font-family:"Fraunces",serif;
                       font-size:38px; font-weight:600;
                       color:#1A1A2E; margin:0;
                       letter-spacing:-0.03em; line-height:1.05;'>
                SMS &amp; Email<br>Spam Classifier
            </h1>
            <p style='font-size:13px; color:#8A9299; margin:10px 0 0;
                      font-family:"IBM Plex Mono",monospace;
                      font-weight:400; line-height:1.6; letter-spacing:0.02em;'>
                Real-time classification · TF-IDF vectorization · Supervised ML
            </p>
        </div>
        <div style='display:flex; gap:6px; flex-wrap:wrap; align-items:flex-end;'>
            <span style='background:#F0F2F5; border:1px solid #E2E6EA;
                         color:#555F6D; font-size:9px; font-weight:500;
                         padding:4px 10px; border-radius:1px;
                         letter-spacing:0.14em; font-family:"IBM Plex Mono",monospace;
                         text-transform:uppercase;'>
                NLP
            </span>
            <span style='background:#F0F2F5; border:1px solid #E2E6EA;
                         color:#555F6D; font-size:9px; font-weight:500;
                         padding:4px 10px; border-radius:1px;
                         letter-spacing:0.14em; font-family:"IBM Plex Mono",monospace;
                         text-transform:uppercase;'>
                Machine Learning
            </span>
            <span style='background:#F0F2F5; border:1px solid #E2E6EA;
                         color:#555F6D; font-size:9px; font-weight:500;
                         padding:4px 10px; border-radius:1px;
                         letter-spacing:0.14em; font-family:"IBM Plex Mono",monospace;
                         text-transform:uppercase;'>
                Python
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Pipeline strip
steps = ["Raw Input", "Lowercase", "Remove Noise",
         "Stopwords", "Stemming", "TF-IDF", "ML Model", "Result"]
cols  = st.columns(len(steps))
for i, (col, step) in enumerate(zip(cols, steps)):
    with col:
        num   = f"0{i+1}" if i < 9 else str(i+1)
        is_last = i == len(steps) - 1
        color = "#E8920A" if is_last else "#E2E6EA"
        tc    = "#FFFFFF" if is_last else "#8A9299"
        bg    = "#E8920A" if is_last else "#F8F9FA"
        connector = "→" if i < len(steps) - 1 else ""
        st.markdown(f"""
        <div class='pipeline-step'
             style='background:{bg}; border:1px solid {color};
                    border-radius:2px; padding:9px 6px;
                    text-align:center; position:relative;'>
            <p style='font-family:"IBM Plex Mono",monospace;
                      font-size:8px; color:{"#FFFFFF" if is_last else "#CDD3DB"};
                      margin:0; letter-spacing:0.1em;'>{num}</p>
            <p style='font-size:10px; font-weight:500; color:{tc};
                      margin:4px 0 0; line-height:1.2;
                      font-family:"IBM Plex Mono",monospace; letter-spacing:0.02em;'>{step}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(["  ANALYZE  ", "  BATCH  ", "  PERFORMANCE  "])


# ============================================================
# TAB 1 — Analyze
# ============================================================

with tab1:

    left, right = st.columns([3, 2], gap="large")

    with left:

        # Sample buttons
        st.markdown("""
        <p style='font-size:9px; font-weight:600; color:#8A9299;
                  letter-spacing:0.16em; text-transform:uppercase;
                  margin-bottom:12px; font-family:"IBM Plex Mono",monospace;'>
            Quick Test Samples
        </p>
        """, unsafe_allow_html=True)

        samples = {
            "🔴 Spam #1": "WINNER!! Claim your FREE £1000 prize now! Call 08712300150",
            "🔴 Spam #2": "FREE entry! Win FA Cup final tkts! Text FA to 87121 now",
            "🟢 Ham #1" : "Hey, are you coming to the 3pm meeting tomorrow?",
            "🟢 Ham #2" : "Can you pick up some groceries on your way home please?",
        }

        b1, b2, b3, b4 = st.columns(4)
        for col, (lbl, txt) in zip([b1, b2, b3, b4], samples.items()):
            with col:
                if st.button(lbl, key=f"smp_{lbl}", use_container_width=True):
                    st.session_state['msg_area'] = txt
                    st.session_state['auto_run'] = True

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        msg = st.text_area(
            label="message_input",
            label_visibility="collapsed",
            height=148,
            placeholder="Paste or type an SMS / email message to analyze...",
            key="msg_area"
        )

        cw = len(msg)
        ww = len(msg.split()) if msg.strip() else 0
        st.markdown(f"""
        <p style='font-family:"IBM Plex Mono",monospace; font-size:10px;
                  color:#CDD3DB; text-align:right; margin-top:-4px;
                  letter-spacing:0.04em;'>
            {cw} chars · {ww} words
        </p>
        """, unsafe_allow_html=True)

        run_btn = st.button(
            "Analyze Message →",
            type="primary",
            use_container_width=True
        )

        # Auto-trigger analysis when a sample button is clicked
        if st.session_state.pop('auto_run', False):
            run_btn = True

    with right:
        st.markdown("""
        <p style='font-size:9px; font-weight:600; color:#8A9299;
                  letter-spacing:0.16em; text-transform:uppercase;
                  margin-bottom:14px; font-family:"IBM Plex Mono",monospace;'>
            Processing Pipeline
        </p>
        """, unsafe_allow_html=True)

        pipeline_items = [
            ("text.lower()",         "Convert to lowercase"),
            ("re.sub(r'http\\S+')",  "Strip URLs and links"),
            ("re.sub(r'\\d{7,}')",   "Remove phone numbers"),
            ("str.punctuation",      "Remove punctuation"),
            ("stopwords.remove()",   "Drop common words"),
            ("PorterStemmer()",      "Reduce to root form"),
            ("TfidfVectorizer()",    "Score word importance"),
            ("model.predict()",      "Output classification"),
        ]

        for fn, desc in pipeline_items:
            st.markdown(f"""
            <div style='display:flex; gap:12px; margin-bottom:10px;
                        align-items:flex-start; padding-bottom:10px;
                        border-bottom:1px solid #E2E6EA;'>
                <code style='font-family:"IBM Plex Mono",monospace;
                             font-size:9px; color:#D4820A;
                             background:#FEF3E2; border:1px solid #F5D8A0;
                             border-radius:1px; padding:2px 7px;
                             white-space:nowrap; flex-shrink:0; letter-spacing:0.04em;'>
                    {fn}
                </code>
                <span style='font-size:11px; color:#555F6D;
                             line-height:1.5; padding-top:2px;
                             font-family:"IBM Plex Sans",sans-serif;'>
                    {desc}
                </span>
            </div>
            """, unsafe_allow_html=True)

    # ── Prediction result ──
    if run_btn:
        if not msg.strip():
            st.warning("Please enter a message to analyze.")
        elif not api_ok:
            st.error("API is offline. Run: `python app/flask_api.py`")
        else:
            with st.spinner("Analyzing..."):
                time.sleep(0.35)
                result = predict_message(msg)

            if result is None:
                st.error("Failed to reach Flask API.")
            else:
                # Store result + the message that produced it in session state
                st.session_state['last_result'] = result
                st.session_state['last_msg']    = msg

    # ── Always render result if one exists in session state ──
    if 'last_result' in st.session_state:
        result     = st.session_state['last_result']
        msg_used   = st.session_state['last_msg']
        label      = result['label']
        confidence = result['confidence']
        cleaned    = result['cleaned']

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        is_spam = label == 'spam'
        verdict = "SPAM" if is_spam else "HAM"
        v_color = "#D93025" if is_spam else "#0F9D58"
        v_bg    = "#FDECEA" if is_spam else "#E6F4EA"
        v_bd    = "#F5C6C3" if is_spam else "#A8D5B5"

        # Risk is only meaningful for SPAM — for HAM it's always None
        if is_spam:
            risk    = "High" if confidence > 90 else "Medium" if confidence > 70 else "Low"
            r_color = "#D93025" if risk == "High" else "#E8920A" if risk == "Medium" else "#0F9D58"
        else:
            risk    = "None"
            r_color = "#0F9D58"

        # Four stat cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(stat_card(verdict, "Verdict", v_color), unsafe_allow_html=True)
        with c2:
            st.markdown(stat_card(f"{confidence}%", "Confidence", "#E8920A"), unsafe_allow_html=True)
        with c3:
            st.markdown(stat_card(risk, "Risk Level", r_color), unsafe_allow_html=True)
        with c4:
            tc = len(cleaned.split()) if cleaned else 0
            st.markdown(stat_card(str(tc), "Tokens", "#8A9299", "after cleaning"), unsafe_allow_html=True)

        # Confidence bar
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
        fig = confidence_bar_chart(label, confidence)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Result banner
        title = "THREAT DETECTED — Spam" if is_spam else "SIGNAL CLEAR — Legitimate"
        body  = (
            f"This message has been classified as <b>unsolicited or potentially malicious</b> content with {confidence}% confidence. Do not click any links or call any numbers in this message."
            if is_spam else
            f"This message has been classified as a <b>legitimate communication</b> with {confidence}% confidence. No suspicious patterns were detected."
        )

        pulse_anim = "animation:pulseRed 2s ease-in-out 2;" if is_spam else "animation:pulseGreen 2s ease-in-out 2;"

        st.markdown(f"""
        <div class='result-reveal'
             style='background:{v_bg}; border:1px solid {v_bd};
                    border-left:3px solid {v_color};
                    border-radius:0 3px 3px 0;
                    padding:20px 24px; margin:14px 0;
                    position:relative; overflow:hidden;
                    {pulse_anim}'>
            <p style='font-family:"IBM Plex Mono",monospace;
                      font-size:9px; color:{v_color}; opacity:0.8;
                      margin:0 0 6px; letter-spacing:0.18em; text-transform:uppercase;'>
                Classification Result
            </p>
            <p style='font-family:"Fraunces",serif;
                      font-size:18px; font-weight:600;
                      color:{v_color}; margin:0 0 8px; letter-spacing:-0.01em;'>
                {title}
            </p>
            <p style='font-size:12px; color:#555F6D;
                      margin:0; line-height:1.7;
                      font-family:"IBM Plex Sans",sans-serif;'>
                {body}
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("View processing details"):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("""
                <p style='font-size:9px; font-weight:600;
                          color:#8A9299; margin-bottom:8px;
                          font-family:"IBM Plex Mono",monospace;
                          letter-spacing:0.12em; text-transform:uppercase;'>
                    Original Input
                </p>
                """, unsafe_allow_html=True)
                st.code(msg_used, language=None)
            with d2:
                st.markdown("""
                <p style='font-size:9px; font-weight:600;
                          color:#8A9299; margin-bottom:8px;
                          font-family:"IBM Plex Mono",monospace;
                          letter-spacing:0.12em; text-transform:uppercase;'>
                    Post-Processing Tokens
                </p>
                """, unsafe_allow_html=True)
                st.code(
                    cleaned if cleaned
                    else "(empty after cleaning)",
                    language=None
                )
            st.markdown(f"""
            <div style='background:#F8F9FA; border:1px solid #E2E6EA;
                        border-radius:2px; padding:14px 18px; margin-top:8px;
                        font-family:"IBM Plex Mono",monospace;
                        font-size:11px; color:#8A9299; line-height:2.2;'>
                label      → <span style='color:{v_color};'>{label}</span><br>
                confidence → <span style='color:#E8920A;'>{confidence}%</span><br>
                tokens     → <span style='color:#555F6D;'>{tc} words</span><br>
                model      → <span style='color:#555F6D;'>{model_type}</span>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# TAB 2 — Batch
# ============================================================

with tab2:

    st.markdown("""
    <p style='font-size:10px; color:#8A9299; margin-bottom:18px;
              font-family:"IBM Plex Mono",monospace; letter-spacing:0.06em;'>
        One message per line · Maximum 20 messages per run
    </p>
    """, unsafe_allow_html=True)

    batch_txt = st.text_area(
        label="batch",
        label_visibility="collapsed",
        height=200,
        placeholder=(
            "WINNER!! You have been selected for a £500 cash prize!\n"
            "Hey, are you coming to class tomorrow?\n"
            "FREE entry! Win FA Cup tickets! Text WIN to 87121\n"
            "Can we reschedule our meeting to Friday afternoon?\n"
            "Congratulations! Your number has won a free iPhone!"
        )
    )

    batch_btn = st.button(
        "Run Batch Analysis →",
        type="primary",
        use_container_width=True,
        key="batch_run"
    )

    if batch_btn:
        if not batch_txt.strip():
            st.warning("Enter at least one message.")
        elif not api_ok:
            st.error("API offline.")
        else:
            msgs = [m.strip() for m in batch_txt.strip().split('\n') if m.strip()]
            if len(msgs) > 20:
                st.info("Showing first 20 messages only.")
                msgs = msgs[:20]

            with st.spinner(f"Classifying {len(msgs)} messages..."):
                rows = []
                for i, m in enumerate(msgs, 1):
                    res = predict_message(m)
                    if res:
                        rows.append({
                            '#'          : i,
                            'Message'    : m[:68] + ('…' if len(m) > 68 else ''),
                            'Label'      : '🔴 SPAM' if res['label']=='spam' else '🟢 HAM',
                            'Confidence' : f"{res['confidence']}%",
                            '_raw'       : res['label'],
                        })

            if rows:
                spam_n = sum(1 for r in rows if r['_raw']=='spam')
                ham_n  = len(rows) - spam_n
                rate   = f"{spam_n/len(rows)*100:.0f}%"

                m1, m2, m3, m4 = st.columns(4)
                for col, val, lbl, clr in zip(
                    [m1, m2, m3, m4],
                    [len(rows), spam_n, ham_n, rate],
                    ['Total', 'Spam', 'Ham', 'Spam Rate'],
                    ['#E8920A', '#D93025', '#0F9D58', '#8A9299']
                ):
                    with col:
                        st.markdown(stat_card(str(val), lbl, clr), unsafe_allow_html=True)

                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                df = pd.DataFrame(rows)[['#', 'Message', 'Label', 'Confidence']]
                st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================
# TAB 3 — Performance
# ============================================================

with tab3:

    SCREENS = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'screenshots')
    )

    st.markdown("""
    <p style='font-size:10px; color:#8A9299; margin-bottom:24px;
              font-family:"IBM Plex Mono",monospace; letter-spacing:0.06em;'>
        Evaluation on held-out test set · 80/20 split · 1,034 test messages
    </p>
    """, unsafe_allow_html=True)

    # Metric row
    p1, p2, p3, p4 = st.columns(4)
    for col, val, lbl, clr in zip(
        [p1, p2, p3, p4],
        ['98.3%', '98.1%', '92.4%', '95.2%'],
        ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        ['#E8920A', '#0F9D58', '#8A9299', '#1A1A2E']
    ):
        with col:
            st.markdown(stat_card(val, lbl, clr), unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # ── Model comparison chart ──
    comp_path = os.path.join(SCREENS, 'model_comparison.png')
    comp_b64  = load_image_b64(comp_path)

    if comp_b64:
        st.markdown("""
        <p style='font-size:9px; font-weight:600; color:#8A9299;
                  letter-spacing:0.16em; text-transform:uppercase;
                  margin-bottom:10px; font-family:"IBM Plex Mono",monospace;'>
            Model Comparison — Naive Bayes vs Logistic Regression
        </p>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#F8F9FA; border:1px solid #E2E6EA;
                    border-radius:3px; padding:16px; margin-bottom:20px;'>
            <img src='{comp_b64}'
                 style='width:100%; border-radius:2px; display:block;'>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Run `python src/train.py` to generate charts → screenshots/")

    # ── Confusion matrices ──
    st.markdown("""
    <p style='font-size:9px; font-weight:600; color:#8A9299;
              letter-spacing:0.16em; text-transform:uppercase;
              margin-bottom:10px; font-family:"IBM Plex Mono",monospace;'>
        Confusion Matrices
    </p>
    """, unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    for col, fname, title in zip(
        [cc1, cc2],
        ['confusion_naive_bayes.png', 'confusion_logistic_regression.png'],
        ['Naive Bayes', 'Logistic Regression']
    ):
        with col:
            p   = os.path.join(SCREENS, fname)
            b64 = load_image_b64(p)
            if b64:
                st.markdown(f"""
                <div style='background:#F8F9FA; border:1px solid #E2E6EA;
                            border-radius:3px; padding:14px;'>
                    <p style='font-size:10px; font-weight:600; color:#8A9299;
                              margin:0 0 10px; font-family:"IBM Plex Mono",monospace;
                              letter-spacing:0.1em; text-transform:uppercase;'>{title}</p>
                    <img src='{b64}'
                         style='width:100%; border-radius:2px; display:block;'>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"{fname} not found in screenshots/")

    # ── Definitions ──
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:9px; font-weight:600; color:#8A9299;
              letter-spacing:0.16em; text-transform:uppercase;
              margin-bottom:14px; font-family:"IBM Plex Mono",monospace;'>
        Metric Definitions
    </p>
    """, unsafe_allow_html=True)

    defs = [
        ("Accuracy",   "#E8920A",
         "Proportion of all messages classified correctly by the model."),
        ("Precision",  "#0F9D58",
         "Of all messages flagged as spam, what fraction were actually spam. "
         "High precision = fewer false alarms."),
        ("Recall",     "#8A9299",
         "Of all actual spam messages in the dataset, what fraction did the "
         "model successfully detect. High recall = fewer missed spam."),
        ("F1 Score",   "#1A1A2E",
         "Harmonic mean of precision and recall. Best single metric when "
         "classes are imbalanced (more ham than spam)."),
    ]

    for term, color, defn in defs:
        st.markdown(f"""
        <div style='display:flex; gap:16px; padding:14px 0;
                    border-bottom:1px solid #E2E6EA;
                    align-items:flex-start;'>
            <span style='min-width:76px; font-size:11px; font-weight:600;
                         color:{color}; padding-top:1px;
                         font-family:"IBM Plex Mono",monospace;
                         letter-spacing:0.06em;'>{term}</span>
            <span style='font-size:12px; color:#555F6D;
                         line-height:1.7;
                         font-family:"IBM Plex Sans",sans-serif;'>{defn}</span>
        </div>
        """, unsafe_allow_html=True)
