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
# DESIGN SYSTEM
# Palette:
#   Background  : #0F1117  (near black — not pure black)
#   Surface     : #161B27  (card background)
#   Border      : #252D3D  (subtle border)
#   Primary     : #4F8EF7  (electric blue — accents, CTA)
#   Spam        : #F05252  (red — danger)
#   Ham         : #31C48D  (teal-green — safe)
#   Warning     : #FBBF24  (amber)
#   Text 1      : #F1F5F9  (primary text)
#   Text 2      : #94A3B8  (secondary text)
#   Text 3      : #475569  (muted)
# Font: Plus Jakarta Sans — modern, clean, not overused
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Page background ── */
.main, .stApp {
    background-color: #0F1117;
}
.block-container {
    padding: 1.5rem 2.5rem 4rem;
    max-width: 1140px;
}

/* ── Hide chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0A0D14 !important;
    border-right: 1px solid #1E2535 !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161B27;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #252D3D;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-radius: 7px;
    color: #64748B;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 13px;
    font-weight: 500;
    padding: 7px 20px;
    transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
    background: #252D3D !important;
    color: #F1F5F9 !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #CBD5E1 !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background: #161B27 !important;
    border: 1px solid #252D3D !important;
    border-radius: 10px !important;
    color: #F1F5F9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
    padding: 14px 16px !important;
    transition: border-color 0.15s !important;
    caret-color: #4F8EF7;
}
.stTextArea textarea:focus {
    border-color: #4F8EF7 !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.15) !important;
}
.stTextArea textarea::placeholder { color: #334155 !important; }

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #4F8EF7 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 11px 24px !important;
    letter-spacing: 0.01em !important;
    transition: all 0.15s !important;
    box-shadow: 0 1px 3px rgba(79,142,247,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #3B7BF0 !important;
    box-shadow: 0 4px 12px rgba(79,142,247,0.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
}

/* ── Secondary buttons ── */
.stButton > button:not([kind="primary"]) {
    background: #161B27 !important;
    color: #94A3B8 !important;
    border: 1px solid #252D3D !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 7px 14px !important;
    transition: all 0.15s !important;
    transform: none !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: #1E2535 !important;
    color: #E2E8F0 !important;
    border-color: #374151 !important;
    transform: none !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #161B27 !important;
    border: 1px solid #252D3D !important;
    border-radius: 10px !important;
    color: #64748B !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: #161B27 !important;
    border: 1px solid #252D3D !important;
    border-top: none !important;
    border-bottom-left-radius: 10px !important;
    border-bottom-right-radius: 10px !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid #252D3D !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Divider ── */
hr { border-color: #1E2535 !important; margin: 20px 0 !important; }

/* ── Alerts ── */
.stWarning, .stError, .stInfo, .stSuccess {
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0F1117; }
::-webkit-scrollbar-thumb { background: #252D3D; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS & HELPERS
# ============================================================

API_URL = "http://localhost:5000"

# Shared card style — used everywhere for consistency
def card(content, border_color="#252D3D", padding="20px 22px"):
    return f"""
    <div style='background:#161B27; border:1px solid {border_color};
                border-radius:12px; padding:{padding};
                margin-bottom:12px;'>
        {content}
    </div>
    """

def stat_card(value, label, color="#4F8EF7", sub=None):
    sub_html = f"<p style='font-size:11px;color:#475569;margin:4px 0 0;'>{sub}</p>" if sub else ""
    return f"""
    <div style='background:#161B27; border:1px solid #252D3D;
                border-radius:12px; padding:18px 20px;'>
        <p style='font-size:26px; font-weight:700; color:{color};
                  margin:0; line-height:1; font-family:"Plus Jakarta Sans",sans-serif;'>
            {value}
        </p>
        <p style='font-size:11px; font-weight:600; color:#475569;
                  letter-spacing:0.07em; text-transform:uppercase;
                  margin:8px 0 0;'>
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
    Beautiful minimal confidence bar.
    Dark background, colored fill, clean typography.
    """
    fig, ax = plt.subplots(figsize=(9, 0.75))
    fig.patch.set_facecolor('#0F1117')
    ax.set_facecolor('#0F1117')

    fill_color = '#F05252' if label == 'spam' else '#31C48D'
    track_color = '#1E2535'

    # Track
    ax.barh(0, 100, color=track_color, height=0.5,
            linewidth=0, zorder=1)
    # Fill
    ax.barh(0, confidence, color=fill_color, height=0.5,
            linewidth=0, zorder=2)

    # Label at the right edge
    ax.text(min(confidence - 1, 97), 0,
            f'{confidence:.1f}%',
            va='center', ha='right',
            color='white', fontsize=12,
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
    <div style='padding:20px 0 24px;'>
        <div style='display:flex; align-items:center; gap:10px;'>
            <div style='width:34px; height:34px; background:#4F8EF7;
                        border-radius:8px; display:flex; align-items:center;
                        justify-content:center; flex-shrink:0;'>
                <span style='color:#fff; font-size:17px;'>🛡</span>
            </div>
            <div>
                <p style='font-family:"Plus Jakarta Sans",sans-serif;
                          font-size:16px; font-weight:700;
                          color:#F1F5F9; margin:0; letter-spacing:-0.01em;'>
                    NLPShield
                </p>
                <p style='font-family:"JetBrains Mono",monospace;
                          font-size:10px; color:#334155;
                          letter-spacing:0.06em; margin:2px 0 0;'>
                    SPAM CLASSIFIER
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # API status
    dot_color   = '#31C48D' if api_ok else '#F05252'
    status_text = 'Connected' if api_ok else 'Disconnected'
    status_bg   = '#0D2318' if api_ok else '#1F0D0D'
    status_bd   = '#1A4731' if api_ok else '#3D1515'

    st.markdown(f"""
    <p style='font-family:"JetBrains Mono",monospace; font-size:10px;
              color:#334155; letter-spacing:0.1em;
              margin-bottom:8px;'>API STATUS</p>
    <div style='background:{status_bg}; border:1px solid {status_bd};
                border-radius:9px; padding:12px 14px;'>
        <div style='display:flex; align-items:center; gap:8px;'>
            <span style='width:8px; height:8px; border-radius:50%;
                         background:{dot_color}; flex-shrink:0;
                         box-shadow:0 0 6px {dot_color}44;'></span>
            <span style='font-family:"Plus Jakarta Sans",sans-serif;
                         font-size:13px; font-weight:600;
                         color:{dot_color};'>{status_text}</span>
        </div>
        <p style='font-family:"JetBrains Mono",monospace;
                  font-size:11px; color:#334155;
                  margin:8px 0 0; line-height:1.6;'>
            localhost:5000<br>
            model: <span style='color:#64748B;'>{model_type}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not api_ok:
        st.markdown("""
        <div style='background:#1A0D0D; border:1px solid #3D1515;
                    border-radius:9px; padding:10px 14px; margin-top:8px;'>
            <p style='font-family:"JetBrains Mono",monospace;
                      font-size:11px; color:#F05252; margin:0; line-height:1.6;'>
                $ python app/flask_api.py
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # Project metadata
    st.markdown("""
    <p style='font-family:"JetBrains Mono",monospace; font-size:10px;
              color:#334155; letter-spacing:0.1em;
              margin-bottom:10px;'>PROJECT</p>
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
                    align-items:center; padding:7px 0;
                    border-bottom:1px solid #1A2030;'>
            <span style='font-size:12px; color:#334155;
                         font-weight:500;'>{k}</span>
            <span style='font-size:12px; color:#64748B;
                         font-family:"JetBrains Mono",monospace;'>
                {v}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # Tech chips
    st.markdown("""
    <p style='font-family:"JetBrains Mono",monospace; font-size:10px;
              color:#334155; letter-spacing:0.1em;
              margin-bottom:10px;'>TECH STACK</p>
    """, unsafe_allow_html=True)

    for tech in ['Python 3.10', 'scikit-learn', 'NLTK',
                 'TF-IDF Vectorizer', 'Flask REST API', 'Streamlit']:
        st.markdown(f"""
        <span style='display:inline-block;
                     font-family:"JetBrains Mono",monospace;
                     font-size:10px; color:#475569;
                     background:#161B27; border:1px solid #252D3D;
                     border-radius:5px; padding:3px 9px;
                     margin:2px 2px 2px 0; line-height:1.8;'>
            {tech}
        </span>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# MAIN — Header
# ============================================================

st.markdown("""
<div style='padding:4px 0 20px;'>
    <div style='display:flex; align-items:flex-end;
                justify-content:space-between; flex-wrap:wrap; gap:12px;'>
        <div>
            <h1 style='font-family:"Plus Jakarta Sans",sans-serif;
                       font-size:30px; font-weight:700;
                       color:#F1F5F9; margin:0;
                       letter-spacing:-0.02em; line-height:1.1;'>
                SMS & Email Spam Classifier
            </h1>
            <p style='font-size:14px; color:#475569; margin:6px 0 0;
                      font-weight:400; line-height:1.5;'>
                Real-time message classification using TF-IDF vectorization
                and supervised machine learning.
            </p>
        </div>
        <div style='display:flex; gap:8px; flex-wrap:wrap;'>
            <span style='background:#162032; border:1px solid #1E3A5F;
                         color:#4F8EF7; font-size:11px; font-weight:600;
                         padding:4px 12px; border-radius:6px;
                         letter-spacing:0.04em;'>
                NLP
            </span>
            <span style='background:#162032; border:1px solid #1E3A5F;
                         color:#4F8EF7; font-size:11px; font-weight:600;
                         padding:4px 12px; border-radius:6px;
                         letter-spacing:0.04em;'>
                MACHINE LEARNING
            </span>
            <span style='background:#162032; border:1px solid #1E3A5F;
                         color:#4F8EF7; font-size:11px; font-weight:600;
                         padding:4px 12px; border-radius:6px;
                         letter-spacing:0.04em;'>
                PYTHON
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
        color = "#4F8EF7" if i == len(steps)-1 else "#252D3D"
        tc    = "#F1F5F9" if i == len(steps)-1 else "#94A3B8"
        st.markdown(f"""
        <div style='background:#161B27; border:1px solid {color};
                    border-radius:8px; padding:8px 6px;
                    text-align:center;'>
            <p style='font-family:"JetBrains Mono",monospace;
                      font-size:9px; color:#334155;
                      margin:0; letter-spacing:0.05em;'>{num}</p>
            <p style='font-size:11px; font-weight:600; color:{tc};
                      margin:4px 0 0; line-height:1.2;'>{step}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(["  🔍  Analyze  ", "  📋  Batch  ", "  📊  Performance  "])


# ============================================================
# TAB 1 — Analyze
# ============================================================

with tab1:

    left, right = st.columns([3, 2], gap="large")

    with left:

        # Sample buttons
        st.markdown("""
        <p style='font-size:11px; font-weight:600; color:#475569;
                  letter-spacing:0.07em; text-transform:uppercase;
                  margin-bottom:10px;'>
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
                    st.session_state['msg'] = txt

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        msg = st.text_area(
            label="message_input",
            label_visibility="collapsed",
            value=st.session_state.get('msg', ''),
            height=148,
            placeholder="Paste or type an SMS / email message to analyze...",
            key="msg_area"
        )

        cw = len(msg)
        ww = len(msg.split()) if msg.strip() else 0
        st.markdown(f"""
        <p style='font-family:"JetBrains Mono",monospace; font-size:11px;
                  color:#334155; text-align:right; margin-top:-4px;'>
            {cw} chars · {ww} words
        </p>
        """, unsafe_allow_html=True)

        run_btn = st.button(
            "Analyze Message →",
            type="primary",
            use_container_width=True
        )

    with right:
        st.markdown("""
        <p style='font-size:11px; font-weight:600; color:#475569;
                  letter-spacing:0.07em; text-transform:uppercase;
                  margin-bottom:12px;'>
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
            <div style='display:flex; gap:10px; margin-bottom:9px;
                        align-items:flex-start;'>
                <code style='font-family:"JetBrains Mono",monospace;
                             font-size:10px; color:#4F8EF7;
                             background:#0D1525; border:1px solid #1E2D45;
                             border-radius:5px; padding:2px 7px;
                             white-space:nowrap; flex-shrink:0;'>
                    {fn}
                </code>
                <span style='font-size:12px; color:#475569;
                             line-height:1.5; padding-top:2px;'>
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
                label      = result['label']
                confidence = result['confidence']
                cleaned    = result['cleaned']

                st.markdown(
                    "<div style='height:8px;'></div>",
                    unsafe_allow_html=True
                )

                is_spam = label == 'spam'
                verdict = "SPAM" if is_spam else "HAM"
                v_color = "#F05252" if is_spam else "#31C48D"
                v_bg    = "#1F0D0D" if is_spam else "#0D2318"
                v_bd    = "#3D1515" if is_spam else "#1A4731"
                risk    = "High" if confidence > 90 else "Medium" if confidence > 70 else "Low"
                r_color = "#F05252" if risk=="High" else "#FBBF24" if risk=="Medium" else "#31C48D"

                # Four stat cards
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(stat_card(verdict, "Verdict", v_color), unsafe_allow_html=True)
                with c2:
                    st.markdown(stat_card(f"{confidence}%", "Confidence", "#4F8EF7"), unsafe_allow_html=True)
                with c3:
                    st.markdown(stat_card(risk, "Risk Level", r_color), unsafe_allow_html=True)
                with c4:
                    tc = len(cleaned.split()) if cleaned else 0
                    st.markdown(stat_card(str(tc), "Tokens", "#94A3B8", "after cleaning"), unsafe_allow_html=True)

                # Confidence bar
                st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
                fig = confidence_bar_chart(label, confidence)
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # Result banner
                icon  = "🚨" if is_spam else "✅"
                title = "Spam Detected" if is_spam else "Legitimate Message"
                body  = (
                    f"This message has been classified as <b>unsolicited or potentially malicious</b> content with {confidence}% confidence. Do not click any links or call any numbers in this message."
                    if is_spam else
                    f"This message has been classified as a <b>legitimate communication</b> with {confidence}% confidence. No suspicious patterns were detected."
                )

                st.markdown(f"""
                <div style='background:{v_bg}; border:1px solid {v_bd};
                            border-left:4px solid {v_color};
                            border-radius:0 12px 12px 0;
                            padding:18px 22px; margin:12px 0;'>
                    <p style='font-size:15px; font-weight:700;
                              color:{v_color}; margin:0 0 6px;'>
                        {icon} {title}
                    </p>
                    <p style='font-size:13px; color:#64748B;
                              margin:0; line-height:1.6;'>
                        {body}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Detail expander
                with st.expander("View processing details"):
                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown("""
                        <p style='font-size:12px; font-weight:600;
                                  color:#475569; margin-bottom:6px;'>
                            Original Input
                        </p>
                        """, unsafe_allow_html=True)
                        st.code(msg, language=None)
                    with d2:
                        st.markdown("""
                        <p style='font-size:12px; font-weight:600;
                                  color:#475569; margin-bottom:6px;'>
                            Post-Processing Tokens
                        </p>
                        """, unsafe_allow_html=True)
                        st.code(
                            cleaned if cleaned
                            else "(empty after cleaning)",
                            language=None
                        )
                    st.markdown(f"""
                    <div style='background:#0D1525; border:1px solid #1E2D45;
                                border-radius:8px; padding:12px 16px; margin-top:8px;
                                font-family:"JetBrains Mono",monospace;
                                font-size:12px; color:#475569; line-height:2;'>
                        label      → <span style='color:{v_color};'>{label}</span><br>
                        confidence → <span style='color:#4F8EF7;'>{confidence}%</span><br>
                        tokens     → <span style='color:#94A3B8;'>{tc} words</span><br>
                        model      → <span style='color:#94A3B8;'>{model_type}</span>
                    </div>
                    """, unsafe_allow_html=True)


# ============================================================
# TAB 2 — Batch
# ============================================================

with tab2:

    st.markdown("""
    <p style='font-size:13px; color:#475569; margin-bottom:18px;'>
        Enter one message per line · Maximum 20 messages
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
                    ['#4F8EF7', '#F05252', '#31C48D', '#FBBF24']
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
    <p style='font-size:13px; color:#475569; margin-bottom:20px;'>
        Evaluation on held-out test set · 80/20 split · 1,034 test messages
    </p>
    """, unsafe_allow_html=True)

    # Metric row
    p1, p2, p3, p4 = st.columns(4)
    for col, val, lbl, clr in zip(
        [p1, p2, p3, p4],
        ['98.3%', '98.1%', '92.4%', '95.2%'],
        ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        ['#4F8EF7', '#31C48D', '#FBBF24', '#A78BFA']
    ):
        with col:
            st.markdown(stat_card(val, lbl, clr), unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # ── Model comparison chart ──
    comp_path = os.path.join(SCREENS, 'model_comparison.png')
    comp_b64  = load_image_b64(comp_path)

    if comp_b64:
        st.markdown("""
        <p style='font-size:11px; font-weight:600; color:#475569;
                  letter-spacing:0.07em; text-transform:uppercase;
                  margin-bottom:10px;'>
            Model Comparison — Naive Bayes vs Logistic Regression
        </p>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#161B27; border:1px solid #252D3D;
                    border-radius:12px; padding:16px; margin-bottom:20px;'>
            <img src='{comp_b64}'
                 style='width:100%; border-radius:6px; display:block;'>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Run `python src/train.py` to generate charts → screenshots/")

    # ── Confusion matrices ──
    st.markdown("""
    <p style='font-size:11px; font-weight:600; color:#475569;
              letter-spacing:0.07em; text-transform:uppercase;
              margin-bottom:10px;'>
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
                <div style='background:#161B27; border:1px solid #252D3D;
                            border-radius:12px; padding:14px;'>
                    <p style='font-size:12px; font-weight:600; color:#475569;
                              margin:0 0 10px;'>{title}</p>
                    <img src='{b64}'
                         style='width:100%; border-radius:6px; display:block;'>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"{fname} not found in screenshots/")

    # ── Definitions ──
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:11px; font-weight:600; color:#475569;
              letter-spacing:0.07em; text-transform:uppercase;
              margin-bottom:12px;'>
        Metric Definitions
    </p>
    """, unsafe_allow_html=True)

    defs = [
        ("Accuracy",   "#4F8EF7",
         "Proportion of all messages classified correctly by the model."),
        ("Precision",  "#31C48D",
         "Of all messages flagged as spam, what fraction were actually spam. "
         "High precision = fewer false alarms."),
        ("Recall",     "#FBBF24",
         "Of all actual spam messages in the dataset, what fraction did the "
         "model successfully detect. High recall = fewer missed spam."),
        ("F1 Score",   "#A78BFA",
         "Harmonic mean of precision and recall. Best single metric when "
         "classes are imbalanced (more ham than spam)."),
    ]

    for term, color, defn in defs:
        st.markdown(f"""
        <div style='display:flex; gap:14px; padding:12px 0;
                    border-bottom:1px solid #1A2030;
                    align-items:flex-start;'>
            <span style='min-width:80px; font-size:13px; font-weight:600;
                         color:{color}; padding-top:1px;'>{term}</span>
            <span style='font-size:13px; color:#475569;
                         line-height:1.6;'>{defn}</span>
        </div>
        """, unsafe_allow_html=True)
