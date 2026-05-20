# ============================================================
# FILE: app/app.py
# PURPOSE: Streamlit frontend UI
#          User types message → sends to Flask API →
#          displays prediction result with confidence + charts
# MEMBER: Member 4
# REQUIRES: Flask API running on http://localhost:5000
# ============================================================


# --- IMPORTS ---

import streamlit as st
# streamlit: turns Python scripts into interactive web apps
# Every st.something() call renders a UI element

import requests
# requests: sends HTTP requests from Python
# We use it to call our Flask API

import json
# json: to handle JSON data
# requests already handles most JSON but we import for safety

import pandas as pd
# pandas: for displaying data tables in the UI

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# matplotlib: for drawing charts inside Streamlit

import matplotlib.patches as mpatches
# mpatches: for creating colored legend patches in charts

import os
# os: for building file paths to screenshots/

import time
# time: for adding small delays (makes UI feel smoother)


# ============================================================
# PAGE CONFIGURATION
# Must be the FIRST Streamlit command in the file
# ============================================================

st.set_page_config(
    page_title="Spam Detector",
    # Title shown in browser tab

    page_icon="🛡️",
    # Icon shown in browser tab

    layout="wide",
    # "wide" uses full browser width
    # Better for dashboards and demo presentations

    initial_sidebar_state="expanded"
    # Sidebar open by default
)


# ============================================================
# CUSTOM CSS STYLING
# Makes the app look professional for the presentation
# ============================================================

st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #0e1117;
    }

    /* Spam result card - red */
    .spam-box {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
        border: 2px solid #ff4b4b;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
    }

    /* Ham result card - green */
    .ham-box {
        background: linear-gradient(135deg, #00c85322, #00c85344);
        border: 2px solid #00c853;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
    }

    /* Result label text */
    .result-label {
        font-size: 48px;
        font-weight: 800;
        margin: 0;
    }

    /* Confidence text */
    .confidence-text {
        font-size: 20px;
        margin-top: 8px;
        opacity: 0.9;
    }

    /* Metric cards */
    .metric-card {
        background: #1e2330;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #2d3347;
    }

    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 12px;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)
# unsafe_allow_html=True: allows raw HTML/CSS
# Streamlit sanitizes HTML by default — this bypasses it
# Safe to use for our own CSS


# ============================================================
# CONSTANTS
# ============================================================

API_URL = "http://localhost:5000"
# Flask API base URL
# All requests go here


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_api_status():
    """
    Checks if Flask API is running.
    Returns True if healthy, False if not reachable.
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        # timeout=3: give up after 3 seconds if no response
        data = response.json()
        return data.get('status') == 'healthy'
    except:
        return False
        # If ANY error occurs (connection refused, timeout, etc.)
        # return False — API is not running


def predict_message(message):
    """
    Sends message to Flask API and returns prediction result.
    Returns dict with label, confidence, cleaned.
    Returns None if request fails.
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"message": message},
            # json= automatically sets Content-Type header
            # and converts dict to JSON string
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.ConnectionError:
        return None
        # ConnectionError = Flask server is not running


def make_confidence_chart(label, confidence):
    """
    Creates a horizontal bar chart showing confidence score.
    Returns a matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 1.5))
    fig.patch.set_facecolor('#0e1117')
    # Set figure background to match Streamlit dark theme

    ax.set_facecolor('#1e2330')
    # Set plot area background

    color = '#ff4b4b' if label == 'spam' else '#00c853'
    # Red for spam, green for ham

    # Draw the confidence bar
    ax.barh(0, confidence, color=color, height=0.5, alpha=0.85)
    # barh = horizontal bar chart
    # 0    = y position
    # confidence = bar length (0 to 100)
    # height=0.5 = thin bar

    # Draw the remaining bar (gray background)
    ax.barh(0, 100 - confidence,
            left=confidence,
            # left= means start this bar where previous one ended
            color='#2d3347', height=0.5, alpha=0.85)

    # Add percentage label at end of bar
    ax.text(confidence - 2, 0,
            f'{confidence:.1f}%',
            va='center', ha='right',
            color='white', fontsize=14, fontweight='bold')

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    # Turn off all axes, ticks, labels

    plt.tight_layout(pad=0)
    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/spam.png", width=80)
    st.title("Spam Detector")
    st.caption("NLP + Machine Learning Project")

    st.divider()

    # API Status indicator
    st.subheader("🔌 API Status")

    api_ok = check_api_status()
    # Check if Flask is running right now

    if api_ok:
        st.success("Flask API: Online ✅")
    else:
        st.error("Flask API: Offline ❌")
        st.warning(
            "Start the API first:\n\n"
            "```\npython app/flask_api.py\n```"
        )

    st.divider()

    # Project info
    st.subheader("📋 Project Info")
    st.markdown("""
    - **Domain**: NLP + ML
    - **Model**: Naive Bayes / LR
    - **Dataset**: UCI SMS Spam
    - **Accuracy**: ~98%
    - **Stack**: Python, sklearn
    """)

    st.divider()

    # Tech stack
    st.subheader("🛠️ Tech Stack")
    st.markdown("""
    `Python` `scikit-learn` `NLTK`
    `Flask` `Streamlit` `pandas`
    `TF-IDF` `Pickle`
    """)


# ============================================================
# MAIN PAGE
# ============================================================

# ── Header ──
st.markdown(
    "<h1 style='text-align:center; color:#e0e0e0;'>"
    "🛡️ Spam SMS / Email Detector"
    "</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#888; font-size:18px;'>"
    "Powered by TF-IDF + Machine Learning"
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# TABS — Main layout
# Tab 1: Single prediction (live demo)
# Tab 2: Batch testing
# Tab 3: Model performance charts
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🔍 Single Prediction",
    "📋 Batch Test",
    "📊 Model Performance"
])


# ============================================================
# TAB 1 — Single Prediction (This is your live demo tab)
# ============================================================

with tab1:

    st.markdown("### Type a message to check if it's spam:")

    # Sample messages buttons for quick demo
    st.markdown("**Quick test samples:**")

    col1, col2, col3, col4 = st.columns(4)
    # Create 4 equal columns for sample buttons

    sample_spam1 = "WINNER!! Claim your FREE £1000 prize now! Call 08712300150"
    sample_spam2 = "FREE entry! Win FA Cup tickets! Text FA to 87121 now"
    sample_ham1  = "Hey, are you coming to the meeting at 3pm tomorrow?"
    sample_ham2  = "Can you pick up groceries on your way home please?"

    with col1:
        if st.button("🔴 Spam Sample 1", use_container_width=True):
            st.session_state['input_message'] = sample_spam1
            # session_state stores values between reruns
            # When button clicked, fill the text area

    with col2:
        if st.button("🔴 Spam Sample 2", use_container_width=True):
            st.session_state['input_message'] = sample_spam2

    with col3:
        if st.button("🟢 Ham Sample 1", use_container_width=True):
            st.session_state['input_message'] = sample_ham1

    with col4:
        if st.button("🟢 Ham Sample 2", use_container_width=True):
            st.session_state['input_message'] = sample_ham2

    st.markdown("")

    # Text input area
    message_input = st.text_area(
        label="Enter your message here:",
        value=st.session_state.get('input_message', ''),
        # Pre-fill with sample if button was clicked
        height=120,
        placeholder="Type or paste an SMS/email message here...",
        key="message_area"
    )

    # Predict button
    predict_btn = st.button(
        "🔍 Analyze Message",
        type="primary",
        # type="primary" makes it blue and prominent
        use_container_width=True
    )

    # ── Run prediction when button clicked ──
    if predict_btn:

        if not message_input.strip():
            st.warning("⚠️ Please enter a message first.")

        elif not api_ok:
            st.error(
                "❌ Flask API is not running. "
                "Open a terminal and run: `python app/flask_api.py`"
            )

        else:
            # Show spinner while waiting for API response
            with st.spinner("Analyzing message..."):
                time.sleep(0.3)
                # Small delay makes it feel like it's working
                result = predict_message(message_input)

            if result is None:
                st.error("❌ Could not connect to API. Is Flask running?")

            else:
                label      = result['label']
                confidence = result['confidence']
                cleaned    = result['cleaned']

                st.divider()

                # ── Result Display ──
                if label == 'spam':
                    st.markdown(f"""
                    <div class="spam-box">
                        <p class="result-label" style="color:#ff4b4b;">
                            🚨 SPAM DETECTED
                        </p>
                        <p class="confidence-text" style="color:#ffaaaa;">
                            Confidence: {confidence}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown(f"""
                    <div class="ham-box">
                        <p class="result-label" style="color:#00c853;">
                            ✅ LEGITIMATE (HAM)
                        </p>
                        <p class="confidence-text" style="color:#aaffaa;">
                            Confidence: {confidence}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Confidence Bar Chart ──
                st.markdown("**Confidence Score:**")
                fig = make_confidence_chart(label, confidence)
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # ── Details expander ──
                with st.expander("🔬 View Processing Details"):
                    st.markdown("**Original message:**")
                    st.info(message_input)

                    st.markdown("**After cleaning (what model saw):**")
                    st.code(cleaned, language=None)

                    st.markdown("**What happened step by step:**")
                    st.markdown("""
                    1. Converted to lowercase
                    2. Removed URLs, phone numbers, emails
                    3. Removed punctuation
                    4. Removed stopwords (the, is, a, an...)
                    5. Applied stemming (running → run)
                    6. Converted to TF-IDF numbers
                    7. Fed into trained ML model
                    8. Model returned prediction + confidence
                    """)


# ============================================================
# TAB 2 — Batch Testing
# ============================================================

with tab2:

    st.markdown("### Test multiple messages at once")
    st.caption(
        "Enter one message per line. "
        "Maximum 20 messages."
    )

    batch_input = st.text_area(
        label="Enter messages (one per line):",
        height=200,
        placeholder=(
            "WINNER!! You have been selected for a cash prize!\n"
            "Hey, are you coming to class tomorrow?\n"
            "FREE entry! Win FA Cup tickets now!\n"
            "Can we reschedule our meeting to Friday?"
        )
    )

    batch_btn = st.button(
        "🔍 Analyze All Messages",
        type="primary",
        use_container_width=True,
        key="batch_btn"
    )

    if batch_btn:

        if not batch_input.strip():
            st.warning("⚠️ Please enter at least one message.")

        elif not api_ok:
            st.error("❌ Flask API is not running.")

        else:
            messages = [
                m.strip()
                for m in batch_input.strip().split('\n')
                if m.strip()
            ]
            # Split by newline, remove empty lines and whitespace

            if len(messages) > 20:
                st.warning("⚠️ Maximum 20 messages. Using first 20.")
                messages = messages[:20]

            with st.spinner(f"Analyzing {len(messages)} messages..."):
                results = []
                for msg in messages:
                    result = predict_message(msg)
                    if result:
                        results.append({
                            'Message'   : msg[:60] + ('...' if len(msg) > 60 else ''),
                            'Prediction': '🔴 SPAM' if result['label'] == 'spam' else '🟢 HAM',
                            'Confidence': f"{result['confidence']}%",
                            'Label'     : result['label']
                        })

            if results:
                spam_count = sum(1 for r in results if r['Label'] == 'spam')
                ham_count  = len(results) - spam_count

                # Summary metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Total Messages", len(results))
                with m2:
                    st.metric("Spam Found", spam_count,
                              delta=f"{spam_count/len(results)*100:.0f}%")
                with m3:
                    st.metric("Ham Found", ham_count,
                              delta=f"{ham_count/len(results)*100:.0f}%")

                st.divider()

                # Results table
                df_results = pd.DataFrame(results)
                df_display = df_results[['Message', 'Prediction', 'Confidence']]

                st.dataframe(
                    df_display,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# TAB 3 — Model Performance Charts
# ============================================================

with tab3:

    st.markdown("### Model Evaluation Results")

    screenshots_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'screenshots'
    )
    # Build path to screenshots/ folder

    # ── Model Comparison Chart ──
    comparison_path = os.path.join(
        screenshots_dir, 'model_comparison.png'
    )

    if os.path.exists(comparison_path):
        st.markdown("#### Naive Bayes vs Logistic Regression")
        st.image(comparison_path, use_column_width=True)
    else:
        st.warning("model_comparison.png not found in screenshots/")

    st.divider()

    # ── Confusion Matrices ──
    st.markdown("#### Confusion Matrices")

    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        nb_path = os.path.join(
            screenshots_dir, 'confusion_naive_bayes.png'
        )
        if os.path.exists(nb_path):
            st.markdown("**Naive Bayes**")
            st.image(nb_path, use_column_width=True)
        else:
            st.warning("confusion_naive_bayes.png not found")

    with cm_col2:
        lr_path = os.path.join(
            screenshots_dir, 'confusion_logistic_regression.png'
        )
        if os.path.exists(lr_path):
            st.markdown("**Logistic Regression**")
            st.image(lr_path, use_column_width=True)
        else:
            st.warning("confusion_logistic_regression.png not found")

    st.divider()

    # ── Metrics explanation ──
    st.markdown("#### Understanding the Metrics")

    e1, e2 = st.columns(2)

    with e1:
        st.info("""
        **Accuracy** — Overall correct predictions
        ~98% means 98 out of 100 messages classified correctly

        **Precision** — Of predicted spam, how many were spam?
        High precision = fewer false alarms
        """)

    with e2:
        st.info("""
        **Recall** — Of actual spam, how many did we catch?
        High recall = fewer spam messages slipping through

        **F1 Score** — Balance between precision and recall
        Best single metric to compare models
        """)