import os
import textwrap

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_BASE_URL}/predict"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SESSION STATE
# ============================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ============================================================
# HTML HELPER
# ============================================================

def render_html(html: str) -> None:
    st.html(textwrap.dedent(html).strip())


# ============================================================
# THEME TOKENS
# ============================================================

dark_mode = st.session_state.dark_mode

if dark_mode:
    # ── Dark theme ──────────────────────────────────────────
    BG               = "#050719"
    BG_RADIAL_1      = "rgba(129, 53, 255, 0.15)"
    BG_RADIAL_2      = "rgba(42, 120, 255, 0.12)"

    TEXT             = "#f7f7ff"
    MUTED            = "rgba(247, 247, 255, 0.55)"
    SUBMUTED         = "rgba(247, 247, 255, 0.38)"

    CARD             = "rgba(13, 16, 48, 0.88)"
    CARD_BORDER      = "rgba(130, 75, 255, 0.45)"

    INPUT_BG         = "#0b0e2b"
    INPUT_BORDER     = "#3e42a0"
    INPUT_FOCUS      = "#8b5cf6"

    NAV_DIVIDER      = "linear-gradient(90deg, transparent, #a53dff, #3b82f6, transparent)"

    BRAND_GRADIENT   = "linear-gradient(90deg, #ffffff, #a855f7, #5b8cff)"
    HERO_GRADIENT    = "linear-gradient(90deg, #e1d5ff, #b17aff, #8b5cf6, #4d7cff)"
    BTN_GRADIENT     = "linear-gradient(90deg, #a83cff, #7048ff, #2563eb)"
    BTN_SHADOW       = "rgba(91, 65, 255, 0.27)"
    BTN_SHADOW_HOVER = "rgba(91, 65, 255, 0.42)"

    SECTION_ICON_COLOR = "#ffffff"
    SECTION_TEXT_COLOR = "#ffffff"

    DISCLAIMER_BORDER = "rgba(245, 158, 11, 0.6)"
    DISCLAIMER_BG     = "rgba(245, 158, 11, 0.055)"
    DISCLAIMER_STRONG = "#f59e0b"

    FOOTER_BORDER    = "rgba(130, 75, 255, 0.45)"
    FOOTER_TEXT      = "#f7f7ff"
    FOOTER_MUTED     = "rgba(247, 247, 255, 0.45)"

    PROGRESS_EMPTY   = "#1e2147"

    TOGGLE_LABEL     = "☀️ Light mode"

else:
    # ── Light theme ─────────────────────────────────────────
    BG               = "#f4f5ff"
    BG_RADIAL_1      = "rgba(139, 92, 246, 0.07)"
    BG_RADIAL_2      = "rgba(59, 130, 246, 0.06)"

    TEXT             = "#12143a"
    MUTED            = "#4b5099"
    SUBMUTED         = "#7b82b8"

    CARD             = "rgba(255, 255, 255, 0.97)"
    CARD_BORDER      = "rgba(108, 77, 255, 0.22)"

    INPUT_BG         = "#ffffff"
    INPUT_BORDER     = "#c8cde8"
    INPUT_FOCUS      = "#7c3aed"

    NAV_DIVIDER      = "linear-gradient(90deg, transparent, #9333ea, #3b82f6, transparent)"

    BRAND_GRADIENT   = "linear-gradient(90deg, #27213f, #7048ff, #287ff0)"
    HERO_GRADIENT    = "linear-gradient(90deg, #5b21b6, #7c3aed, #4f46e5, #2563eb)"
    BTN_GRADIENT     = "linear-gradient(90deg, #7c3aed, #5b21b6, #1d4ed8)"
    BTN_SHADOW       = "rgba(91, 65, 255, 0.20)"
    BTN_SHADOW_HOVER = "rgba(91, 65, 255, 0.32)"

    SECTION_ICON_COLOR = "#12143a"
    SECTION_TEXT_COLOR = "#12143a"

    DISCLAIMER_BORDER = "rgba(217, 119, 6, 0.55)"
    DISCLAIMER_BG     = "rgba(245, 158, 11, 0.06)"
    DISCLAIMER_STRONG = "#b45309"

    FOOTER_BORDER    = "rgba(108, 77, 255, 0.22)"
    FOOTER_TEXT      = "#12143a"
    FOOTER_MUTED     = "#6b7280"

    PROGRESS_EMPTY   = "#e0e3f5"

    TOGGLE_LABEL     = "🌙 Dark mode"


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    f"""
<style>

/* ── Google Font ──────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Reset & base ─────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

/* ── App background ───────────────────────────────────────── */
.stApp {{
    background:
        radial-gradient(circle at 10% 5%,  {BG_RADIAL_1}, transparent 28%),
        radial-gradient(circle at 90% 10%, {BG_RADIAL_2}, transparent 30%),
        {BG};
    color: {TEXT};
}}

/* ── Main container ───────────────────────────────────────── */
.main .block-container {{
    max-width: 1300px;
    padding-top:    0 !important;
    padding-bottom: 2rem !important;
    padding-left:   5% !important;
    padding-right:  5% !important;
}}

/* ── Hide Streamlit chrome ────────────────────────────────── */
#MainMenu                        {{ visibility: hidden; }}
footer                           {{ visibility: hidden; }}
header[data-testid="stHeader"]   {{ background: transparent; }}
[data-testid="stDecoration"]     {{ display: none; }}

/* ── Remove top dead space ────────────────────────────────── */
[data-testid="stAppViewContainer"] > section > div:first-child {{
    padding-top: 0.5rem !important;
}}

/* ── TOP NAV ──────────────────────────────────────────────── */
.top-nav {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0 0.5rem 0;
}}

.brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.35rem;
    font-weight: 900;
    letter-spacing: -0.4px;
}}

.brand-icon {{ font-size: 1.65rem; line-height: 1; }}

.brand-gradient {{
    background: {BRAND_GRADIENT};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

/* ── NAV DIVIDER ──────────────────────────────────────────── */
.nav-divider {{
    height: 1.5px;
    margin-bottom: 0.5rem;
    background: {NAV_DIVIDER};
    opacity: 0.7;
}}

/* ── HERO ─────────────────────────────────────────────────── */
.hero {{
    text-align: center;
    padding: 1.6rem 0 1.2rem;
}}

.hero-title {{
    margin: 0 0 0.15rem;
    font-size: clamp(3.2rem, 6vw, 5rem);
    line-height: 1.0;
    font-weight: 900;
    letter-spacing: -3.5px;
    background: {HERO_GRADIENT};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.hero-subtitle {{
    margin: 0 0 0.5rem;
    font-size: 1.25rem;
    font-weight: 800;
    color: {TEXT};
    letter-spacing: -0.2px;
}}

.hero-description {{
    max-width: 660px;
    margin: 0 auto;
    color: {MUTED};
    font-size: 0.97rem;
    line-height: 1.6;
}}

/* ── SECTION HEADING ──────────────────────────────────────── */
.section-heading {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.4rem 0 0.7rem;
    font-size: 1.15rem;
    font-weight: 800;
    color: {SECTION_TEXT_COLOR};
    letter-spacing: -0.2px;
}}

.section-icon {{ font-size: 1.2rem; }}

/* ── ANALYSIS CARD ────────────────────────────────────────── */
.analysis-card {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    border-radius: 16px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 12px 36px rgba(0,0,0,0.10);
}}

.analysis-description {{
    color: {MUTED};
    font-size: 0.9rem;
    line-height: 1.5;
}}

/* ── TEXT AREA ────────────────────────────────────────────── */
textarea {{
    background:     {INPUT_BG}     !important;
    color:          {TEXT}         !important;
    border:         1px solid {INPUT_BORDER} !important;
    border-radius:  13px           !important;
    font-size:      0.95rem        !important;
    line-height:    1.55           !important;
    caret-color:    {INPUT_FOCUS}  !important;
}}
textarea:focus {{
    border-color: {INPUT_FOCUS} !important;
    box-shadow: 0 0 0 2px {INPUT_FOCUS}30, 0 0 18px {INPUT_FOCUS}22 !important;
    outline: none !important;
}}

/* ── CHAR COUNTER ─────────────────────────────────────────── */
[data-testid="InputInstructions"] {{
    color: {SUBMUTED} !important;
    font-size: 0.78rem !important;
}}

/* ── PRIMARY BUTTON ───────────────────────────────────────── */
.stButton > button {{
    width: 100%;
    min-height: 46px;
    border: none !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.1px;
    background: {BTN_GRADIENT} !important;
    box-shadow: 0 7px 22px {BTN_SHADOW} !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 12px 30px {BTN_SHADOW_HOVER} !important;
}}
.stButton > button:active {{
    transform: translateY(0);
}}

/* ── RESULT CARDS ─────────────────────────────────────────── */
.result-card {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    border-radius: 18px;
    padding: 1.3rem 1.4rem;
    box-shadow: 0 14px 40px rgba(0,0,0,0.10);
    height: 100%;
}}

/* ── SAMPLE CARDS ─────────────────────────────────────────── */
.sample-card {{
    min-height: 100px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    border-radius: 14px;
    padding: 0.85rem 0.9rem 0.6rem;
    color: {TEXT};
    box-shadow: 0 8px 24px rgba(0,0,0,0.07);
    margin-bottom: 0.4rem;
}}

.sample-icon  {{ font-size: 1.55rem; display: block; margin-bottom: 0.35rem; }}
.sample-title {{ font-size: 0.85rem; font-weight: 700; line-height: 1.35; color: {TEXT}; }}

/* ── CAPTION ──────────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] p {{
    color: {SUBMUTED} !important;
    font-size: 0.8rem !important;
}}

/* ── DISCLAIMER ───────────────────────────────────────────── */
.disclaimer {{
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 0.95rem 1.1rem;
    border: 1px solid {DISCLAIMER_BORDER};
    border-radius: 14px;
    background: {DISCLAIMER_BG};
    font-size: 0.87rem;
    line-height: 1.55;
    color: {MUTED};
}}
.disclaimer-icon {{ font-size: 1.35rem; flex-shrink: 0; padding-top: 0.05rem; }}
.disclaimer strong {{ color: {DISCLAIMER_STRONG}; }}

/* ── FOOTER ───────────────────────────────────────────────── */
.footer {{
    border-top: 1px solid {FOOTER_BORDER};
    margin-top: 1.5rem;
    padding-top: 1rem;
}}
.footer-layout {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
    flex-wrap: wrap;
}}
.footer-brand {{
    font-size: 1.05rem;
    font-weight: 900;
    color: {FOOTER_TEXT};
}}
.footer-description {{
    margin-top: 0.15rem;
    color: {FOOTER_MUTED};
    font-size: 0.76rem;
}}
.tech-stack {{
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.65rem;
    color: {FOOTER_MUTED};
    font-size: 0.79rem;
}}
.tech-item {{
    display: flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
}}
.tech-dot {{ opacity: 0.45; }}
.footer-copyright {{
    text-align: center;
    margin-top: 0.65rem;
    color: {FOOTER_MUTED};
    font-size: 0.72rem;
}}

/* ── TOGGLE ───────────────────────────────────────────────── */
[data-testid="stToggle"] {{
    justify-content: flex-end;
}}
[data-testid="stToggle"] label {{
    color: {MUTED} !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
}}

/* ── SPINNER ──────────────────────────────────────────────── */
[data-testid="stSpinner"] p {{
    color: {MUTED} !important;
}}

/* ── WARNING / ERROR ──────────────────────────────────────── */
[data-testid="stAlert"] {{
    border-radius: 12px !important;
}}

/* ── RESPONSIVE ───────────────────────────────────────────── */
@media (max-width: 900px) {{
    .main .block-container {{
        padding-left:  3% !important;
        padding-right: 3% !important;
    }}
    .hero-title    {{ font-size: 3rem; letter-spacing: -2px; }}
    .footer-layout {{ flex-direction: column; align-items: flex-start; }}
}}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# TOP NAV
# ============================================================

nav_col1, nav_col2 = st.columns([8, 2], vertical_alignment="center")

with nav_col1:
    render_html(
        """
        <div class="top-nav">
            <div class="brand">
                <span class="brand-icon">🔎</span>
                <span class="brand-gradient">TruthLens AI</span>
            </div>
        </div>
        """
    )

with nav_col2:
    new_value = st.toggle(
        TOGGLE_LABEL,
        value=dark_mode,
        key="theme_toggle",
    )
    if new_value != dark_mode:
        st.session_state.dark_mode = new_value
        st.rerun()

render_html('<div class="nav-divider"></div>')


# ============================================================
# HERO
# ============================================================

render_html(
    f"""
    <div class="hero">
        <div class="hero-title">TruthLens AI</div>
        <div class="hero-subtitle">AI-Powered Fake News Detection</div>
        <div class="hero-description">
            Analyze news headlines and articles using a fine-tuned natural language processing model
            and get an AI-based Fake or Real prediction.
        </div>
    </div>
    """
)


# ============================================================
# ANALYZE SECTION HEADING
# ============================================================

render_html(
    """
    <div class="section-heading">
        <span class="section-icon">📰</span>
        Analyze a News Article
    </div>
    """
)

render_html(
    f"""
    <div class="analysis-card">
        <div class="analysis-description">
            Paste a news headline or article below.
            TruthLens AI will analyze the text and estimate
            whether it resembles Fake or Real news.
        </div>
    </div>
    """
)


# ============================================================
# NEWS INPUT
# ============================================================

news_text = st.text_area(
    "News article",
    value=st.session_state.news_text,
    height=130,
    max_chars=5000,
    placeholder="Paste a news headline or article here...",
    label_visibility="collapsed",
)

st.session_state.news_text = news_text


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button("🔍  Analyze News", use_container_width=True):
    if not news_text.strip():
        st.warning("Please enter a news headline or article.")
    else:
        with st.spinner("TruthLens AI is analyzing the article..."):
            try:
                response = requests.post(
                    PREDICT_URL,
                    json={"text": news_text},
                    timeout=120,
                )
                response.raise_for_status()
                st.session_state.prediction_result = response.json()

            except requests.exceptions.ConnectionError:
                st.error(
                    "Unable to connect to the FastAPI backend. "
                    "Make sure the API is running."
                )
            except requests.exceptions.Timeout:
                st.error("The prediction request timed out.")
            except requests.exceptions.HTTPError as error:
                st.error(f"The API returned an error: {error}")
            except Exception as error:
                st.error(f"Unexpected error: {error}")


# ============================================================
# RESULT
# ============================================================

result = st.session_state.prediction_result

if result:
    prediction   = result["prediction"]
    confidence   = float(result["confidence"])
    probabilities = result["probabilities"]
    fake_prob    = float(probabilities["fake"])
    real_prob    = float(probabilities["real"])

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if prediction.lower() == "fake":
        icon         = "⚠️"
        result_color = "#ff536b"
        result_text  = "LIKELY FAKE"
    else:
        icon         = "✅"
        result_color = "#22d487"
        result_text  = "LIKELY REAL"

    rc1, rc2 = st.columns(2)

    with rc1:
        render_html(
            f"""
            <div class="result-card">
                <div style="color:{result_color}; font-size:1.5rem; font-weight:900; margin-bottom:0.5rem;">
                    {icon}&nbsp;{result_text}
                </div>
                <div style="color:{MUTED}; font-size:0.88rem; margin-bottom:0.2rem;">
                    Model Confidence
                </div>
                <div style="color:{TEXT}; font-size:2.6rem; font-weight:900; letter-spacing:-1.5px; line-height:1.1;">
                    {confidence * 100:.2f}%
                </div>
            </div>
            """
        )

    with rc2:
        render_html(
            f"""
            <div class="result-card">
                <div style="color:{TEXT}; font-size:1rem; font-weight:800; margin-bottom:0.85rem;">
                    Prediction Probabilities
                </div>

                <!-- Fake bar -->
                <div style="display:flex; justify-content:space-between;
                            color:{MUTED}; font-size:0.84rem; margin-bottom:0.3rem;">
                    <span>Fake</span>
                    <span>{fake_prob * 100:.2f}%</span>
                </div>
                <div style="height:7px; border-radius:999px; overflow:hidden; background:{PROGRESS_EMPTY}; margin-bottom:0.9rem;">
                    <div style="height:100%; width:{fake_prob*100}%;
                                background:linear-gradient(90deg,#ff536b,#ff8fa3);
                                border-radius:999px;"></div>
                </div>

                <!-- Real bar -->
                <div style="display:flex; justify-content:space-between;
                            color:{MUTED}; font-size:0.84rem; margin-bottom:0.3rem;">
                    <span>Real</span>
                    <span>{real_prob * 100:.2f}%</span>
                </div>
                <div style="height:7px; border-radius:999px; overflow:hidden; background:{PROGRESS_EMPTY};">
                    <div style="height:100%; width:{real_prob*100}%;
                                background:linear-gradient(90deg,#22d487,#4ade80);
                                border-radius:999px;"></div>
                </div>
            </div>
            """
        )


# ============================================================
# SAMPLE NEWS
# ============================================================

render_html(
    """
    <div class="section-heading">
        <span class="section-icon">🧪</span>
        Try Sample News
    </div>
    """
)

samples = [
    (
        "💧",
        "NASA confirms water on Mars",
        "NASA has confirmed that liquid water is currently flowing across the surface of Mars.",
    ),
    (
        "🍫",
        "Chocolate improves memory",
        "Scientists have discovered that eating chocolate every day dramatically improves human memory.",
    ),
    (
        "🌍",
        "Climate change will end by 2030",
        "Experts say climate change will completely end by 2030 because of recent technological advances.",
    ),
    (
        "📱",
        "Free phones for all citizens",
        "The government has announced a nationwide program to provide a free smartphone to every citizen.",
    ),
]

sample_cols = st.columns(4)

for i, (icon, title, sample_text) in enumerate(samples):
    with sample_cols[i]:
        render_html(
            f"""
            <div class="sample-card">
                <span class="sample-icon">{icon}</span>
                <div class="sample-title">{title}</div>
            </div>
            """
        )
        if st.button("Analyze", key=f"sample_btn_{i}", use_container_width=True):
            st.session_state.news_text = sample_text
            try:
                response = requests.post(
                    PREDICT_URL,
                    json={"text": sample_text},
                    timeout=120,
                )
                response.raise_for_status()
                st.session_state.prediction_result = response.json()
                st.rerun()
            except Exception as error:
                st.error(f"Unable to analyze sample: {error}")

st.caption("Click Analyze on a sample to try the application.")


# ============================================================
# DISCLAIMER
# ============================================================

render_html(
    """
    <div class="section-heading">
        <span class="section-icon">🛡️</span>
        Before You Trust the Result
    </div>
    """
)

render_html(
    f"""
    <div class="disclaimer">
        <span class="disclaimer-icon">🛡️</span>
        <div>
            <strong>Important:</strong>
            TruthLens AI is an AI-based prediction system, not a definitive fact-checking service.
            A high confidence score does not guarantee that an article is factually true or false.
            Important claims should always be verified using reliable and independent sources.
        </div>
    </div>
    """
)


# ============================================================
# FOOTER
# ============================================================

render_html(
    f"""
    <div class="footer">
        <div class="footer-layout">

            <div>
                <div class="footer-brand">🔎 TruthLens AI</div>
                <div class="footer-description">
                    AI-powered news analysis using Machine Learning and DistilBERT
                </div>
            </div>

            <div class="tech-stack">
                <div class="tech-item">🐍 Python</div>
                <span class="tech-dot">•</span>
                <div class="tech-item">⚡ FastAPI</div>
                <span class="tech-dot">•</span>
                <div class="tech-item">🎈 Streamlit</div>
                <span class="tech-dot">•</span>
                <div class="tech-item">🤗 DistilBERT</div>
                <span class="tech-dot">•</span>
                <div class="tech-item">🐳 Docker</div>
            </div>

        </div>

        <div class="footer-copyright">
            © 2026 TruthLens AI. All rights reserved.
        </div>
    </div>
    """
)