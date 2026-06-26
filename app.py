import streamlit as st
import pickle
from preprocess import clean_text
from recommender import MovieRecommender
from tmdb_client import get_movie_details
from database import init_db, add_review, get_sentiment_counts, get_recent_reviews

st.set_page_config(
    page_title="CineSense",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 20% 0%, #1b1035 0%, #0f0a1f 45%, #0a0715 100%);
        color: #e6e3f0;
    }
    .main .block-container { padding-top: 2.5rem; max-width: 1100px; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Hero ───────────────────────────────────────── */
    .hero-title, p.hero-title, div[data-testid="stMarkdownContainer"] p.hero-title {
        font-size: 52px !important; font-weight: 800 !important; color: #ffffff !important;
        margin: 0 !important; letter-spacing: -1px !important; line-height: 1.1 !important;
    }
    .hero-sub, p.hero-sub, div[data-testid="stMarkdownContainer"] p.hero-sub {
        color: #a89fc4 !important; font-size: 16px !important; margin-top: 10px !important; margin-bottom: 24px !important;
    }

    /* ── Tabs ───────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        color: #a89fc4 !important;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(160deg, rgba(192,132,252,0.18), rgba(255,110,199,0.08)) !important;
        color: #ffffff !important;
    }

    /* ── Stats row ──────────────────────────────────── */
    .metric-row { display: flex; gap: 12px; margin-bottom: 24px; }
    .metric-box {
        flex: 1;
        background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px; padding: 16px 10px; text-align: center;
        backdrop-filter: blur(6px);
    }
    .metric-num {
        font-size: 21px; font-weight: 700;
        background: linear-gradient(90deg, #ff6ec7, #ff9d6c);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .metric-lbl { font-size: 11.5px; color: #8b81a8; margin: 4px 0 0; }

    /* ── Text area ──────────────────────────────────── */
    .stTextArea textarea, .stTextInput input {
        background-color: #1a1530 !important;
        background: #1a1530 !important;
        border: 1.5px solid rgba(255,255,255,0.15) !important;
        border-radius: 14px !important;
        color: #f5f3fb !important;
        -webkit-text-fill-color: #f5f3fb !important;
        caret-color: #c084fc !important;
        font-size: 15px !important;
        padding: 14px !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #c084fc !important;
        box-shadow: 0 0 0 3px rgba(192,132,252,0.15) !important;
        background-color: #1a1530 !important;
    }
    .stTextArea textarea::placeholder, .stTextInput input::placeholder { color: #8a81a3 !important; opacity: 1 !important; }
    .stTextArea > div, .stTextArea > div > div { background-color: transparent !important; background: transparent !important; }

    /* ── Buttons ────────────────────────────────────── */
    .stButton button[kind="primary"] {
        background: linear-gradient(90deg, #ff6ec7, #c084fc) !important;
        border: none !important; border-radius: 12px !important;
        font-weight: 600 !important; padding: 0.65rem 1rem !important;
        box-shadow: 0 4px 18px rgba(192,132,252,0.35) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 22px rgba(192,132,252,0.5) !important;
    }
    .stButton button[kind="secondary"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1.5px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important; color: #d8d3eb !important;
        font-weight: 500 !important;
    }
    .stButton button[kind="secondary"]:hover {
        border-color: rgba(255,255,255,0.3) !important;
        background: rgba(255,255,255,0.08) !important;
    }
    .stButton button { border-radius: 10px !important; }

    /* ── Mood chips ─────────────────────────────────── */
    div[data-testid="column"] .stButton button {
        width: 100%;
    }

    /* ── Result cards ───────────────────────────────── */
    .result-card-pos {
        background: linear-gradient(135deg, rgba(34,197,94,0.16), rgba(34,197,94,0.04));
        border: 1.5px solid rgba(74,222,128,0.35);
        border-radius: 16px; padding: 22px 26px;
        display: flex; align-items: center; gap: 18px;
        animation: fadeIn 0.35s ease;
    }
    .result-card-neg {
        background: linear-gradient(135deg, rgba(239,68,68,0.16), rgba(239,68,68,0.04));
        border: 1.5px solid rgba(248,113,113,0.35);
        border-radius: 16px; padding: 22px 26px;
        display: flex; align-items: center; gap: 18px;
        animation: fadeIn 0.35s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-emoji {
        font-size: 38px; line-height: 1; width: 56px; height: 56px;
        display: flex; align-items: center; justify-content: center;
        background: rgba(255,255,255,0.06); border-radius: 50%;
    }
    .result-title-pos { font-size: 19px; font-weight: 700; color: #4ade80; margin: 0 0 4px; }
    .result-title-neg { font-size: 19px; font-weight: 700; color: #f87171; margin: 0 0 4px; }
    .result-conf { font-size: 13px; margin: 0 0 10px; color: #c9c4dd; }
    .conf-track { height: 7px; background: rgba(255,255,255,0.08); border-radius: 4px; width: 230px; overflow: hidden; }
    .conf-fill-pos { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #22c55e, #4ade80); }
    .conf-fill-neg { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #ef4444, #f87171); }

    /* ── Movie cards ────────────────────────────────── */
    .movie-card {
        background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px; padding: 14px;
        height: 100%;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .movie-card:hover { border-color: rgba(192,132,252,0.4); transform: translateY(-2px); }
    .movie-title { font-size: 15px; font-weight: 700; color: #ffffff; margin: 8px 0 4px; }
    .movie-genre { font-size: 11.5px; color: #c084fc; margin: 0 0 6px; }
    .movie-rating { font-size: 12px; color: #8b81a8; }

    /* ── Tally box ──────────────────────────────────── */
    .tally-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px; padding: 16px; margin-top: 16px;
    }
    .tally-row { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 14px; }
    .tally-pos { color: #4ade80; font-weight: 600; }
    .tally-neg { color: #f87171; font-weight: 600; }

    .section-lbl {
        font-size: 11.5px; font-weight: 700; color: #8b81a8;
        text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 12px;
    }
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
    }
    div[data-testid="stExpander"] summary { color: #c9c4dd !important; }
</style>
""", unsafe_allow_html=True)

# ── Load models / engines (cached so they don't reload every rerun) ──
@st.cache_resource
def load_sentiment_model():
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

@st.cache_resource
def load_recommender():
    return MovieRecommender()

model, tfidf = load_sentiment_model()
recommender = load_recommender()
init_db()

# ── Session state ────────────────────────────────────────
defaults = {
    "review_text": "",
    "trigger_analysis": False,
    "pending_text": None,
    "selected_mood": None,
    "mood_free_text": "",
    "expanded_movie": None,
    "review_movie_title": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.pending_text is not None:
    st.session_state.review_text = st.session_state.pending_text
    st.session_state.pending_text = None

# ── Header ────────────────────────────────────────────────
st.markdown('<p class="hero-title">🎬 CineSense</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Mood-based movie discovery + AI sentiment analysis on reviews.</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎭  Mood Recommender", "📝  Sentiment Analyzer"])

# ════════════════════════════════════════════════════════════
# TAB 1 — MOOD-BASED MOVIE RECOMMENDER
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-lbl">What\'s your mood today?</p>', unsafe_allow_html=True)

    moods = recommender.available_moods()
    mood_cols = st.columns(4)
    for i, mood in enumerate(moods):
        with mood_cols[i % 4]:
            is_selected = st.session_state.selected_mood == mood
            if st.button(mood, key=f"mood_{mood}", use_container_width=True,
                         type="primary" if is_selected else "secondary"):
                st.session_state.selected_mood = mood
                st.session_state.expanded_movie = None
                st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    free_text = st.text_input(
        "Anything more specific? (optional)",
        placeholder="e.g. something like a heist movie but funny",
        label_visibility="collapsed",
        key="mood_free_text"
    )

    if st.session_state.selected_mood:
        with st.spinner("Finding movies that match your mood..."):
            results = recommender.recommend_by_mood(
                st.session_state.selected_mood,
                free_text=st.session_state.mood_free_text,
                top_n=8
            )

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown(f'<p class="section-lbl">Movies for "{st.session_state.selected_mood}" mood</p>', unsafe_allow_html=True)

        cols = st.columns(4)
        for idx, row in results.iterrows():
            with cols[idx % 4]:
                st.markdown(f"""
                <div class="movie-card">
                    <div style="font-size:32px; text-align:center;">🎬</div>
                    <p class="movie-title">{row['title']}</p>
                    <p class="movie-genre">{row['genres']}</p>
                    <p class="movie-rating">⭐ {row['rating']:.1f} · {str(row['release_date'])[:4] if row['release_date'] == row['release_date'] else 'N/A'}</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("View details", key=f"view_{idx}_{row['title']}", use_container_width=True):
                    st.session_state.expanded_movie = row['title']
                    st.rerun()

        # ── Expanded detail panel ──
        if st.session_state.expanded_movie:
            movie_row = results[results['title'] == st.session_state.expanded_movie]
            if not movie_row.empty:
                movie_row = movie_row.iloc[0]
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

                with st.container():
                    st.markdown(f"### {movie_row['title']}")
                    detail_cols = st.columns([1, 2])

                    with st.spinner("Fetching poster & streaming info..."):
                        tmdb_info = get_movie_details(
                            movie_row['title'],
                            year=str(movie_row['release_date'])[:4] if movie_row['release_date'] == movie_row['release_date'] else None
                        )

                    with detail_cols[0]:
                        if tmdb_info["poster_url"]:
                            st.image(tmdb_info["poster_url"], use_container_width=True)
                        else:
                            st.markdown("*No poster available*")

                    with detail_cols[1]:
                        st.markdown(f"**Genre:** {movie_row['genres']}")
                        st.markdown(f"**Rating:** ⭐ {movie_row['rating']:.1f}")
                        st.markdown(f"**Story:** {movie_row['overview']}")

                        providers = tmdb_info["providers"]
                        if providers and any(providers.values()):
                            st.markdown("**Where to watch:**")
                            for category, label in [("flatrate", "Stream"), ("rent", "Rent"), ("buy", "Buy")]:
                                names = providers.get(category, [])
                                if names:
                                    st.markdown(f"- {label}: {', '.join(names)}")
                        else:
                            st.markdown("*Streaming availability not found for your region.*")

                    if st.button("Close details", key="close_details"):
                        st.session_state.expanded_movie = None
                        st.rerun()

# ════════════════════════════════════════════════════════════
# TAB 2 — SENTIMENT ANALYZER + REVIEW TALLY
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="metric-row">
        <div class="metric-box"><p class="metric-num">50K</p><p class="metric-lbl">Training reviews</p></div>
        <div class="metric-box"><p class="metric-num">~90%</p><p class="metric-lbl">Test accuracy</p></div>
        <div class="metric-box"><p class="metric-num">TF-IDF</p><p class="metric-lbl">Feature method</p></div>
        <div class="metric-box"><p class="metric-num">LR</p><p class="metric-lbl">Algorithm</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-lbl">Which movie is your review for?</p>', unsafe_allow_html=True)
    movie_title_input = st.text_input(
        "Movie name",
        placeholder="e.g. Interstellar",
        label_visibility="collapsed",
        key="review_movie_title"
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-lbl">Your review</p>', unsafe_allow_html=True)

    user_input = st.text_area(
        "Your review",
        placeholder="e.g. The performances were outstanding and the cinematography was breathtaking...",
        height=150,
        label_visibility="collapsed",
        key="review_text"
    )

    col1, col2, col3 = st.columns([1.2, 1, 3])
    with col1:
        analyze_clicked = st.button("Analyze →", type="primary", use_container_width=True)
    with col2:
        clear_clicked = st.button("Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.pending_text = ""
        st.session_state.trigger_analysis = False
        st.rerun()

    analyze = analyze_clicked or st.session_state.trigger_analysis
    st.session_state.trigger_analysis = False

    if analyze:
        if not user_input.strip():
            st.warning("Please enter a review first.")
        elif not movie_title_input.strip():
            st.warning("Please tell us which movie this review is for.")
        else:
            with st.spinner("Analyzing..."):
                cleaned = clean_text(user_input)
                vec = tfidf.transform([cleaned])
                pred = model.predict(vec)[0]
                proba = model.predict_proba(vec)[0]
                sentiment_label = "positive" if pred == 1 else "negative"
                confidence = proba[1] if pred == 1 else proba[0]

            add_review(movie_title_input, user_input, sentiment_label, float(confidence))

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            if pred == 1:
                st.markdown(f"""
                <div class="result-card-pos">
                    <div class="result-emoji">😊</div>
                    <div>
                        <p class="result-title-pos">That was a positive review!</p>
                        <p class="result-conf">Confidence: {confidence:.1%}</p>
                        <div class="conf-track"><div class="conf-fill-pos" style="width:{confidence*100:.0f}%"></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-neg">
                    <div class="result-emoji">😞</div>
                    <div>
                        <p class="result-title-neg">That was a negative review.</p>
                        <p class="result-conf">Confidence: {confidence:.1%}</p>
                        <div class="conf-track"><div class="conf-fill-neg" style="width:{confidence*100:.0f}%"></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("See cleaned text"):
                st.code(cleaned)

    # ── Live tally for the entered movie ──
    if movie_title_input.strip():
        counts = get_sentiment_counts(movie_title_input)
        total = counts["positive"] + counts["negative"]
        if total > 0:
            st.markdown(f"""
            <div class="tally-box">
                <p class="section-lbl" style="margin-bottom:10px;">Reviews for "{movie_title_input}"</p>
                <div class="tally-row"><span>👍 Positive</span><span class="tally-pos">{counts['positive']}</span></div>
                <div class="tally-row"><span>👎 Negative</span><span class="tally-neg">{counts['negative']}</span></div>
                <div class="tally-row" style="color:#8b81a8;"><span>Total reviews</span><span>{total}</span></div>
            </div>
            """, unsafe_allow_html=True)

    # ── Examples ──
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-lbl">Try an example</p>', unsafe_allow_html=True)

    examples = [
        ("😊", "Absolutely brilliant! One of the best films I've ever seen."),
        ("😞", "Terrible script, wooden acting. Complete waste of time."),
        ("😊", "A masterpiece of storytelling. Deeply moving and visually stunning."),
        ("😞", "Boring, predictable, and painfully long. I do not recommend it."),
        ("😊", "Funny, heartfelt, and beautifully directed. Loved every minute."),
    ]

    for i, (emoji, text) in enumerate(examples):
        if st.button(f"{emoji}  {text}", use_container_width=True, key=f"example_{i}"):
            st.session_state.pending_text = text
            st.session_state.trigger_analysis = True
            st.rerun()

# ── Footer ────────────────────────────────────────────────
st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#6b6285;font-size:12px;'>"
    "Built with scikit-learn + Streamlit + TMDB API · CineSense</p>",
    unsafe_allow_html=True
)