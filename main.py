"""
main.py
FastAPI backend for MoviSpect.

Wraps the existing modules (preprocess.py, recommender.py, database.py,
tmdb_client.py) and the trained sentiment model into HTTP endpoints the
frontend (moviespect_frontend.html) calls directly.

Run locally with:
    uvicorn main:app --reload --port 8000

The frontend's fetch() calls expect this server running at
http://localhost:8000 — see the CORS settings below if you serve the
HTML from a different origin/port.
"""

import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocess import clean_text
from recommender import MovieRecommender
from tmdb_client import get_movie_details
from database import init_db, add_review, get_sentiment_counts, get_recent_reviews

# ── App setup ──────────────────────────────────────────────
app = FastAPI(title="MoviSpect API")

# Allow the frontend (opened as a local file or served from any port)
# to call this API. Tighten this to specific origins before deploying
# publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path(__file__).parent / "models"

# ── Load models once at startup ───────────────────────────
with open(MODELS_DIR / "model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)
with open(MODELS_DIR / "tfidf.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

recommender = MovieRecommender()
init_db()


# ── Request/response schemas ──────────────────────────────
class AnalyzeRequest(BaseModel):
    movie_title: str
    review_text: str


class AnalyzeResponse(BaseModel):
    sentiment: str
    confidence: float
    tally: dict


# ── Routes ─────────────────────────────────────────────────

@app.get("/api/health")
def health():
    """Quick check that the server and models loaded correctly."""
    return {"status": "ok", "movies_loaded": len(recommender.df)}


@app.get("/api/moods")
def get_moods():
    """Return the list of supported moods for the chip grid."""
    return {"moods": recommender.available_moods()}


@app.get("/api/recommend")
def recommend(mood: Optional[str] = None, free_text: Optional[str] = "", top_n: int = 8):
    """
    Mood Reel's main endpoint. Provide `mood` (one of /api/moods),
    and/or `free_text` for refinement. At least one should be present.
    """
    if not mood and not free_text:
        raise HTTPException(400, "Provide a mood and/or free_text query.")

    # If only free_text is given with no mood, fall back to a neutral
    # mood bucket so recommend_by_mood still has a genre pool to search.
    effective_mood = mood or "Curious"

    results = recommender.recommend_by_mood(effective_mood, free_text=free_text or "", top_n=top_n)
    movies = results.to_dict(orient="records")
    return {"movies": movies}


@app.get("/api/search-movies")
def search_movies(q: str):
    """Autocomplete endpoint for The Verdict's movie picker."""
    if not q or len(q.strip()) < 1:
        return {"matches": []}
    matches = recommender.search_title(q)
    return {"matches": matches.to_dict(orient="records")}


@app.get("/api/movie-details")
def movie_details(title: str, year: Optional[str] = None):
    """
    Poster + where-to-watch info for the detail overlay. Falls back to
    empty/null fields if TMDB has no match or the API key isn't set —
    the frontend already handles that gracefully.
    """
    return get_movie_details(title, year)


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    """
    The Verdict's main endpoint: runs the real trained sentiment model
    on the review text, stores it, and returns the updated tally for
    that movie.
    """
    movie_title = payload.movie_title.strip()
    review_text = payload.review_text.strip()

    if not movie_title or not review_text:
        raise HTTPException(400, "movie_title and review_text are both required.")

    cleaned = clean_text(review_text)
    vec = tfidf_vectorizer.transform([cleaned])
    pred = sentiment_model.predict(vec)[0]
    proba = sentiment_model.predict_proba(vec)[0]

    sentiment = "positive" if pred == 1 else "negative"
    confidence = float(proba[1] if pred == 1 else proba[0])

    add_review(movie_title, review_text, sentiment, confidence)
    tally = get_sentiment_counts(movie_title)

    return AnalyzeResponse(sentiment=sentiment, confidence=confidence, tally=tally)


@app.get("/api/tally")
def tally(movie_title: str):
    """Standalone tally lookup (e.g. to show counts without submitting)."""
    counts = get_sentiment_counts(movie_title)
    return {"movie_title": movie_title, **counts, "total": counts["positive"] + counts["negative"]}


@app.get("/api/recent-reviews")
def recent_reviews(movie_title: str, limit: int = 5):
    """Most recent reviews for a movie, useful for a future 'reviews feed'."""
    rows = get_recent_reviews(movie_title, limit)
    return {
        "reviews": [
            {"review_text": r[0], "sentiment": r[1], "confidence": r[2], "created_at": r[3]}
            for r in rows
        ]
    }
