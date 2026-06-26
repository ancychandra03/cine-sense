"""
database.py
Handles all SQLite storage for user reviews and sentiment tallies.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "reviews.db"


def get_connection():
    """Create (if needed) and return a SQLite connection."""
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db():
    """Create the reviews table if it doesn't already exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_title TEXT NOT NULL,
            review_text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def add_review(movie_title: str, review_text: str, sentiment: str, confidence: float):
    """Insert a new review record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO reviews (movie_title, review_text, sentiment, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (movie_title.strip(), review_text.strip(), sentiment, confidence, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def get_sentiment_counts(movie_title: str) -> dict:
    """Return {'positive': n, 'negative': n} for a given movie title (case-insensitive)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT sentiment, COUNT(*) FROM reviews
        WHERE LOWER(movie_title) = LOWER(?)
        GROUP BY sentiment
        """,
        (movie_title.strip(),)
    )
    rows = cursor.fetchall()
    conn.close()

    counts = {"positive": 0, "negative": 0}
    for sentiment, count in rows:
        counts[sentiment] = count
    return counts


def get_recent_reviews(movie_title: str, limit: int = 5) -> list:
    """Return the most recent reviews for a movie."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT review_text, sentiment, confidence, created_at FROM reviews
        WHERE LOWER(movie_title) = LOWER(?)
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (movie_title.strip(), limit)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_all_reviewed_movies() -> list:
    """Return a list of distinct movie titles that have at least one review."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT movie_title FROM reviews ORDER BY movie_title")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]
