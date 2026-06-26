"""
recommender.py
A content-based movie recommender.

Honest approach (documented for the README too):
- This is NOT a neural/AI recommender. It's a classic content-based
  filtering system: each mood maps to a set of relevant genres, and
  within that genre pool, movies are ranked using TF-IDF similarity
  between the user's mood description and each movie's overview text.
- This reuses the same TF-IDF technique from the sentiment analyzer,
  applied to a different problem (text similarity vs text classification).
"""

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "data" / "movies_clean.csv"

# Mood -> relevant genre pool. This mapping is a manual, explainable
# editorial choice (not learned), which keeps the system transparent.
MOOD_GENRE_MAP = {
    "Happy":      ["Comedy", "Family", "Animation", "Music"],
    "Sad":        ["Drama", "Romance"],
    "Thrilled":   ["Action", "Thriller", "Adventure"],
    "Scared":     ["Horror", "Mystery", "Thriller"],
    "Romantic":   ["Romance", "Drama"],
    "Curious":    ["Mystery", "Science Fiction", "Documentary"],
    "Relaxed":    ["Family", "Animation", "Comedy"],
    "Adventurous": ["Adventure", "Action", "Fantasy", "Science Fiction"],
}


class MovieRecommender:
    def __init__(self):
        self.df = pd.read_csv(DATA_PATH)
        self.df["combined_text"] = (
            self.df["genres"].fillna("") + " " + self.df["overview"].fillna("")
        )
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_text"])

    def available_moods(self):
        return list(MOOD_GENRE_MAP.keys())

    def recommend_by_mood(self, mood: str, free_text: str = "", top_n: int = 8) -> pd.DataFrame:
        """
        Recommend movies for a given mood, optionally refined by free-text
        description (e.g. "something like a heist movie but funny").
        """
        genres = MOOD_GENRE_MAP.get(mood, [])
        pool = self.df[self.df["genres"].apply(
            lambda g: any(genre in str(g) for genre in genres)
        )].copy()

        if pool.empty:
            pool = self.df.copy()

        query_text = mood + " " + " ".join(genres) + " " + free_text

        pool_indices = pool.index
        pool_matrix = self.tfidf_matrix[pool_indices]

        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, pool_matrix).flatten()

        pool = pool.copy()
        pool["similarity"] = sims

        # Blend similarity with popularity/rating so obscure low-quality
        # matches don't dominate purely on text overlap.
        pool["score"] = (
            pool["similarity"] * 0.7
            + (pool["rating"].fillna(0) / 10) * 0.2
            + (pool["popularity"].fillna(0) / pool["popularity"].max()) * 0.1
        )

        result = pool.sort_values("score", ascending=False).head(top_n)
        return result[["title", "genres", "overview", "rating", "release_date"]].reset_index(drop=True)

    def search_title(self, query: str) -> pd.DataFrame:
        """Simple substring search for the review section's movie picker."""
        matches = self.df[self.df["title"].str.contains(query, case=False, na=False)]
        return matches[["title", "genres", "release_date"]].head(10)
