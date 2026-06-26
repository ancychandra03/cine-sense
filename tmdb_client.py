"""
tmdb_client.py
Thin wrapper around the TMDB API for:
  - searching a movie by title to get its TMDB id + poster
  - fetching watch providers (where to stream) for that movie

Requires TMDB_API_KEY in a .env file (see .env.example).
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Default region for watch-provider lookups. Change to match your audience.
DEFAULT_REGION = "IN"


def _get(endpoint: str, params: dict = None):
    if not API_KEY:
        return None
    params = params or {}
    params["api_key"] = API_KEY
    try:
        resp = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=6)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def search_movie(title: str, year: str = None):
    """Search TMDB for a movie title, return the best match (or None)."""
    params = {"query": title}
    if year:
        params["year"] = year
    data = _get("/search/movie", params)
    if not data or not data.get("results"):
        return None
    return data["results"][0]


def get_poster_url(movie: dict):
    """Build a full poster image URL from a TMDB movie result."""
    if not movie or not movie.get("poster_path"):
        return None
    return f"{IMAGE_BASE}{movie['poster_path']}"


def get_watch_providers(tmdb_movie_id: int, region: str = DEFAULT_REGION):
    """
    Return a dict like {'flatrate': [...names...], 'rent': [...], 'buy': [...]}
    for the given region, or None if unavailable.
    """
    data = _get(f"/movie/{tmdb_movie_id}/watch/providers")
    if not data:
        return None
    results = data.get("results", {})
    region_data = results.get(region)
    if not region_data:
        return None

    providers = {}
    for category in ["flatrate", "rent", "buy"]:
        entries = region_data.get(category, [])
        providers[category] = [p["provider_name"] for p in entries]
    return providers


def get_movie_details(title: str, year: str = None):
    """
    Convenience function: search + fetch poster + watch providers in one call.
    Returns a dict with poster_url and providers, or sensible empty defaults
    if TMDB has no match or the API key isn't set.
    """
    movie = search_movie(title, year)
    if not movie:
        return {"poster_url": None, "providers": None, "tmdb_id": None}

    poster_url = get_poster_url(movie)
    providers = get_watch_providers(movie["id"])

    return {
        "poster_url": poster_url,
        "providers": providers,
        "tmdb_id": movie["id"],
    }
