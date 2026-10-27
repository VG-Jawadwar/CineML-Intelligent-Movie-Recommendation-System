import os
import logging
from typing import Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Load .env variables (including Render secret files if present)
if os.path.exists("/etc/secrets/.env"):
    load_dotenv("/etc/secrets/.env")
load_dotenv()

logger = logging.getLogger("cineml.tmdb")

# Default TMDB credentials for CineML project
DEFAULT_TMDB_API_KEY = "TMDB_API_KEY_VALUE"
DEFAULT_TMDB_ACCESS_TOKEN = (
    "TMDB_ACCESS_TOKEN_VALUE"
)

# Known placeholder values that should not trigger network calls
PLACEHOLDER_KEYS = {
    "",
    "your_tmdb_api_key_here",
    "your_api_key",
    "your_api_read_access_token_if_using_bearer_auth",
    "your_tmdb_access_token_here",
    "none",
    "null"
}


def create_svg_placeholder(title: str = "Movie") -> str:
    """Generate a clean cinematic SVG data-URI poster placeholder."""
    safe_title = (
        title.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
    if len(safe_title) > 30:
        safe_title = safe_title[:27] + "..."

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 750" width="100%" height="100%">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#141923" />
      <stop offset="50%" stop-color="#1b2234" />
      <stop offset="100%" stop-color="#0d1117" />
    </linearGradient>
    <radialGradient id="glow" cx="50%" cy="40%" r="60%">
      <stop offset="0%" stop-color="#e50914" stop-opacity="0.25" />
      <stop offset="100%" stop-color="#e50914" stop-opacity="0" />
    </radialGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#bg)" />
  <rect width="100%" height="100%" fill="url(#glow)" />
  <rect x="20" y="20" width="460" height="710" rx="16" fill="none" stroke="#e50914" stroke-opacity="0.3" stroke-width="2" stroke-dasharray="8 6" />
  
  <!-- Film Slate Icon -->
  <g transform="translate(200, 240)">
    <path d="M0 10 C0 4.477 4.477 0 10 0 L90 0 C95.523 0 100 4.477 100 10 L100 75 C100 80.523 95.523 85 90 85 L10 85 C4.477 85 0 80.523 0 75 Z" fill="#21262d" stroke="#30363d" stroke-width="2"/>
    <polygon points="40,25 70,42.5 40,60" fill="#e50914" />
  </g>

  <text x="250" y="400" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-size="24" font-weight="700" fill="#f0f6fc" text-anchor="middle">
    {safe_title}
  </text>
  <text x="250" y="440" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-size="14" font-weight="500" fill="#8b949e" text-anchor="middle">
    CineML Poster Fallback
  </text>
</svg>"""
    import urllib.parse
    return f"data:image/svg+xml;utf8,{urllib.parse.quote(svg)}"


class TMDBService:
    """
    Robust TMDB Client for poster fetching and metadata enrichment.
    Handles caching, timeouts, retries, and offline / placeholder fallbacks.
    """

    TMDB_BASE_URL = "https://api.themoviedb.org/3"
    IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

    def __init__(self, api_key: Optional[str] = None, access_token: Optional[str] = None):
        self.api_key = api_key or os.getenv("TMDB_API_KEY", "")
        self.access_token = access_token or os.getenv("TMDB_ACCESS_TOKEN", "")

        # Session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # In-memory cache: movie_id -> metadata dictionary
        self._cache: Dict[int, Optional[Dict[str, Any]]] = {}

    def get_api_key(self) -> str:
        """Resolve API key from environment, instance attribute, or default."""
        env_key = os.getenv("TMDB_API_KEY")
        if env_key is not None:
            return env_key.strip().strip('"').strip("'")
        if self.api_key:
            return str(self.api_key).strip().strip('"').strip("'")
        return DEFAULT_TMDB_API_KEY

    def get_access_token(self) -> str:
        """Resolve Bearer access token from environment, instance attribute, or default."""
        env_token = os.getenv("TMDB_ACCESS_TOKEN")
        if env_token is not None:
            return env_token.strip().strip('"').strip("'")
        if self.access_token:
            return str(self.access_token).strip().strip('"').strip("'")
        return DEFAULT_TMDB_ACCESS_TOKEN

    def is_configured(self) -> bool:
        """Check if a real non-placeholder TMDB credential is provided."""
        current_key = self.get_api_key()
        current_token = self.get_access_token()

        has_valid_key = current_key and current_key.lower() not in PLACEHOLDER_KEYS
        has_valid_token = current_token and current_token.lower() not in PLACEHOLDER_KEYS

        return bool(has_valid_key or has_valid_token)

    def get_movie_details(self, tmdb_id: int, fallback_title: str = "Movie") -> Dict[str, Any]:
        """
        Fetch movie details and poster from TMDB API with in-memory caching.
        Returns a dictionary with poster_url, release_date, genres, etc.
        Gracefully returns fallback data on error or when credentials are missing.
        """
        tmdb_id = int(tmdb_id)

        # Check in-memory cache first
        if tmdb_id in self._cache:
            cached = self._cache[tmdb_id]
            if cached is not None:
                return cached
            # If cached as None (previous failure), return fallback
            return {
                "poster_url": create_svg_placeholder(fallback_title),
                "release_date": None,
                "genres": [],
                "vote_average": None,
                "is_cached": True,
            }

        if not self.is_configured():
            # Graceful placeholder without logging secrets or making futile network requests
            fallback_res = {
                "poster_url": create_svg_placeholder(fallback_title),
                "release_date": None,
                "genres": [],
                "vote_average": None,
                "is_cached": False,
            }
            # Cache the missing credential result so subsequent batch requests are instant
            self._cache[tmdb_id] = fallback_res
            return fallback_res

        # Prepare request
        current_key = self.get_api_key()
        current_token = self.get_access_token()

        url = f"{self.TMDB_BASE_URL}/movie/{tmdb_id}"
        headers = {"Accept": "application/json"}
        params = {"language": "en-US"}

        if current_token and current_token.lower() not in PLACEHOLDER_KEYS:
            headers["Authorization"] = f"Bearer {current_token}"
        elif current_key and current_key.lower() not in PLACEHOLDER_KEYS:
            params["api_key"] = current_key

        try:
            response = self.session.get(url, headers=headers, params=params, timeout=2.5)

            if response.status_code == 200:
                data = response.json()
                poster_path = data.get("poster_path")
                poster_url = (
                    f"{self.IMAGE_BASE_URL}{poster_path}"
                    if poster_path
                    else create_svg_placeholder(fallback_title)
                )

                genres = [g["name"] for g in data.get("genres", []) if "name" in g]

                result = {
                    "poster_url": poster_url,
                    "release_date": data.get("release_date"),
                    "genres": genres,
                    "vote_average": data.get("vote_average"),
                    "tagline": data.get("tagline"),
                    "runtime": data.get("runtime"),
                    "is_cached": False,
                }
                self._cache[tmdb_id] = result
                return result
            else:
                logger.debug(f"TMDB request for ID {tmdb_id} returned status code {response.status_code}")
                self._cache[tmdb_id] = None
        except Exception as exc:
            # Network error or timeout: fail gracefully without crashing
            logger.debug(f"TMDB request exception for ID {tmdb_id}: {exc}")
            self._cache[tmdb_id] = None

        return {
            "poster_url": create_svg_placeholder(fallback_title),
            "release_date": None,
            "genres": [],
            "vote_average": None,
            "is_cached": False,
        }

    def enrich_movie(self, movie: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a local movie dictionary with TMDB poster and metadata."""
        tmdb_id = movie.get("tmdb_id") or movie.get("movie_id")
        title = movie.get("title", "Movie")
        details = self.get_movie_details(tmdb_id, fallback_title=title)

        enriched = dict(movie)
        if details.get("poster_url"):
            # Prefer TMDB high-res poster; if TMDB fell back to SVG, keep original poster if valid
            if "image.tmdb.org" in str(details["poster_url"]):
                enriched["poster_url"] = details["poster_url"]
            elif not enriched.get("poster_url"):
                enriched["poster_url"] = details["poster_url"]

        if details.get("release_date"):
            enriched["release_date"] = details["release_date"]
        if details.get("genres") and not enriched.get("genres"):
            enriched["genres"] = details["genres"]
        if details.get("vote_average") is not None:
            enriched["vote_average"] = details["vote_average"]

        return enriched


# Global TMDB Service singleton
tmdb_service = TMDBService()
