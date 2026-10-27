import os
import sys
import pickle
import logging
from typing import List, Dict, Any, Optional

# Setup logger
logger = logging.getLogger("cineml.recommender")

# Compatibility shim for NumPy 2.x pickled artifacts loaded on NumPy 1.x
try:
    import numpy._core
except ImportError:
    try:
        import numpy.core
        sys.modules['numpy._core'] = numpy.core
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
        if hasattr(numpy.core, '_multiarray_umath'):
            sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath
    except Exception as e:
        logger.warning(f"Failed to apply NumPy compatibility shim: {e}")

import numpy as np
import pandas as pd


class MovieRecommender:
    """
    Core Recommendation Engine.
    Loads and reuses the precomputed Content-Based Filtering artifacts:
      - artifacts/movies.pkl: Pandas DataFrame with 173,268 movies
      - artifacts/all_neighbors.pkl: Precomputed Nearest Neighbors cosine similarity matrix
    """

    def __init__(self, movies_path: str = "artifacts/movies.pkl", neighbors_path: str = "artifacts/all_neighbors.pkl"):
        self.movies_path = movies_path
        self.neighbors_path = neighbors_path
        self.movies: Optional[pd.DataFrame] = None
        self.all_neighbors: Optional[np.ndarray] = None
        self.id_to_row: Dict[int, int] = {}
        self.titles_lower: List[str] = []
        self.is_loaded: bool = False

    def load_artifacts(self) -> None:
        """Load movies and neighbor matrices into memory once."""
        if self.is_loaded:
            return

        if not os.path.exists(self.movies_path):
            raise FileNotFoundError(f"Movies artifact not found at {self.movies_path}")
        if not os.path.exists(self.neighbors_path):
            raise FileNotFoundError(f"Neighbors artifact not found at {self.neighbors_path}")

        logger.info(f"Loading movie dataset from {self.movies_path}...")
        with open(self.movies_path, "rb") as f:
            raw_movies = pickle.load(f)

        # Reset index to guarantee exact 0-indexed integer rows matching all_neighbors
        self.movies = raw_movies.reset_index(drop=True)

        logger.info(f"Loading neighbor similarity matrix from {self.neighbors_path}...")
        with open(self.neighbors_path, "rb") as f:
            self.all_neighbors = pickle.load(f)

        # Build fast ID -> Row index lookup map
        # Note: In case of rare duplicates in ID, first occurrence takes precedence
        self.id_to_row = {}
        for row_idx, movie_id in enumerate(self.movies["id"]):
            movie_id_int = int(movie_id)
            if movie_id_int not in self.id_to_row:
                self.id_to_row[movie_id_int] = row_idx

        # Precompute lowercase titles for rapid case-insensitive searching
        self.titles_lower = [str(t).lower() for t in self.movies["title"]]

        self.is_loaded = True
        logger.info(f"Successfully loaded {len(self.movies)} movies and {len(self.all_neighbors)} neighbor vectors.")

    def _format_movie_metadata(self, row_idx: int) -> Dict[str, Any]:
        """Convert a movie row to a clean serializable dictionary."""
        row = self.movies.iloc[row_idx]
        movie_id = int(row["id"])
        title = str(row["title"])

        # Format overview safely
        overview_raw = row["overview"] if "overview" in row else ""
        if isinstance(overview_raw, list):
            overview = " ".join(str(w) for w in overview_raw)
        elif pd.isna(overview_raw):
            overview = ""
        else:
            overview = str(overview_raw)

        # Format genres
        genres_raw = row["genres"] if "genres" in row else []
        genres: List[str] = []
        if isinstance(genres_raw, list):
            genres = [str(g) for g in genres_raw]
        elif isinstance(genres_raw, str) and genres_raw.strip():
            genres = [genres_raw.strip()]

        return {
            "movie_id": movie_id,
            "tmdb_id": movie_id,
            "row_index": row_idx,
            "title": title,
            "overview": overview,
            "genres": genres,
            "release_date": None,  # Can be enriched via TMDB API
            "poster_url": None,     # Enriched via TMDB API
        }

    def get_movie_by_id(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve local metadata for a movie by its TMDB movie_id."""
        if not self.is_loaded:
            self.load_artifacts()

        row_idx = self.id_to_row.get(int(movie_id))
        if row_idx is None:
            return None
        return self._format_movie_metadata(row_idx)

    def search_movies(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search movies in local dataset by partial case-insensitive title match.
        Prioritizes:
          1. Exact matches
          2. Prefix (starts-with) matches
          3. Substring contains matches
        """
        if not self.is_loaded:
            self.load_artifacts()

        cleaned_query = (query or "").strip().lower()
        if not cleaned_query:
            # Return top popular/first few movies as sensible defaults
            bounded_limit = min(max(1, limit), 50)
            return [self._format_movie_metadata(i) for i in range(min(bounded_limit, len(self.movies)))]

        exact_matches = []
        prefix_matches = []
        contains_matches = []

        bounded_limit = min(max(1, limit), 50)

        for row_idx, title in enumerate(self.titles_lower):
            if title == cleaned_query:
                exact_matches.append(row_idx)
            elif title.startswith(cleaned_query):
                prefix_matches.append(row_idx)
            elif cleaned_query in title:
                contains_matches.append(row_idx)

            # Optimization: Stop once we have gathered sufficient candidates
            if len(exact_matches) + len(prefix_matches) + len(contains_matches) >= bounded_limit * 3:
                break

        ordered_indices = (exact_matches + prefix_matches + contains_matches)[:bounded_limit]
        return [self._format_movie_metadata(idx) for idx in ordered_indices]

    def get_recommendations(self, movie_id: int, limit: int = 10) -> Optional[Dict[str, Any]]:
        """
        Return the top-K recommendations for a given movie using precomputed neighbor matrix.
        Preserves the exact content-based similarity ranking from the ML model.
        """
        if not self.is_loaded:
            self.load_artifacts()

        row_idx = self.id_to_row.get(int(movie_id))
        if row_idx is None:
            return None

        # Clamp limit to range [1, 50]
        clamped_limit = min(max(1, limit), 50)

        # Retrieve neighbor indices from matrix. Row index 0 is self, 1.. are neighbors
        available_neighbors = len(self.all_neighbors[row_idx]) - 1
        actual_limit = min(clamped_limit, available_neighbors)
        neighbor_indices = self.all_neighbors[row_idx][1 : actual_limit + 1]

        selected_movie = self._format_movie_metadata(row_idx)
        recommendations = [self._format_movie_metadata(int(idx)) for idx in neighbor_indices]

        return {
            "selected_movie": selected_movie,
            "count": len(recommendations),
            "recommendations": recommendations,
        }


# Global engine singleton
recommender_engine = MovieRecommender()
