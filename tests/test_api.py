import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api import app
from recommender import recommender_engine
from tmdb_service import tmdb_service, create_svg_placeholder


@pytest.fixture(scope="session", autouse=True)
def load_model():
    """Ensure recommender engine artifacts are loaded once for the test session."""
    recommender_engine.load_artifacts()


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


# 1. Health check test
def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "CineML" in data["service"]
    assert data["model_loaded"] is True
    assert data["movies_count"] > 100000


# 2. Movie search with matching titles
def test_search_movies_matching(client):
    response = client.get("/movies?q=inception&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "inception"
    assert len(data["results"]) >= 1
    # Check Inception exists in results
    titles = [m["title"].lower() for m in data["results"]]
    assert any("inception" in t for t in titles)
    first_res = data["results"][0]
    assert "movie_id" in first_res
    assert "tmdb_id" in first_res
    assert "title" in first_res
    assert "poster_url" in first_res


# 3. Movie search with no matches
def test_search_movies_no_match(client):
    response = client.get("/movies?q=zzzzxxyyzzqwertynonexistent12345")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["results"] == []


# 4. Recommendation for a valid movie ID (Inception = 27205)
def test_recommendation_valid_id(client):
    response = client.get("/recommend?movie_id=27205&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["selected_movie"]["movie_id"] == 27205
    assert data["selected_movie"]["title"] == "Inception"
    assert data["count"] == 5
    assert len(data["recommendations"]) == 5
    # Ensure selected movie is excluded from recommendations
    rec_ids = [m["movie_id"] for m in data["recommendations"]]
    assert 27205 not in rec_ids


# 5. Recommendation for an invalid movie ID
def test_recommendation_invalid_id(client):
    response = client.get("/recommend?movie_id=999999999999&limit=5")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


# 6. Invalid recommendation limit validation
def test_recommendation_invalid_limit(client):
    # limit must be <= 50 and >= 1
    response = client.get("/recommend?movie_id=27205&limit=0")
    assert response.status_code == 422  # FastAPI validation error

    response_high = client.get("/recommend?movie_id=27205&limit=100")
    assert response_high.status_code == 422


# 7. Single movie metadata endpoint
def test_single_movie_endpoint(client):
    response = client.get("/movie/27205")
    assert response.status_code == 200
    data = response.json()
    assert data["movie_id"] == 27205
    assert data["title"] == "Inception"
    assert isinstance(data["overview"], str)


def test_single_movie_not_found(client):
    response = client.get("/movie/-1")
    assert response.status_code == 404


# 8. Handling duplicate movie titles via stable IDs
def test_duplicate_titles_distinct_ids(client):
    # Search for common title that may have multiple versions
    response = client.get("/movies?q=Alice in Wonderland&limit=10")
    assert response.status_code == 200
    data = response.json()
    ids = [m["movie_id"] for m in data["results"]]
    # Every movie returned must have a unique stable movie_id
    assert len(ids) == len(set(ids))


# 9. Consistency between selected movie ID, dataframe row, and neighbor indices
def test_indexing_consistency():
    # Inception is row 0
    row_idx = recommender_engine.id_to_row[27205]
    assert row_idx == 0
    row = recommender_engine.movies.iloc[row_idx]
    assert row["id"] == 27205
    assert row["title"] == "Inception"

    # Row 0 neighbor vector in all_neighbors: index 0 is 0
    assert recommender_engine.all_neighbors[0][0] == 0

    # First neighbor should be index 129882
    first_neighbor_idx = recommender_engine.all_neighbors[0][1]
    neighbor_row = recommender_engine.movies.iloc[first_neighbor_idx]
    assert neighbor_row["id"] == 1293777
    assert neighbor_row["title"] == "Past Never Dies"


# 10. TMDB unavailable or timeout mock test
def test_tmdb_mock_timeout(client):
    with patch.object(tmdb_service.session, "get", side_effect=Exception("Connection timed out")):
        # Clear cache for 27205 to force request attempt
        tmdb_service._cache.pop(27205, None)
        # Force is_configured to True so it attempts the request
        with patch.object(tmdb_service, "is_configured", return_value=True):
            response = client.get("/recommend?movie_id=27205&limit=3")
            assert response.status_code == 200
            data = response.json()
            # Must still succeed gracefully with local data and fallback posters
            assert data["selected_movie"]["title"] == "Inception"
            assert data["selected_movie"]["poster_url"] is not None
            assert len(data["recommendations"]) == 3


# 11. Missing TMDB credentials test
def test_missing_tmdb_credentials(client):
    with patch.dict("os.environ", {"TMDB_API_KEY": "", "TMDB_ACCESS_TOKEN": ""}):
        with patch.object(tmdb_service, "api_key", ""), patch.object(tmdb_service, "access_token", ""):
            assert tmdb_service.is_configured() is False
            details = tmdb_service.get_movie_details(27205, fallback_title="Inception")
            assert details["poster_url"].startswith("data:image/svg+xml")


# 12. TMDB successful response mock test
def test_tmdb_successful_mock(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "poster_path": "/edv5CZvWj09upOsy2Y6IwDhK8bt.jpg",
        "release_date": "2010-07-15",
        "genres": [{"id": 28, "name": "Action"}, {"id": 878, "name": "Science Fiction"}],
        "vote_average": 8.4
    }

    # Clear cache for 157336 (Interstellar)
    tmdb_service._cache.pop(157336, None)
    with patch.object(tmdb_service, "is_configured", return_value=True):
        with patch.object(tmdb_service.session, "get", return_value=mock_response):
            movie = tmdb_service.get_movie_details(157336, fallback_title="Interstellar")
            assert "https://image.tmdb.org/t/p/w500/edv5CZvWj09upOsy2Y6IwDhK8bt.jpg" == movie["poster_url"]
            assert movie["release_date"] == "2010-07-15"
            assert "Science Fiction" in movie["genres"]
            assert movie["vote_average"] == 8.4


# 13. Genres endpoint test
def test_genres_endpoint(client):
    response = client.get("/genres")
    assert response.status_code == 200
    data = response.json()
    assert "genres" in data
    assert len(data["genres"]) > 0
    assert "Action" in data["genres"]


# 14. Preferences recommendations endpoint test
def test_preferences_endpoint(client):
    payload = {
        "genres": ["Action", "Science Fiction"],
        "min_rating": 6.0,
        "limit": 5
    }
    response = client.post("/recommend/preferences", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] <= 5
    assert len(data["movies"]) <= 5
    assert "genres_applied" in data
    assert "Action" in data["genres_applied"]
    if data["movies"]:
        first = data["movies"][0]
        assert "movie_id" in first
        assert "title" in first

