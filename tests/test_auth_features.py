import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from api import app
from recommender import recommender_engine
import auth_service

@pytest.fixture(scope="session", autouse=True)
def load_model():
    recommender_engine.load_artifacts()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_auth_config_endpoint(client):
    res = client.get("/auth/config")
    assert res.status_code == 200
    data = res.json()
    assert "google_client_id" in data


def test_dev_login_and_me(client):
    # 1. Dev login
    login_res = client.post("/auth/dev-login", json={"email": "tester@cineml.test", "name": "Test User"})
    assert login_res.status_code == 200
    data = login_res.json()
    assert "token" in data
    assert data["user"]["email"] == "tester@cineml.test"
    token = data["token"]

    # 2. Check /auth/me with Bearer token
    me_res = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["user"]["email"] == "tester@cineml.test"


def test_watchlist_toggle_and_retrieval(client):
    # Login
    login_res = client.post("/auth/dev-login", json={"email": "watchlist.user@cineml.test"})
    token = login_res.json()["token"]

    movie_id = 27205  # Inception

    # 1. Add to watchlist
    toggle_res = client.post(
        "/user/watchlist/toggle",
        json={"movie_id": movie_id},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert toggle_res.status_code == 200
    assert toggle_res.json()["in_watchlist"] is True

    # 2. Get watchlist
    get_res = client.get("/user/watchlist", headers={"Authorization": f"Bearer {token}"})
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["count"] >= 1
    assert any(m["movie_id"] == movie_id for m in data["movies"])

    # 3. Toggle off
    toggle_off_res = client.post(
        "/user/watchlist/toggle",
        json={"movie_id": movie_id},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert toggle_off_res.status_code == 200
    assert toggle_off_res.json()["in_watchlist"] is False


def test_movie_like_toggle(client):
    # Login
    login_res = client.post("/auth/dev-login", json={"email": "like.user@cineml.test"})
    token = login_res.json()["token"]

    movie_id = 19995  # Avatar

    # 1. Like movie
    like_res = client.post(
        f"/movie/{movie_id}/like",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert like_res.status_code == 200
    data = like_res.json()
    assert data["liked"] is True

    # 2. Check likes info
    info_res = client.get(
        f"/movie/{movie_id}/likes",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert info_res.status_code == 200
    assert info_res.json()["user_liked"] is True

    # 3. Unlike
    unlike_res = client.post(
        f"/movie/{movie_id}/like",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert unlike_res.status_code == 200
    assert unlike_res.json()["liked"] is False


def test_movie_feedback(client):
    # Login
    login_res = client.post("/auth/dev-login", json={"email": "critic@cineml.test", "name": "Film Critic"})
    token = login_res.json()["token"]

    movie_id = 27205  # Inception

    # 1. Post review
    post_res = client.post(
        f"/movie/{movie_id}/feedback",
        json={"rating": 9.5, "comment": "Mind-bending masterpiece by Christopher Nolan!"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert post_res.status_code == 200
    assert post_res.json()["status"] == "ok"
    assert post_res.json()["feedback"]["rating"] == 9.5

    # 2. Read reviews
    get_res = client.get(f"/movie/{movie_id}/feedback")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["count"] >= 1
    assert any("Nolan" in f["comment"] for f in data["feedbacks"])
