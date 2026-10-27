import os
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Body, Request, Depends, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from recommender import recommender_engine
from tmdb_service import tmdb_service
from database import (
    get_movies_collection,
    get_users_collection,
    is_mongodb_ready,
    is_mongodb_configured,
    get_last_db_error,
)
from auth_service import (
    get_google_client_id,
    create_access_token,
    verify_access_token,
    verify_google_credential,
    get_or_create_user,
    toggle_user_watchlist,
    get_user_watchlist_ids,
    toggle_movie_like,
    get_movie_likes_info,
    add_movie_feedback,
    get_movie_feedback,
)

# Load environment configuration
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("cineml.api")


# Lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing CineML recommendation engine...")
    try:
        recommender_engine.load_artifacts()
        logger.info("ML recommendation engine ready.")
    except Exception as e:
        logger.error(f"Failed to load recommendation artifacts: {e}", exc_info=True)

    if is_mongodb_ready():
        logger.info("Connected to MongoDB Atlas successfully.")
    else:
        logger.warning("MongoDB Atlas is not connected or MONGO_URI is missing. Will use in-memory dataset fallback.")

    yield
    logger.info("Shutting down CineML API.")


app = FastAPI(
    title="CineML Movie Recommendation API",
    description="Intelligent movie recommender with MongoDB Atlas storage & ML Nearest Neighbors for 'You Might Also Like'.",
    version="2.1.0",
    lifespan=lifespan,
)

# CORS Configuration - permit local dev, Render, Vercel and file:// access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Frontend Static Assets
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(frontend_dir):
    css_dir = os.path.join(frontend_dir, "css")
    js_dir = os.path.join(frontend_dir, "js")
    if os.path.isdir(css_dir):
        app.mount("/css", StaticFiles(directory=css_dir), name="css")
    if os.path.isdir(js_dir):
        app.mount("/js", StaticFiles(directory=js_dir), name="js")


# Pydantic schemas for structured JSON responses
class MovieItem(BaseModel):
    movie_id: int
    tmdb_id: int
    title: str
    overview: Optional[str] = ""
    genres: Optional[List[str]] = []
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    vote_average: Optional[float] = None


class MovieSearchResponse(BaseModel):
    query: str
    count: int
    results: List[MovieItem]


class RecommendationResponse(BaseModel):
    selected_movie: MovieItem
    count: int
    recommendations: List[MovieItem]


class MovieDetailResponse(MovieItem):
    you_might_also_like: List[MovieItem] = []
    total_likes: Optional[int] = 0
    user_liked: Optional[bool] = False
    feedbacks: Optional[List[Dict[str, Any]]] = []


class GoogleAuthRequest(BaseModel):
    credential: str = Field(..., description="Google OAuth ID Token JWT")


class DevLoginRequest(BaseModel):
    email: Optional[str] = Field("dev.tester@cineml.local", description="Test user email")
    name: Optional[str] = Field("Dev Tester", description="Test user display name")


class WatchlistToggleRequest(BaseModel):
    movie_id: int = Field(..., description="Movie ID to add or remove from watchlist")


class FeedbackCreateRequest(BaseModel):
    rating: float = Field(5.0, ge=1.0, le=10.0, description="Rating from 1 to 10")
    comment: str = Field(..., min_length=1, max_length=1000, description="Feedback or review text")


class PreferenceRequest(BaseModel):
    genres: List[str] = Field(default_factory=list, description="List of preferred genres")
    min_rating: Optional[float] = Field(0.0, description="Minimum rating filter")
    limit: int = Field(18, ge=1, le=50, description="Max movies to return")


class PreferenceResponse(BaseModel):
    genres_applied: List[str]
    count: int
    source: str
    movies: List[MovieItem]


DEFAULT_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]


def format_mongo_movie(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardize a movie document from MongoDB.
    Supports sample_mflix, custom collections, and seeded collections seamlessly.
    """
    # 1. Resolve ID (sample_mflix uses imdb.id, our seed uses movie_id/tmdb_id)
    movie_id = None
    if "movie_id" in doc and doc["movie_id"] is not None:
        try:
            movie_id = int(doc["movie_id"])
        except (ValueError, TypeError):
            pass
    elif "tmdb_id" in doc and doc["tmdb_id"] is not None:
        try:
            movie_id = int(doc["tmdb_id"])
        except (ValueError, TypeError):
            pass
    elif isinstance(doc.get("imdb"), dict) and doc["imdb"].get("id") is not None:
        try:
            movie_id = int(doc["imdb"]["id"])
        except (ValueError, TypeError):
            pass
    
    if movie_id is None:
        movie_id = abs(hash(str(doc.get("_id", "movie")))) % 100_000_000

    title = str(doc.get("title") or "Untitled")

    # 2. Resolve Plot / Overview
    overview = (
        doc.get("plot")
        or doc.get("fullplot")
        or doc.get("overview")
        or "No overview synopsis available."
    )
    if isinstance(overview, list):
        overview = " ".join(str(w) for w in overview)
    elif not isinstance(overview, str):
        overview = str(overview)

    # 3. Resolve Genres
    genres: List[str] = []
    raw_genres = doc.get("genres")
    if isinstance(raw_genres, list):
        genres = [str(g).strip() for g in raw_genres if str(g).strip()]
    elif isinstance(raw_genres, str) and raw_genres.strip():
        genres = [g.strip() for g in raw_genres.split(",") if g.strip()]

    # 4. Resolve Poster URL
    poster_url = doc.get("poster") or doc.get("poster_url")
    if poster_url and not str(poster_url).startswith("http"):
        poster_url = None

    # 5. Resolve Rating (sample_mflix has imdb.rating, seed has vote_average)
    vote_average = None
    if isinstance(doc.get("imdb"), dict):
        try:
            val = doc["imdb"].get("rating")
            if val is not None and val != "":
                vote_average = float(val)
        except (TypeError, ValueError):
            pass
    if vote_average is None and doc.get("vote_average") is not None:
        try:
            val = doc["vote_average"]
            if val is not None and val != "":
                vote_average = float(val)
        except (TypeError, ValueError):
            pass

    # 6. Resolve Year / Release Date
    release_date = None
    if doc.get("year"):
        release_date = str(doc["year"])
    elif doc.get("released"):
        release_date = str(doc["released"]).split()[0]
    elif doc.get("release_date"):
        release_date = str(doc["release_date"])

    formatted = {
        "movie_id": movie_id,
        "tmdb_id": movie_id,
        "title": title,
        "overview": overview,
        "genres": genres,
        "poster_url": poster_url,
        "vote_average": vote_average,
        "release_date": release_date,
    }
    return tmdb_service.enrich_movie(formatted)


# ---------------- API ENDPOINTS ----------------

@app.get("/health", summary="API Health Check")
def get_health_json():
    """Returns JSON diagnostics about the ML model and MongoDB Atlas connection."""
    return health_check()


@app.get("/app", summary="CineML Web Application")
def get_web_app():
    """Serves the CineML frontend application."""
    index_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend index.html not found"}


@app.get("/", summary="API Health & Status")
def root_endpoint(request: Request):
    """
    Root endpoint.
    If accessed via a browser expecting HTML, serves the CineML web UI.
    If accessed programmatically, returns JSON diagnostics.
    """
    accept = request.headers.get("accept", "")
    index_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if "text/html" in accept and not accept.startswith("*/*") and os.path.exists(index_path):
        return FileResponse(index_path)

    return health_check()


def health_check():
    """
    Health check payload.
    Reports status of ML engine, TMDB configuration, and MongoDB Atlas connection.
    """
    mongo_ready = is_mongodb_ready()
    mongo_count = 0
    if mongo_ready:
        col = get_movies_collection()
        if col is not None:
            try:
                mongo_count = col.estimated_document_count()
            except Exception:
                mongo_count = 0

    return {
        "status": "ok",
        "service": "CineML Movie Recommendation API",
        "model_loaded": recommender_engine.is_loaded,
        "movies_count": len(recommender_engine.movies) if recommender_engine.is_loaded else 0,
        "local_movies_count": len(recommender_engine.movies) if recommender_engine.is_loaded else 0,
        "mongodb_connected": mongo_ready,
        "mongodb_configured": is_mongodb_configured(),
        "mongodb_movies_count": mongo_count,
        "mongodb_error": None if mongo_ready else get_last_db_error(),
        "tmdb_configured": tmdb_service.is_configured(),
    }


@app.get("/genres", summary="Get Available Genres")
def get_genres():
    """
    Returns available movie genres for the user preference form.
    Loads from MongoDB Atlas if available; otherwise returns default standard genres.
    """
    col = get_movies_collection()
    if col is not None and is_mongodb_ready():
        try:
            distinct_genres = col.distinct("genres")
            ignore_set = {"Short", "Talk-Show", "News", "Film-Noir"}
            clean_genres = sorted([
                g for g in distinct_genres 
                if isinstance(g, str) and g.strip() and g not in ignore_set
            ])
            if clean_genres:
                return {"genres": clean_genres}
        except Exception as e:
            logger.warning(f"Failed to fetch genres from MongoDB Atlas: {e}")

    return {"genres": DEFAULT_GENRES}


@app.post("/recommend/preferences", response_model=PreferenceResponse, summary="Recommend Movies by User Preferences")
def recommend_by_preferences(payload: PreferenceRequest = Body(...)):
    """
    Recommend movies based on the user's preference form (genres and min rating).
    Fetches directly from MongoDB Atlas; falls back smoothly if DB is not ready.
    """
    if not recommender_engine.is_loaded:
        recommender_engine.load_artifacts()

    selected_genres = [g.strip() for g in payload.genres if g.strip()]
    col = get_movies_collection()
    movies_found: List[Dict[str, Any]] = []
    source = "mongodb_atlas"

    if col is not None and is_mongodb_ready():
        try:
            query: Dict[str, Any] = {}
            if selected_genres:
                expanded = list(selected_genres)
                if "Science Fiction" in selected_genres and "Sci-Fi" not in selected_genres:
                    expanded.append("Sci-Fi")
                elif "Sci-Fi" in selected_genres and "Science Fiction" not in selected_genres:
                    expanded.append("Science Fiction")
                query["genres"] = {"$in": expanded}

            if payload.min_rating and payload.min_rating > 0:
                query["$or"] = [
                    {"imdb.rating": {"$gte": float(payload.min_rating)}},
                    {"vote_average": {"$gte": float(payload.min_rating)}}
                ]

            cursor = col.find(query).limit(payload.limit)
            for doc in cursor:
                movies_found.append(format_mongo_movie(doc))
        except Exception as e:
            logger.error(f"Error querying MongoDB Atlas: {e}", exc_info=True)
            movies_found = []

    # Fast fallback if MongoDB returned empty
    if not movies_found:
        source = "local_dataset"
        bounded = min(payload.limit, 20)
        for idx in range(min(bounded * 2, len(recommender_engine.movies))):
            movie = recommender_engine._format_movie_metadata(idx)
            movies_found.append(tmdb_service.enrich_movie(movie))
            if len(movies_found) >= payload.limit:
                break

    return PreferenceResponse(
        genres_applied=selected_genres,
        count=len(movies_found),
        source=source,
        movies=movies_found
    )


@app.get("/recommend", response_model=RecommendationResponse, summary="Get Content-Based Recommendations")
def get_recommendations(
    movie_id: int = Query(..., description="Stable TMDB movie ID from dataset"),
    limit: int = Query(10, ge=1, le=50, description="Number of recommendations (1-50)")
):
    """
    Generate content-based movie recommendations for a selected movie ID.
    Uses the precomputed cosine-similarity Nearest Neighbors model.
    """
    if not recommender_engine.is_loaded:
        recommender_engine.load_artifacts()

    rec_data = recommender_engine.get_recommendations(movie_id=movie_id, limit=limit)
    if not rec_data:
        raise HTTPException(
            status_code=404,
            detail=f"Movie with ID '{movie_id}' was not found in the CineML database."
        )

    enriched_selected = tmdb_service.enrich_movie(rec_data["selected_movie"])
    enriched_recommendations = [
        tmdb_service.enrich_movie(movie) for movie in rec_data["recommendations"]
    ]

    return RecommendationResponse(
        selected_movie=enriched_selected,
        count=len(enriched_recommendations),
        recommendations=enriched_recommendations
    )


@app.get("/movie/{movie_id}", response_model=MovieDetailResponse, summary="Get Movie Details & 'You Might Also Like'")
def get_movie_details(
    movie_id: int,
    recs_limit: int = 8,
    authorization: Optional[str] = Header(None)
):
    """
    Fetches a clicked movie's detailed data from MongoDB Atlas.
    Computes and returns the e-commerce style 'You Might Also Like' similar movies
    using the ML Nearest Neighbors cosine similarity model, along with likes and reviews.
    """
    limit_val = recs_limit if isinstance(recs_limit, int) else 8
    if not recommender_engine.is_loaded:
        recommender_engine.load_artifacts()

    # 1. Fetch main movie from MongoDB Atlas (checking movie_id, imdb.id, tmdb_id)
    movie_data = None
    col = get_movies_collection()
    if col is not None and is_mongodb_ready():
        try:
            doc = col.find_one({
                "$or": [
                    {"movie_id": movie_id},
                    {"imdb.id": movie_id},
                    {"tmdb_id": movie_id}
                ]
            })
            if doc:
                movie_data = format_mongo_movie(doc)
        except Exception as e:
            logger.warning(f"Error loading movie {movie_id} from MongoDB: {e}")

    # Fallback to local recommender engine if not found in MongoDB
    if not movie_data:
        raw_local = recommender_engine.get_movie_by_id(movie_id)
        if raw_local:
            movie_data = tmdb_service.enrich_movie(raw_local)

    if not movie_data:
        raise HTTPException(
            status_code=404,
            detail=f"Movie with ID '{movie_id}' not found."
        )

    # 2. Get ML Recommendations ("You Might Also Like")
    similar_movies: List[Dict[str, Any]] = []

    # First attempt: Try direct neighbor lookup with movie_id in ML engine
    rec_result = recommender_engine.get_recommendations(movie_id=movie_id, limit=limit_val)

    # Second attempt: If movie_id not in ML engine, lookup movie by title in ML engine!
    if not rec_result and movie_data.get("title"):
        title_matches = recommender_engine.search_movies(movie_data["title"], limit=1)
        if title_matches:
            ml_id = title_matches[0]["movie_id"]
            rec_result = recommender_engine.get_recommendations(movie_id=ml_id, limit=limit_val)

    # Process ML recommendations
    if rec_result and "recommendations" in rec_result:
        for rec in rec_result["recommendations"]:
            rec_id = rec["movie_id"]
            db_rec = None
            if col is not None and is_mongodb_ready():
                try:
                    db_doc = col.find_one({"$or": [{"movie_id": rec_id}, {"title": rec["title"]}]})
                    if db_doc:
                        db_rec = format_mongo_movie(db_doc)
                except Exception:
                    pass
            similar_movies.append(db_rec if db_rec else tmdb_service.enrich_movie(rec))

    # Third attempt fallback: If ML engine had no matches, find top rated in same genres from MongoDB!
    if not similar_movies and col is not None and is_mongodb_ready() and movie_data.get("genres"):
        try:
            extra_docs = col.find({
                "genres": {"$in": movie_data["genres"][:2]},
                "title": {"$ne": movie_data["title"]}
            }).limit(limit_val)
            for d in extra_docs:
                similar_movies.append(format_mongo_movie(d))
        except Exception:
            pass

    # 3. Retrieve likes and feedback
    user = get_current_user_optional(authorization)
    user_email = user.get("email") if user else None
    likes_info = get_movie_likes_info(movie_id, user_email)
    feedbacks = get_movie_feedback(movie_id, limit=10)

    response_dict = dict(movie_data)
    response_dict["you_might_also_like"] = similar_movies
    response_dict["total_likes"] = likes_info.get("total_likes", 0)
    response_dict["user_liked"] = likes_info.get("user_liked", False)
    response_dict["feedbacks"] = feedbacks
    return MovieDetailResponse(**response_dict)



@app.get("/movies", response_model=MovieSearchResponse, summary="Search Movies by Title")
def search_movies(
    q: str = Query("", description="Partial or full movie title to search for"),
    limit: int = Query(10, ge=1, le=50, description="Max results to return (1-50)")
):
    """
    Search movie dataset by partial case-insensitive title match.
    Enriches candidates with TMDB posters where available.
    """
    if not recommender_engine.is_loaded:
        recommender_engine.load_artifacts()

    raw_results = recommender_engine.search_movies(query=q, limit=limit)
    enriched_results = [tmdb_service.enrich_movie(m) for m in raw_results]

    return MovieSearchResponse(
        query=q,
        count=len(enriched_results),
        results=enriched_results
    )


# ---------------- AUTH & USER INTERACTION ENDPOINTS ----------------

def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """Optional user extractor from Bearer token."""
    if not authorization:
        return None
    return verify_access_token(authorization)


def get_current_user_required(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Required user extractor from Bearer token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication token required. Please sign in.")
    user = verify_access_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session token. Please sign in again.")
    return user


@app.get("/auth/config", summary="Get Public Auth Configuration")
def get_auth_config():
    """Returns Google Client ID for frontend Google Identity Services initialization."""
    return {
        "google_client_id": get_google_client_id()
    }


@app.post("/auth/google", summary="Authenticate with Google OAuth ID Token")
def auth_google(req: GoogleAuthRequest):
    """Verifies Google ID token, saves user in MongoDB Atlas, and returns session JWT."""
    user_info = verify_google_credential(req.credential)
    if not user_info:
        raise HTTPException(status_code=400, detail="Invalid Google OAuth credential token.")

    user = get_or_create_user(user_info)
    token = create_access_token({
        "email": user["email"],
        "name": user["name"],
        "picture": user.get("picture", "")
    })
    return {
        "token": token,
        "user": {
            "email": user["email"],
            "name": user["name"],
            "picture": user.get("picture", ""),
            "watchlist": user.get("watchlist", [])
        }
    }


@app.post("/auth/dev-login", summary="Quick Development / Test Login")
def dev_login(req: DevLoginRequest = Body(default=DevLoginRequest())):
    """Fast login for local testing without requiring Google popup."""
    email = (req.email or "dev.tester@cineml.local").strip().lower()
    name = (req.name or "Dev Tester").strip()
    user_info = {
        "email": email,
        "name": name,
        "picture": f"https://api.dicebear.com/7.x/bottts/svg?seed={email}",
    }
    user = get_or_create_user(user_info)
    token = create_access_token({
        "email": user["email"],
        "name": user["name"],
        "picture": user.get("picture", "")
    })
    return {
        "token": token,
        "user": {
            "email": user["email"],
            "name": user["name"],
            "picture": user.get("picture", ""),
            "watchlist": user.get("watchlist", [])
        }
    }


@app.get("/auth/me", summary="Get Current Logged-in User Profile")
def get_me(user: Dict[str, Any] = Depends(get_current_user_required)):
    """Returns current user details and active watchlist from MongoDB."""
    col = get_users_collection()
    user_doc = None
    if col is not None and is_mongodb_ready():
        user_doc = col.find_one({"email": user["email"]})
    watchlist = user_doc.get("watchlist", []) if user_doc else []
    return {
        "user": {
            "email": user["email"],
            "name": user.get("name", "User"),
            "picture": user.get("picture", ""),
            "watchlist": watchlist
        }
    }


@app.post("/user/watchlist/toggle", summary="Toggle Movie in User Watchlist")
def toggle_watchlist(
    req: WatchlistToggleRequest,
    user: Dict[str, Any] = Depends(get_current_user_required)
):
    """Add or remove movie from user's watchlist in MongoDB."""
    res = toggle_user_watchlist(user["email"], req.movie_id)
    return res


@app.get("/user/watchlist", summary="Get User Watchlist with Movie Details")
def get_watchlist(user: Dict[str, Any] = Depends(get_current_user_required)):
    """Fetches full movie cards for all movies in user's personal watchlist."""
    movie_ids = get_user_watchlist_ids(user["email"])
    movies = []
    col = get_movies_collection()

    for mid in movie_ids:
        doc = None
        if col is not None and is_mongodb_ready():
            try:
                doc = col.find_one({"$or": [{"movie_id": mid}, {"imdb.id": mid}, {"tmdb_id": mid}]})
            except Exception:
                pass
        if doc:
            movies.append(format_mongo_movie(doc))
        else:
            local_match = recommender_engine.get_movie_by_id(mid)
            if local_match:
                movies.append(tmdb_service.enrich_movie(local_match))
            else:
                details = tmdb_service.get_movie_details(mid, fallback_title="Saved Movie")
                movies.append({
                    "movie_id": mid,
                    "tmdb_id": mid,
                    "title": "Saved Movie",
                    "overview": "Saved in your personal watchlist.",
                    "genres": details.get("genres", []),
                    "poster_url": details.get("poster_url"),
                    "vote_average": details.get("vote_average"),
                    "release_date": details.get("release_date")
                })

    return {
        "count": len(movies),
        "movies": movies
    }


@app.post("/movie/{movie_id}/like", summary="Toggle Like for a Movie")
def like_movie(
    movie_id: int,
    user: Dict[str, Any] = Depends(get_current_user_required)
):
    """Toggle user like for a movie and return updated total like count."""
    res = toggle_movie_like(user["email"], movie_id)
    return res


@app.get("/movie/{movie_id}/likes", summary="Get Total Likes and User Like Status")
def get_likes(
    movie_id: int,
    user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """Get total likes count and whether current user liked the movie."""
    user_email = user.get("email") if user else None
    return get_movie_likes_info(movie_id, user_email)


@app.post("/movie/{movie_id}/feedback", summary="Add User Feedback / Review for a Movie")
def post_feedback(
    movie_id: int,
    req: FeedbackCreateRequest,
    user: Dict[str, Any] = Depends(get_current_user_required)
):
    """Post a star rating and comment for a movie, stored in MongoDB Atlas."""
    feedback = add_movie_feedback(movie_id, user, req.rating, req.comment)
    return {
        "status": "ok",
        "feedback": feedback
    }


@app.get("/movie/{movie_id}/feedback", summary="Get Feedback / Reviews for a Movie")
def get_feedback(movie_id: int, limit: int = Query(25, ge=1, le=100)):
    """Fetch user reviews and ratings for a movie, ordered newest first."""
    feedbacks = get_movie_feedback(movie_id, limit=limit)
    return {
        "count": len(feedbacks),
        "feedbacks": feedbacks
    }


# Serve frontend static files if directory exists
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")
