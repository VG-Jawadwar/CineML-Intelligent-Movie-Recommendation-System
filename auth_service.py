import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
import jwt
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from dotenv import load_dotenv

from database import (
    get_users_collection,
    get_likes_collection,
    get_feedback_collection,
    is_mongodb_ready
)

load_dotenv()
logger = logging.getLogger("cineml.auth")

JWT_SECRET = os.getenv("JWT_SECRET", "cineml_local_dev_secret_key_2026")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_DAYS = 7


def get_google_client_id() -> str:
    """
    Resolve Google Client ID from environment variables.
    Dynamically re-reads .env so changes made by the user are reflected
    immediately without requiring a manual server restart.
    """
    try:
        load_dotenv(override=True)
    except Exception:
        pass

    val = (
        os.getenv("GOOGLE_CLIENT_ID")
        or os.getenv("GOOGLE_OAUTH_CLIENT_ID")
        or os.getenv("GOOGLE_CLIENTID")
        or os.getenv("GOOGLE_ID")
        or os.getenv("CLIENT_ID")
        or os.getenv("OAUTH_CLIENT_ID")
        or ""
    ).strip().strip('"').strip("'")

    # Direct file fallback check in case dotenv was cached
    if not val and os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k_upper = k.strip().upper()
                    if k_upper in {"GOOGLE_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_CLIENTID", "GOOGLE_ID", "CLIENT_ID", "OAUTH_CLIENT_ID"}:
                        val = v.strip().strip('"').strip("'")
                        if val:
                            break
        except Exception as e:
            logger.warning(f"Error reading .env for client id: {e}")

    return val


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT session token."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(days=JWT_EXPIRATION_DAYS))
    to_encode.update({"exp": expire, "iat": now})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and verify a signed JWT session token."""
    try:
        clean_token = token.replace("Bearer ", "").strip()
        payload = jwt.decode(clean_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except Exception as e:
        logger.debug(f"JWT verification failed: {e}")
        return None


def verify_google_credential(credential: str) -> Optional[Dict[str, Any]]:
    """
    Verify Google OAuth ID token using google-auth library.
    Returns user info dictionary if valid, None otherwise.
    """
    client_id = get_google_client_id()
    try:
        request = google_requests.Request()
        # Verify token integrity
        id_info = id_token.verify_oauth2_token(
            credential,
            request,
            client_id if client_id else None
        )
        return {
            "email": id_info.get("email"),
            "name": id_info.get("name") or id_info.get("given_name") or "Movie Enthusiast",
            "picture": id_info.get("picture") or "",
            "google_id": id_info.get("sub"),
        }
    except Exception as e:
        logger.warning(f"Google token verification failed: {e}")
        return None


def get_or_create_user(user_info: Dict[str, Any]) -> Dict[str, Any]:
    """Get or create user in MongoDB Atlas."""
    email = user_info.get("email", "").strip().lower()
    if not email:
        raise ValueError("User email is required")

    col = get_users_collection()
    if col is None or not is_mongodb_ready():
        # Fallback in-memory representation if DB is offline
        return {
            "email": email,
            "name": user_info.get("name", "User"),
            "picture": user_info.get("picture", ""),
            "watchlist": [],
        }

    now_iso = datetime.now(timezone.utc).isoformat()
    col.update_one(
        {"email": email},
        {
            "$setOnInsert": {
                "email": email,
                "created_at": now_iso,
                "watchlist": [],
            },
            "$set": {
                "name": user_info.get("name", "User"),
                "picture": user_info.get("picture", ""),
                "updated_at": now_iso,
            }
        },
        upsert=True
    )

    doc = col.find_one({"email": email})
    if doc:
        doc["_id"] = str(doc["_id"])
        return doc
    return user_info


def toggle_user_watchlist(email: str, movie_id: int) -> Dict[str, Any]:
    """Add or remove movie ID from user's watchlist."""
    email = email.strip().lower()
    col = get_users_collection()
    if col is None or not is_mongodb_ready():
        return {"in_watchlist": True, "watchlist": [movie_id]}

    user = col.find_one({"email": email})
    current_watchlist = user.get("watchlist", []) if user else []

    if movie_id in current_watchlist:
        col.update_one({"email": email}, {"$pull": {"watchlist": movie_id}})
        in_watchlist = False
        new_watchlist = [mid for mid in current_watchlist if mid != movie_id]
    else:
        col.update_one({"email": email}, {"$addToSet": {"watchlist": movie_id}})
        in_watchlist = True
        new_watchlist = list(set(current_watchlist + [movie_id]))

    return {"in_watchlist": in_watchlist, "watchlist": new_watchlist}


def get_user_watchlist_ids(email: str) -> List[int]:
    """Retrieve list of movie IDs in user's watchlist."""
    col = get_users_collection()
    if col is None or not is_mongodb_ready():
        return []
    user = col.find_one({"email": email.strip().lower()})
    if not user:
        return []
    return [int(m) for m in user.get("watchlist", [])]


def toggle_movie_like(email: str, movie_id: int) -> Dict[str, Any]:
    """Toggle like status for a movie by a user."""
    email = email.strip().lower()
    movie_id = int(movie_id)
    col = get_likes_collection()

    if col is None or not is_mongodb_ready():
        return {"liked": True, "total_likes": 1}

    existing = col.find_one({"email": email, "movie_id": movie_id})
    if existing:
        col.delete_one({"_id": existing["_id"]})
        liked = False
    else:
        col.insert_one({
            "email": email,
            "movie_id": movie_id,
            "created_at": datetime.now(timezone.utc).isoformat()
        })
        liked = True

    total = col.count_documents({"movie_id": movie_id})
    return {"liked": liked, "total_likes": total}


def get_movie_likes_info(movie_id: int, user_email: Optional[str] = None) -> Dict[str, Any]:
    """Get total likes and whether current user liked the movie."""
    movie_id = int(movie_id)
    col = get_likes_collection()
    if col is None or not is_mongodb_ready():
        return {"total_likes": 0, "user_liked": False}

    total = col.count_documents({"movie_id": movie_id})
    user_liked = False
    if user_email:
        user_liked = bool(col.find_one({"email": user_email.strip().lower(), "movie_id": movie_id}))

    return {"total_likes": total, "user_liked": user_liked}


def add_movie_feedback(
    movie_id: int,
    user_info: Dict[str, Any],
    rating: float,
    comment: str
) -> Dict[str, Any]:
    """Add user feedback/review for a movie."""
    movie_id = int(movie_id)
    email = user_info.get("email", "").strip().lower()
    col = get_feedback_collection()

    doc = {
        "movie_id": movie_id,
        "email": email,
        "name": user_info.get("name", "User"),
        "picture": user_info.get("picture", ""),
        "rating": round(float(rating), 1),
        "comment": comment.strip(),
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    if col is not None and is_mongodb_ready():
        res = col.insert_one(doc)
        doc["_id"] = str(res.inserted_id)

    return doc


def get_movie_feedback(movie_id: int, limit: int = 25) -> List[Dict[str, Any]]:
    """Retrieve list of reviews/feedback for a movie, sorted newest first."""
    movie_id = int(movie_id)
    col = get_feedback_collection()
    if col is None or not is_mongodb_ready():
        return []

    cursor = col.find({"movie_id": movie_id}).sort("created_at", -1).limit(limit)
    feedbacks = []
    for f in cursor:
        f["_id"] = str(f["_id"])
        feedbacks.append(f)
    return feedbacks
