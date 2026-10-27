import os
import logging
from typing import Optional, Dict, Any
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from dotenv import load_dotenv

try:
    import certifi
    _CA_FILE = certifi.where()
except ImportError:
    _CA_FILE = None

# 1. Load local .env or Render secret files if present
if os.path.exists("/etc/secrets/.env"):
    load_dotenv("/etc/secrets/.env")
load_dotenv()

logger = logging.getLogger("cineml.database")

_client: Optional[MongoClient] = None
_db: Optional[Database] = None
_last_error: Optional[str] = None

# Default connection for the student demo Atlas cluster
DEFAULT_ATLAS_URI = "mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"
DEFAULT_DB_NAME = "sample_mflix"


def get_configured_mongo_uri() -> str:
    """Retrieve and sanitize MongoDB connection string from env or fallback."""
    raw = (
        os.getenv("MONGO_URI")
        or os.getenv("MONGODB_URI")
        or os.getenv("MONGO_URL")
        or os.getenv("MONGODB_URL")
        or DEFAULT_ATLAS_URI
    )
    if not raw:
        return ""
    return raw.strip().strip('"').strip("'")


def is_mongodb_configured() -> bool:
    """Check if MONGO_URI is set with a real connection string."""
    uri = get_configured_mongo_uri()
    return bool(uri and "your_mongodb_atlas_connection_string" not in uri)


def get_last_db_error() -> Optional[str]:
    """Return the most recent connection or query error message."""
    return _last_error


def get_client() -> Optional[MongoClient]:
    """Initialize or return existing MongoDB client with TLS verification."""
    global _client, _last_error
    mongo_uri = get_configured_mongo_uri()

    if not mongo_uri or "your_mongodb_atlas_connection_string" in mongo_uri:
        _last_error = "MONGO_URI environment variable is not configured."
        logger.warning(_last_error)
        return None

    if _client is None:
        try:
            logger.info("Connecting to MongoDB Atlas...")
            client_kwargs: Dict[str, Any] = {
                "serverSelectionTimeoutMS": 6000,
                "connectTimeoutMS": 6000,
                "socketTimeoutMS": 6000,
                "appname": "CineML-Recommender"
            }
            if _CA_FILE:
                client_kwargs["tlsCAFile"] = _CA_FILE

            candidate_client = MongoClient(mongo_uri, **client_kwargs)
            # Test ping to verify credentials and network connectivity
            candidate_client.admin.command("ping")
            _client = candidate_client
            _last_error = None
            logger.info("Successfully connected to MongoDB Atlas.")
        except Exception as e:
            _last_error = str(e)
            logger.error(f"Failed to connect to MongoDB Atlas: {e}")
            _client = None
            return None

    return _client


def get_database() -> Optional[Database]:
    """Retrieve CineML database instance."""
    global _db
    client = get_client()
    if client is None:
        return None

    if _db is None:
        db_name = os.getenv("MONGO_DB_NAME", "sample_mflix").strip() or "sample_mflix"
        _db = client[db_name]

    return _db


def get_movies_collection() -> Optional[Collection]:
    """Retrieve movies collection handle."""
    db = get_database()
    if db is None:
        return None
    return db["movies"]


def get_users_collection() -> Optional[Collection]:
    """Retrieve users collection handle."""
    db = get_database()
    if db is None:
        return None
    return db["cineml_users"]


def get_likes_collection() -> Optional[Collection]:
    """Retrieve movie likes collection handle."""
    db = get_database()
    if db is None:
        return None
    return db["cineml_likes"]


def get_feedback_collection() -> Optional[Collection]:
    """Retrieve movie feedback/reviews collection handle."""
    db = get_database()
    if db is None:
        return None
    return db["cineml_feedback"]


def is_mongodb_ready() -> bool:
    """Check if MongoDB Atlas is connected and accessible."""
    global _client, _last_error
    try:
        client = get_client()
        if client is None:
            return False
        client.admin.command("ping")
        _last_error = None
        return True
    except Exception as e:
        _last_error = str(e)
        _client = None
        return False

