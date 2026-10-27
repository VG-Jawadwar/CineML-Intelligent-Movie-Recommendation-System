import os
import sys
import pickle
import argparse
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_movies_collection, is_mongodb_ready
from tmdb_service import create_svg_placeholder

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cineml.seeder")


def parse_args():
    parser = argparse.ArgumentParser(description="Seed movies from artifacts/movies.pkl into MongoDB Atlas.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Number of movies to seed into MongoDB Atlas (default: 5000, use 0 for all 173,268)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for bulk insertion (default: 500)."
    )
    return parser.parse_args()


def seed_database():
    args = parse_args()

    if not is_mongodb_ready():
        logger.error(
            "MongoDB Atlas is not reachable. Please ensure MONGO_URI is set in .env "
            "and your IP is whitelisted in MongoDB Atlas Network Access."
        )
        sys.exit(1)

    movies_collection = get_movies_collection()
    if movies_collection is None:
        logger.error("Could not obtain movies collection from database.")
        sys.exit(1)

    artifacts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "movies.pkl")
    if not os.path.exists(artifacts_path):
        logger.error(f"Movie artifact not found at {artifacts_path}")
        sys.exit(1)

    logger.info(f"Loading movie metadata from {artifacts_path}...")
    with open(artifacts_path, "rb") as f:
        movies_df = pickle.load(f).reset_index(drop=True)

    total_available = len(movies_df)
    target_count = total_available if args.limit <= 0 else min(args.limit, total_available)
    logger.info(f"Preparing to seed {target_count:,} movies into MongoDB Atlas...")

    # Create indexes
    logger.info("Setting up database indexes on 'movie_id', 'genres', 'title', and 'vote_average'...")
    movies_collection.create_index("movie_id", unique=True)
    movies_collection.create_index("genres")
    movies_collection.create_index("title")
    movies_collection.create_index("vote_average")

    batch: List[Dict[str, Any]] = []
    inserted_or_updated = 0

    for i in range(target_count):
        row = movies_df.iloc[i]
        movie_id = int(row["id"])
        title = str(row["title"])

        # Overview
        overview_raw = row["overview"]
        if isinstance(overview_raw, list):
            overview = " ".join(str(w) for w in overview_raw)
        elif not overview_raw or str(overview_raw).lower() == "nan":
            overview = "No synopsis available."
        else:
            overview = str(overview_raw)

        # Genres
        genres_raw = row["genres"]
        genres: List[str] = []
        if isinstance(genres_raw, list):
            genres = [str(g).strip() for g in genres_raw if str(g).strip()]
        elif isinstance(genres_raw, str) and genres_raw.strip():
            genres = [g.strip() for g in genres_raw.split(",") if g.strip()]

        # Generate standard poster fallback URL or SVG
        poster_url = f"https://image.tmdb.org/t/p/w500"  # Will be dynamically loaded or fallback
        # Synthetic base vote average based on row position/id for realistic sorting
        vote_average = round(5.5 + ((movie_id * 13) % 40) / 10.0, 1)  # Range 5.5 - 9.4

        doc = {
            "movie_id": movie_id,
            "tmdb_id": movie_id,
            "row_index": i,
            "title": title,
            "overview": overview,
            "genres": genres,
            "vote_average": vote_average,
            "poster_url": None,
        }
        batch.append(doc)

        if len(batch) >= args.batch_size:
            from pymongo import UpdateOne
            operations = [
                UpdateOne({"movie_id": item["movie_id"]}, {"$set": item}, upsert=True)
                for item in batch
            ]
            movies_collection.bulk_write(operations, ordered=False)
            inserted_or_updated += len(batch)
            logger.info(f"Progress: {inserted_or_updated:,} / {target_count:,} movies synchronized.")
            batch = []

    if batch:
        from pymongo import UpdateOne
        operations = [
            UpdateOne({"movie_id": item["movie_id"]}, {"$set": item}, upsert=True)
            for item in batch
        ]
        movies_collection.bulk_write(operations, ordered=False)
        inserted_or_updated += len(batch)

    logger.info(f"Done! Successfully seeded {inserted_or_updated:,} movies into MongoDB Atlas.")


if __name__ == "__main__":
    seed_database()
