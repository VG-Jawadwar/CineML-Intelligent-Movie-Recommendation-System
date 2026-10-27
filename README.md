# CineML — Intelligent Movie Recommendation System

<p align="center">
  <img src="cineMl.png" alt="CineML Logo" width="55%">
</p>

<p align="center">
  <strong>An End-to-End Content-Based Movie Recommendation System powered by Scikit-Learn, FastAPI, TMDB Integration, and Modern Vanilla Web Technologies.</strong>
</p>

<p align="center">
  <a href="#-project-abstract">Abstract</a> •
  <a href="#-ml-architecture--algorithm">ML Architecture</a> •
  <a href="#-system-architecture">System Architecture</a> •
  <a href="#-api-endpoints-documentation">API Docs</a> •
  <a href="#-windows-setup--execution-guide">Windows Run Guide</a> •
  <a href="#-tmdb-integration">TMDB Setup</a> •
  <a href="#-automated-testing">Testing</a>
</p>

---

## 📌 Project Abstract

**CineML** is an academic Machine Learning mini-project that demonstrates the complete lifecycle of a real-world recommendation system. It bridges offline machine learning pipelines with high-performance backend serving and a responsive, cinematic frontend interface.

The system utilizes precomputed content-based vector embeddings trained on **173,268 movies** extracted from The Movie Database (TMDB). By converting textual features (plot overviews, genres, keywords, and production details) into numerical feature spaces using **TF-IDF vectorization** and measuring document proximity through **Cosine Similarity via K-Nearest Neighbors ($k=101$)**, CineML provides instantaneous, relevant movie recommendations without cold-start user tracking or privacy intrusion.

---

## 🎯 Problem Definition & Objectives

### Problem Definition
Modern video streaming catalogs contain hundreds of thousands of titles. Traditional collaborative filtering algorithms suffer from severe **cold-start problems** for new users or sparse interaction matrices. Users frequently seek titles that share narrative themes, tone, and storytelling elements with films they already enjoy, independent of aggregate rating popularity.

### Objectives
1. **Model Reuse & Preservation**: Preserve and deploy the existing content-based ML model and precomputed neighbor matrix without retraining or altering vector indexing.
2. **Production-Conscious API**: Build a high-throughput REST API using **FastAPI** to serve health checks, case-insensitive partial search, and top-$K$ recommendations.
3. **Graceful TMDB Integration**: Fetch rich movie metadata and poster artwork dynamically from The Movie Database (TMDB) API, with robust in-memory caching, timeouts, and fallback placeholders when credentials or network are offline.
4. **Cinematic Web Interface**: Deliver an accessible, responsive dark-themed web application using modern HTML5, CSS3, and Vanilla JavaScript with debounced search and safe DOM manipulation.
5. **Academic Rigor & Testing**: Implement comprehensive automated testing using `pytest` and `TestClient` covering schema validation, edge cases, and mocked external calls.

---

## 📊 Dataset & Preprocessing Pipeline

### Verified Dataset Source & Statistics
- **Source**: TMDB Movies Dataset (Kaggle: `asaniczka/tmdb-movies-dataset-2023-930k-movies`)
- **Raw Size**: 1,209,193 raw records across 24 columns
- **Cleaned Dataset in Model Artifacts**: **173,268 movies** (`artifacts/movies.pkl`)
- **Neighbor Matrix Dimension**: **173,268 × 101 matrix** (`artifacts/all_neighbors.pkl`)
- **Feature Set**: `['id', 'title', 'overview', 'genres', 'keywords', 'production_companies', 'tags']`
- **Identifier Mapping**: The `id` column directly corresponds to official **TMDB Movie IDs** (e.g., Inception = `27205`, Interstellar = `157336`, The Dark Knight = `155`).

### Preprocessing Steps (from `Recommender system.ipynb`)
1. **Filtering & Deduplication**: Dropped null overviews and non-essential columns.
2. **Feature Extraction**: Extracted names from stringified JSON dictionaries for genres, keywords, and production companies.
3. **Tag Synthesis**: Concatenated tokenized `overview`, `genres`, `keywords`, and `production_companies` into a unified `tags` document.
4. **Stemming**: Applied the Porter Stemmer (`nltk.stem.porter.PorterStemmer`) to reduce words to their linguistic root.

---

## 🧠 ML Architecture & Algorithm

### Content-Based Filtering
Content-Based Filtering recommends items based on intrinsic feature similarity rather than collaborative user behavior. 

```
┌─────────────────────────────────┐
│ Raw Movie Text Metadata         │
│ (Overview, Genres, Keywords)    │
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ Text Normalization & Stemming   │
│ (Porter Stemmer)                │
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ TF-IDF Vectorizer               │
│ (Term Frequency - Inverse Doc)  │
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ NearestNeighbors (k=101)        │
│ Metric: Cosine Distance         │
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ artifacts/all_neighbors.pkl     │
│ [Row i -> [Self, Rec1, Rec2..]] │
└─────────────────────────────────┘
```

1. **TF-IDF Vectorization**:
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
   This scores rare, informative keywords (e.g., "dystopia", "heist", "superhero") higher than generic stopwords.

2. **Cosine Similarity**:
   $$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
   Cosine similarity measures the orientation of two feature vectors regardless of length variations in movie overviews.

3. **Nearest Neighbors Matrix**:
   A brute-force cosine distance model precomputed the top 100 nearest neighbors for each film. Row index `0` represents the movie itself, and indices `1..100` represent the most similar movies ordered by descending cosine similarity.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Browser                          │
│   (HTML5 / Vanilla CSS3 / JavaScript App / Debounced UI)    │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / REST (JSON)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                         │
│  ┌───────────────────────┐       ┌────────────────────────┐ │
│  │   api.py              │       │  tmdb_service.py       │ │
│  │  - GET /              │       │  - Poster Fetcher      │ │
│  │  - GET /movies        │◄─────►│  - In-Memory Cache     │ │
│  │  - GET /recommend     │       │  - SVG Fallback Gen    │ │
│  │  - GET /movie/{id}    │       └───────────┬────────────┘ │
│  └───────────┬───────────┘                   │              │
│              │                               │ HTTPS        │
│              ▼                               ▼              │
│  ┌───────────────────────┐       ┌────────────────────────┐ │
│  │  recommender.py       │       │    TMDB Cloud API      │ │
│  │  - movies.pkl         │       │ (api.themoviedb.org/3) │ │
│  │  - all_neighbors.pkl  │       └────────────────────────┘ │
│  │  - NumPy 1/2 Shim     │                                  │
│  └───────────────────────┘                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔌 API Endpoints Documentation

### 1. Health Check
- **Endpoint**: `GET /`
- **Description**: Verifies API operational status, model readiness, and movie count.
- **Sample Response**:
  ```json
  {
    "status": "ok",
    "service": "CineML Movie Recommendation API",
    "model_loaded": true,
    "movies_count": 173268,
    "tmdb_configured": false
  }
  ```

### 2. Movie Search
- **Endpoint**: `GET /movies?q={query}&limit={limit}`
- **Parameters**: `q` (string, required), `limit` (integer, 1–50, default: 10)
- **Description**: Fast case-insensitive local title search with prefix and substring ranking.
- **Sample Request**: `GET /movies?q=inception&limit=1`
- **Sample Response**:
  ```json
  {
    "query": "inception",
    "count": 1,
    "results": [
      {
        "movie_id": 27205,
        "tmdb_id": 27205,
        "title": "Inception",
        "overview": "Cobb, a skilled thief who commits corporate espionage...",
        "genres": ["Action", "Science Fiction"],
        "release_date": "2010-07-15",
        "poster_url": "https://image.tmdb.org/t/p/w500/edv5CZvWj09upOsy2Y6IwDhK8bt.jpg",
        "vote_average": 8.4
      }
    ]
  }
  ```

### 3. Movie Recommendations
- **Endpoint**: `GET /recommend?movie_id={movie_id}&limit={limit}`
- **Parameters**: `movie_id` (integer, required), `limit` (integer, 1–50, default: 10)
- **Description**: Returns top-$K$ recommended movies for the given movie ID using the precomputed content similarity matrix.
- **Sample Request**: `GET /recommend?movie_id=27205&limit=2`
- **Sample Response**:
  ```json
  {
    "selected_movie": {
      "movie_id": 27205,
      "tmdb_id": 27205,
      "title": "Inception",
      "overview": "Cobb, a skilled thief...",
      "genres": ["Action", "Science Fiction"],
      "release_date": "2010-07-15",
      "poster_url": "https://image.tmdb.org/t/p/w500/edv5CZvWj09upOsy2Y6IwDhK8bt.jpg"
    },
    "count": 2,
    "recommendations": [
      {
        "movie_id": 1293777,
        "tmdb_id": 1293777,
        "title": "Past Never Dies",
        "overview": "A detective haunted by his past...",
        "genres": ["Thriller"],
        "release_date": null,
        "poster_url": "data:image/svg+xml;utf8,..."
      },
      {
        "movie_id": 220024,
        "tmdb_id": 220024,
        "title": "A Time To Die",
        "overview": "Undercover operative fighting against time...",
        "genres": ["Action"],
        "release_date": null,
        "poster_url": "data:image/svg+xml;utf8,..."
      }
    ]
  }
  ```

### 4. Single Movie Metadata
- **Endpoint**: `GET /movie/{movie_id}`
- **Parameters**: `movie_id` (path integer)
- **Description**: Returns local dataset metadata and enriched TMDB poster for a single movie.

---

## 🔑 TMDB Integration & Attribution

### Authentication
CineML supports TMDB API v3 (`TMDB_API_KEY`) and v4 Bearer Tokens (`TMDB_ACCESS_TOKEN`).
- Create an account at [The Movie Database (TMDB)](https://www.themoviedb.org/).
- Generate an API Key under **Account Settings > API**.
- Add the key to `.env` in the project root:
  ```env
  TMDB_API_KEY=your_actual_tmdb_api_key_here
  ```

### Resilient Offline & Placeholder Architecture
- **Missing / Placeholder Keys**: If the API key is not yet set or contains placeholder text (e.g. `your_tmdb_api_key_here`), the backend **does not crash or hang**. It immediately generates a clean cinematic SVG poster placeholder.
- **Timeouts & Caching**: External HTTP calls use a 2.5-second timeout with automated retries and thread-safe in-memory caching for both positive and negative lookups.

### TMDB Attribution Requirement
> *This product uses the TMDB API but is not endorsed or certified by TMDB.*
> 
> *The TMDB logo and trademark are property of The Movie Database.*

---

## 💻 Windows Setup & Execution Guide

### Prerequisites
- Windows 10 / 11
- Python 3.8+ (Python 3.8, 3.9, 3.10, 3.11, 3.12, or 3.13)
- PowerShell or Windows Command Prompt

### Step 1: Clone Repository & Open Directory
```powershell
cd d:\AIMLL_Mini_Project_Movie_Recommendation_System\CineML-Movie-Recommender
```

### Step 2: Create & Activate Virtual Environment
```powershell
# Using Python launcher
py -3.8 -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# (If PowerShell displays execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser)
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env` if not already created:
```powershell
copy .env.example .env
```
Open `.env` and replace `your_tmdb_api_key_here` with your TMDB API key whenever you wish. The application will function with full recommendations and local metadata even before replacing the key.

### Step 5: Start the Backend Server (Terminal 1)
```powershell
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive API Documentation (Swagger UI): `http://127.0.0.1:8000/docs`
- Direct Web Application Endpoint: `http://127.0.0.1:8000/app/`

### Step 6: Start the Frontend Server (Terminal 2 - Optional)
You can access the frontend directly through FastAPI at `http://127.0.0.1:8000/app/`, or serve the frontend folder independently:
```powershell
cd frontend
python -m http.server 5500
```
Then open `http://127.0.0.1:5500` in any web browser.

---

## 🧪 Automated Testing

Automated testing is configured using `pytest` and `fastapi.testclient.TestClient`. External TMDB requests are mocked to allow isolated, reproducible testing without network dependencies.

Run all tests:
```powershell
pytest tests/test_api.py -v
```

### Verified Test Suite (13 / 13 Passing)
- `test_health_check`: Validates API status and model load state.
- `test_search_movies_matching`: Validates case-insensitive search and result fields.
- `test_search_movies_no_match`: Validates empty results for non-existent queries.
- `test_recommendation_valid_id`: Validates recommendation count and verifies self-exclusion.
- `test_recommendation_invalid_id`: Validates 404 response for unknown movie IDs.
- `test_recommendation_invalid_limit`: Validates Pydantic boundary checks ($1 \le \text{limit} \le 50$).
- `test_single_movie_endpoint`: Validates retrieval of single movie metadata.
- `test_single_movie_not_found`: Validates 404 response on invalid single ID.
- `test_duplicate_titles_distinct_ids`: Verifies that duplicate movie titles maintain stable distinct IDs.
- `test_indexing_consistency`: Confirms row indices in `movies.pkl` map 1:1 with `all_neighbors.pkl`.
- `test_tmdb_mock_timeout`: Verifies fallback handling when TMDB times out.
- `test_missing_tmdb_credentials`: Verifies offline SVG placeholder generation without crashing.
- `test_tmdb_successful_mock`: Verifies live metadata and poster path enrichment.

---

## 📈 Evaluation & Model Limitations

### Performance Evaluation Methodology
For academic review, content-based recommendation systems can be evaluated using:
1. **Catalog Coverage**: Proportion of all 173,268 catalog titles that can be retrieved as recommendations across diverse seeds.
2. **Retrieval Latency**: Milliseconds from request to response. CineML delivers recommendations in **< 15 milliseconds** due to precomputed index structures.
3. **Reproducible Qualitative Verification**: Compare top recommendations for iconic titles across genres (e.g. *Inception* $\rightarrow$ psychological sci-fi/thrillers; *Avatar* $\rightarrow$ space exploration/aliens).

### Model Limitations & Future Enhancements
- **Static Vocabulary**: The TF-IDF matrix is bounded to the precomputed dataset vocabulary; newly released films require index regeneration.
- **Lack of User Personalization**: Content-based filtering does not incorporate individual watch history or user collaborative ratings.
- **Future Enhancements**: Implement a hybrid recommender combining Neural Collaborative Filtering (NCF) with BM25 or Sentence-Transformers embeddings (e.g. `all-MiniLM-L6-v2`).

---

## ⚖️ License & Attribution

- **Original Project**: Developed by: Vaibhav Jawadwar
                                      Prabodh Kulkarni
                                      Sandip Patil
                                      Omkar Shelke
- **Academic Mini-Project Extensions**: Full-Stack FastAPI backend, TMDB integration, modern dark-mode frontend, NumPy 2 compatibility shim, and automated test suite.
- **Data Source**: [TMDB Movies Dataset 2023](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies).
- **License**: MIT License.
