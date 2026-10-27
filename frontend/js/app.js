/**
 * CineML — Front-End Application Logic
 * Integrates with FastAPI backend, MongoDB Atlas preferences,
 * Google OAuth, User Watchlists, Movie Likes & Community Feedback,
 * and ML Nearest Neighbors 'You Might Also Like' recommendations.
 */

// Dynamic API Base URL
// When loaded on Render or direct FastAPI server: uses relative paths ('')
// When loaded via local development servers (e.g., Live Server port 5500 or file://):
// tries local port 8000 first, and automatically falls back to deployed Render API!
const RENDER_API_FALLBACK = 'https://cineml-movie-recommender.onrender.com';
let API_BASE = '';

if (
  window.location.protocol === 'file:' || 
  window.location.port === '5500' || 
  window.location.port === '3000' || 
  window.location.port === '5173'
) {
  API_BASE = 'http://127.0.0.1:8000';
}

// Application State
const state = {
  selectedGenres: new Set(['Action', 'Science Fiction']), // Sensible default
  selectedMovieId: null,
  activeSuggestionIndex: -1,
  currentSearchAbortController: null,
  debounceTimer: null,

  // Auth State
  authToken: localStorage.getItem('cineml_auth_token') || null,
  currentUser: null,
  userWatchlist: new Set(), // Set of numeric movie_ids

  // Active Modal Movie State
  currentModalMovieId: null,
  currentModalLikes: 0,
  currentModalUserLiked: false,
  selectedRating: 10,

  // Navigation / View State
  currentView: 'recommendations', // 'recommendations' | 'watchlist'
  lastRecommendationMovies: [],
  googleClientId: null,
};

// DOM Cache
const elements = {
  // Navigation & Status
  statusDot: document.getElementById('status-dot'),
  statusText: document.getElementById('status-text'),
  notificationBanner: document.getElementById('notification-banner'),
  notificationMsg: document.getElementById('notification-message'),
  notificationClose: document.getElementById('notification-close'),

  // Auth Controls (Navbar)
  navWatchlistBtn: document.getElementById('nav-watchlist-btn'),
  navWatchlistCount: document.getElementById('nav-watchlist-count'),
  viewWatchlistCount: document.getElementById('view-watchlist-count'),
  authContainer: document.getElementById('nav-auth-container'),
  authLoggedOut: document.getElementById('auth-logged-out'),
  authLoggedIn: document.getElementById('auth-logged-in'),
  gIdSigninBtn: document.getElementById('g_id_signin_btn'),
  btnDevLogin: document.getElementById('btn-dev-login'),
  btnSignout: document.getElementById('btn-signout'),
  userAvatar: document.getElementById('user-avatar'),
  userDisplayName: document.getElementById('user-display-name'),

  // View Tabs
  tabRecommendations: document.getElementById('tab-recommendations'),
  tabWatchlist: document.getElementById('tab-watchlist'),

  // Preferences Form
  genreChipsContainer: document.getElementById('genre-chips-container'),
  ratingSelect: document.getElementById('rating-select'),
  limitSelect: document.getElementById('limit-select'),
  discoverBtn: document.getElementById('discover-btn'),

  // Title Search Box
  searchInput: document.getElementById('movie-search-input'),
  clearSearchBtn: document.getElementById('clear-search-btn'),
  suggestionsDropdown: document.getElementById('search-suggestions-dropdown'),
  getRecsBtn: document.getElementById('get-recommendations-btn'),

  // Results Section
  resultsHeading: document.getElementById('results-heading'),
  resultsSubtitle: document.getElementById('recommendations-subtitle'),
  resultsBadge: document.getElementById('results-count-badge'),
  resultsCountText: document.getElementById('results-count-text'),
  loadingSpinner: document.getElementById('loading-spinner-container'),
  recsGrid: document.getElementById('recommendations-grid'),
  initialEmptyState: document.getElementById('initial-empty-state'),

  // Movie Details Modal
  modalOverlay: document.getElementById('movie-detail-modal'),
  modalCloseBtn: document.getElementById('modal-close-btn'),
  modalPoster: document.getElementById('modal-poster'),
  modalTitle: document.getElementById('modal-movie-title'),
  modalRating: document.getElementById('modal-rating'),
  modalYear: document.getElementById('modal-year'),
  modalGenres: document.getElementById('modal-genres'),
  modalOverview: document.getElementById('modal-overview'),
  modalRecsTrack: document.getElementById('modal-recs-track'),

  // Modal Action Buttons (Watchlist & Like)
  modalWatchlistBtn: document.getElementById('modal-watchlist-btn'),
  modalWatchlistLabel: document.getElementById('modal-watchlist-label'),
  modalLikeBtn: document.getElementById('modal-like-btn'),
  modalLikeLabel: document.getElementById('modal-like-label'),

  // Feedback / Reviews Section
  feedbackCountLabel: document.getElementById('feedback-count-label'),
  movieFeedbackForm: document.getElementById('movie-feedback-form'),
  feedbackStarPicker: document.getElementById('feedback-star-picker'),
  ratingValueBadge: document.getElementById('rating-value-badge'),
  feedbackCommentInput: document.getElementById('feedback-comment-input'),
  feedbackAuthHint: document.getElementById('feedback-auth-hint'),
  btnSubmitFeedback: document.getElementById('btn-submit-feedback'),
  modalFeedbackList: document.getElementById('modal-feedback-list'),
};

// ---------------- Initialization ----------------

document.addEventListener('DOMContentLoaded', async () => {
  await initApiStatus();
  initStarPicker();
  initGenres();
  attachEventListeners();
  await initAuth();
  // Automatically load initial recommendations based on default genres
  loadPreferenceRecommendations();
});

// ---------------- API Status Checker ----------------

async function initApiStatus() {
  try {
    let res = null;
    let endpoint = API_BASE ? `${API_BASE}/health` : '/health';

    try {
      res = await fetch(endpoint, { method: 'GET' });
      if (!res.ok) {
        res = await fetch(API_BASE ? `${API_BASE}/` : '/', { method: 'GET' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
      }
    } catch (primaryErr) {
      // If local dev server is offline, auto-connect to live Render backend
      if (API_BASE && API_BASE !== RENDER_API_FALLBACK) {
        console.warn(`Local API at ${API_BASE} not reachable. Auto-connecting to live Render backend at ${RENDER_API_FALLBACK}...`);
        API_BASE = RENDER_API_FALLBACK;
        res = await fetch(`${API_BASE}/health`, { method: 'GET' });
        if (!res.ok) {
          res = await fetch(`${API_BASE}/`, { method: 'GET' });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
        }
      } else {
        throw primaryErr;
      }
    }

    const data = await res.json();

    if (data.status === 'ok') {
      elements.statusDot.className = 'status-indicator ready';
      const count = data.mongodb_movies_count || data.local_movies_count || 0;
      let statusLabel = data.mongodb_connected
        ? `MongoDB Atlas Ready (${Number(count).toLocaleString()} movies)`
        : `Local ML Engine Ready (${Number(count).toLocaleString()} movies)`;

      if (data.tmdb_configured) {
        statusLabel += ' • TMDB Active';
      }
      elements.statusText.textContent = statusLabel;
      hideNotification();
    }
  } catch (err) {
    elements.statusDot.className = 'status-indicator error';
    elements.statusText.textContent = 'Backend Offline';
    showNotification('Backend Offline: Could not reach CineML API. If running locally, start server with: py -m uvicorn api:app --reload', 'error');
  }
}

// ---------------- Initialize Genres from API ----------------

async function initGenres() {
  try {
    const res = await fetch(`${API_BASE}/genres`);
    if (!res.ok) return;
    const data = await res.json();
    if (!data.genres || !data.genres.length) return;

    elements.genreChipsContainer.innerHTML = '';
    data.genres.forEach(genre => {
      const chip = document.createElement('div');
      chip.className = `genre-chip ${state.selectedGenres.has(genre) ? 'active' : ''}`;
      chip.dataset.genre = genre;
      chip.textContent = genre;
      elements.genreChipsContainer.appendChild(chip);
    });
  } catch (e) {
    // Keep fallback static genre chips if offline
  }
}

// ---------------- Google OAuth & Authentication ----------------

async function initAuth() {
  await fetchAndSetupGoogleAuth();

  // 2. Check if user has an existing session token
  if (state.authToken) {
    try {
      const meRes = await fetch(`${API_BASE}/auth/me`, {
        headers: getAuthHeaders(),
      });
      if (meRes.ok) {
        const meData = await meRes.json();
        state.currentUser = meData.user;
        state.userWatchlist = new Set((meData.user.watchlist || []).map(Number));
        updateAuthUI();
        updateWatchlistBadges();
        return;
      } else {
        // Token expired or invalid
        localStorage.removeItem('cineml_auth_token');
        state.authToken = null;
        state.currentUser = null;
        state.userWatchlist.clear();
      }
    } catch (e) {
      console.warn('Failed to verify session token:', e);
    }
  }

  updateAuthUI();
  updateWatchlistBadges();
}

async function fetchAndSetupGoogleAuth() {
  try {
    const configRes = await fetch(`${API_BASE}/auth/config`);
    if (configRes.ok) {
      const configData = await configRes.json();
      state.googleClientId = (configData.google_client_id || '').trim();
      if (state.googleClientId) {
        setupGoogleSignInButton(state.googleClientId);
        return true;
      } else {
        renderGooglePromptFallback();
        // Periodically check if user saves GOOGLE_CLIENT_ID in .env
        setTimeout(fetchAndSetupGoogleAuth, 4000);
      }
    }
  } catch (e) {
    console.warn('Could not load auth config:', e);
    renderGooglePromptFallback();
  }
  return false;
}

function renderGooglePromptFallback() {
  if (!elements.gIdSigninBtn || state.googleClientId) return;
  elements.gIdSigninBtn.innerHTML = `
    <button type="button" class="btn-google-prompt" id="btn-google-prompt" title="Log in with Google">
      <svg width="15" height="15" viewBox="0 0 24 24">
        <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
        <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
        <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
        <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
      </svg>
      <span>Log-In</span>
    </button>
  `;

  const btn = document.getElementById('btn-google-prompt');
  if (btn) {
    btn.addEventListener('click', () => {
      showNotification('Google Sign-In: If you added GOOGLE_CLIENT_ID to .env, press Ctrl+S in your editor to save the file.', 'info');
    });
  }
}

function setupGoogleSignInButton(clientId, retries = 0) {
  if (window.google && window.google.accounts && window.google.accounts.id) {
    try {
      window.google.accounts.id.initialize({
        client_id: clientId,
        callback: handleGoogleCredentialResponse,
        auto_select: false,
        cancel_on_tap_outside: true,
      });

      if (elements.gIdSigninBtn) {
        elements.gIdSigninBtn.innerHTML = '';
        window.google.accounts.id.renderButton(
          elements.gIdSigninBtn,
          {
            theme: 'filled_black',
            size: 'medium',
            shape: 'pill',
            text: 'signin',
          }
        );
      }
    } catch (err) {
      console.warn('Google Identity Services initialization warning:', err);
    }
  } else if (retries < 25) {
    // If GIS script hasn't fully loaded, retry every 300ms up to 25 times
    setTimeout(() => {
      setupGoogleSignInButton(clientId, retries + 1);
    }, 300);
  }
}

async function handleGoogleCredentialResponse(response) {
  try {
    const res = await fetch(`${API_BASE}/auth/google`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ credential: response.credential }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Google sign-in failed');
    }

    const data = await res.json();
    state.authToken = data.token;
    localStorage.setItem('cineml_auth_token', data.token);
    state.currentUser = data.user;
    state.userWatchlist = new Set((data.user.watchlist || []).map(Number));

    updateAuthUI();
    updateWatchlistBadges();
    refreshCardBadges();
    updateModalActionButtons();
    showNotification(`Welcome back, ${data.user.name}! Signed in successfully with Google.`, 'info');
  } catch (err) {
    showNotification(`Sign-in error: ${err.message}`, 'error');
  }
}

async function handleDevLogin() {
  try {
    const res = await fetch(`${API_BASE}/auth/dev-login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: 'local.tester@cineml.org',
        name: 'Local Reviewer',
      }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    state.authToken = data.token;
    localStorage.setItem('cineml_auth_token', data.token);
    state.currentUser = data.user;
    state.userWatchlist = new Set((data.user.watchlist || []).map(Number));

    updateAuthUI();
    updateWatchlistBadges();
    refreshCardBadges();
    updateModalActionButtons();
    showNotification(`Signed in as ${data.user.name} (Local Test Account). Ready to save watchlist, like & review!`, 'info');
  } catch (err) {
    showNotification(`Test login failed: ${err.message}`, 'error');
  }
}

function handleSignout() {
  localStorage.removeItem('cineml_auth_token');
  state.authToken = null;
  state.currentUser = null;
  state.userWatchlist.clear();

  updateAuthUI();
  updateWatchlistBadges();
  refreshCardBadges();
  updateModalActionButtons();

  if (state.currentView === 'watchlist') {
    switchToRecommendationsView();
  }

  showNotification('You have signed out successfully.', 'info');
}

function updateAuthUI() {
  if (state.currentUser) {
    elements.authLoggedOut.classList.add('hidden');
    elements.authLoggedIn.classList.remove('hidden');

    elements.userAvatar.src = state.currentUser.picture || 'https://api.dicebear.com/7.x/bottts/svg?seed=cineml';
    elements.userDisplayName.textContent = state.currentUser.name || state.currentUser.email || 'User';
    elements.feedbackAuthHint.textContent = `Posting as ${state.currentUser.name || 'User'}`;
  } else {
    elements.authLoggedIn.classList.add('hidden');
    elements.authLoggedOut.classList.remove('hidden');
    elements.feedbackAuthHint.textContent = 'Log in with Google to post a review';
  }
}

function getAuthHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  if (state.authToken) {
    headers['Authorization'] = `Bearer ${state.authToken}`;
  }
  return headers;
}

function updateWatchlistBadges() {
  const count = state.userWatchlist.size;
  if (elements.navWatchlistCount) elements.navWatchlistCount.textContent = count;
  if (elements.viewWatchlistCount) elements.viewWatchlistCount.textContent = count;
}

// ---------------- Watchlist & Likes Actions ----------------

async function toggleWatchlist(movieId, e) {
  if (e) e.stopPropagation();

  if (!state.currentUser) {
    showNotification('Please log in to add movies to your Watchlist!', 'info');
    return;
  }

  const numId = Number(movieId);
  try {
    const res = await fetch(`${API_BASE}/user/watchlist/toggle`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ movie_id: numId }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (data.in_watchlist) {
      state.userWatchlist.add(numId);
      showNotification('Movie added to your Watchlist! 🔖', 'info');
    } else {
      state.userWatchlist.delete(numId);
      showNotification('Movie removed from your Watchlist.', 'info');
    }

    updateWatchlistBadges();
    refreshCardBadges(numId);
    updateModalActionButtons();

    // If viewing Watchlist tab and removed movie, remove card from view
    if (state.currentView === 'watchlist') {
      const card = elements.recsGrid.querySelector(`.movie-card[data-movie-id="${numId}"]`);
      if (card && !data.in_watchlist) {
        card.style.opacity = '0';
        card.style.transform = 'scale(0.85)';
        setTimeout(() => {
          card.remove();
          const remaining = elements.recsGrid.querySelectorAll('.movie-card').length;
          elements.resultsCountText.textContent = `${remaining} saved`;
          if (remaining === 0) {
            elements.recsGrid.innerHTML = `
              <div class="empty-state">
                <div class="empty-state-icon">🔖</div>
                <h3 class="empty-state-title">Your Watchlist is Empty</h3>
                <p class="empty-state-desc">Browse movies and click the bookmark icon 🔖 to save them here for later!</p>
              </div>
            `;
          }
        }, 250);
      }
    }
  } catch (err) {
    showNotification(`Could not update watchlist: ${err.message}`, 'error');
  }
}

async function toggleLike(movieId, e) {
  if (e) e.stopPropagation();

  if (!state.currentUser) {
    showNotification('Please log in to like movies!', 'info');
    return;
  }

  const numId = Number(movieId);
  try {
    const res = await fetch(`${API_BASE}/movie/${numId}/like`, {
      method: 'POST',
      headers: getAuthHeaders(),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    state.currentModalLikes = data.total_likes;
    state.currentModalUserLiked = data.liked;
    updateModalActionButtons();
  } catch (err) {
    showNotification(`Could not like movie: ${err.message}`, 'error');
  }
}

function refreshCardBadges(specificMovieId = null) {
  const cards = specificMovieId 
    ? elements.recsGrid.querySelectorAll(`.movie-card[data-movie-id="${specificMovieId}"]`)
    : elements.recsGrid.querySelectorAll('.movie-card');

  cards.forEach(card => {
    const mid = Number(card.dataset.movieId);
    const watchlistBtn = card.querySelector('.card-btn-watchlist');
    if (watchlistBtn) {
      const inWatchlist = state.userWatchlist.has(mid);
      watchlistBtn.classList.toggle('in-watchlist', inWatchlist);
      watchlistBtn.textContent = inWatchlist ? '🔖 Saved' : '🔖';
      watchlistBtn.title = inWatchlist ? 'Remove from Watchlist' : 'Add to Watchlist';
    }
  });
}

function updateModalActionButtons() {
  if (!state.currentModalMovieId) return;

  const inWatchlist = state.userWatchlist.has(state.currentModalMovieId);
  elements.modalWatchlistBtn.classList.toggle('active-watchlist', inWatchlist);
  elements.modalWatchlistLabel.textContent = inWatchlist ? 'In Watchlist (Saved)' : 'Add to Watchlist';

  elements.modalLikeBtn.classList.toggle('active-like', state.currentModalUserLiked);
  elements.modalLikeLabel.textContent = `${state.currentModalLikes} ${state.currentModalLikes === 1 ? 'Like' : 'Likes'}`;
  const likeIcon = elements.modalLikeBtn.querySelector('.btn-icon');
  if (likeIcon) {
    likeIcon.textContent = state.currentModalUserLiked ? '❤️' : '🤍';
  }
}

// ---------------- Star Rating Picker & Feedback Form ----------------

function initStarPicker() {
  elements.feedbackStarPicker.innerHTML = '';
  for (let i = 1; i <= 10; i++) {
    const starBtn = document.createElement('button');
    starBtn.type = 'button';
    starBtn.className = 'star-btn';
    starBtn.dataset.star = i;
    starBtn.textContent = '★';
    starBtn.title = `${i} of 10 Stars`;

    starBtn.addEventListener('click', () => setStarRating(i));
    starBtn.addEventListener('mouseenter', () => previewStarRating(i));
    starBtn.addEventListener('mouseleave', () => setStarRating(state.selectedRating));

    elements.feedbackStarPicker.appendChild(starBtn);
  }
  setStarRating(10);
}

function setStarRating(rating) {
  state.selectedRating = rating;
  elements.ratingValueBadge.textContent = `⭐ ${rating} / 10`;
  const stars = elements.feedbackStarPicker.querySelectorAll('.star-btn');
  stars.forEach(s => {
    const val = Number(s.dataset.star);
    s.classList.toggle('active', val <= rating);
  });
}

function previewStarRating(rating) {
  const stars = elements.feedbackStarPicker.querySelectorAll('.star-btn');
  stars.forEach(s => {
    const val = Number(s.dataset.star);
    s.classList.toggle('active', val <= rating);
  });
}

async function handleFeedbackSubmit(e) {
  e.preventDefault();

  if (!state.currentUser) {
    showNotification('Please log in to post a review!', 'info');
    return;
  }

  if (!state.currentModalMovieId) return;

  const comment = elements.feedbackCommentInput.value.trim();
  if (!comment) return;

  elements.btnSubmitFeedback.disabled = true;
  elements.btnSubmitFeedback.textContent = 'Posting...';

  try {
    const res = await fetch(`${API_BASE}/movie/${state.currentModalMovieId}/feedback`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({
        rating: state.selectedRating,
        comment: comment,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    elements.feedbackCommentInput.value = '';
    showNotification('Your review was posted to MongoDB Atlas! Thank you.', 'info');

    // Reload feedback list
    const fbRes = await fetch(`${API_BASE}/movie/${state.currentModalMovieId}/feedback`);
    if (fbRes.ok) {
      const fbData = await fbRes.json();
      renderFeedbackList(fbData.feedbacks || []);
    }
  } catch (err) {
    showNotification(`Could not post feedback: ${err.message}`, 'error');
  } finally {
    elements.btnSubmitFeedback.disabled = false;
    elements.btnSubmitFeedback.textContent = 'Post Review';
  }
}

function renderFeedbackList(feedbacks) {
  const count = feedbacks ? feedbacks.length : 0;
  elements.feedbackCountLabel.textContent = `${count} ${count === 1 ? 'review' : 'reviews'}`;

  if (!feedbacks || feedbacks.length === 0) {
    elements.modalFeedbackList.innerHTML = `
      <div style="color:var(--text-muted); font-size:0.86rem; padding:0.6rem 0; text-align:center;">
        No reviews yet. Be the first to share your rating and thoughts on this movie!
      </div>
    `;
    return;
  }

  elements.modalFeedbackList.innerHTML = '';
  feedbacks.forEach(fb => {
    const item = document.createElement('div');
    item.className = 'feedback-item';

    const avatarUrl = fb.user_picture || `https://api.dicebear.com/7.x/bottts/svg?seed=${encodeURIComponent(fb.user_name || 'user')}`;
    const dateFormatted = formatRelativeTime(fb.created_at);

    item.innerHTML = `
      <div class="feedback-item-header">
        <div class="feedback-author">
          <img class="feedback-avatar" src="${avatarUrl}" alt="${escapeHtml(fb.user_name)}" onerror="this.src='https://api.dicebear.com/7.x/bottts/svg?seed=cineml';" />
          <span class="feedback-name">${escapeHtml(fb.user_name)}</span>
        </div>
        <div class="feedback-meta">
          <span class="feedback-rating-pill">⭐ ${fb.rating} / 10</span>
          <span class="feedback-time">${dateFormatted}</span>
        </div>
      </div>
      <p class="feedback-comment">${escapeHtml(fb.comment)}</p>
    `;

    elements.modalFeedbackList.appendChild(item);
  });
}

function formatRelativeTime(dateString) {
  if (!dateString) return 'recently';
  try {
    const date = new Date(dateString);
    const now = new Date();
    const diffSec = Math.floor((now - date) / 1000);

    if (diffSec < 60) return 'just now';
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHrs = Math.floor(diffMin / 60);
    if (diffHrs < 24) return `${diffHrs}h ago`;
    const diffDays = Math.floor(diffHrs / 24);
    if (diffDays < 30) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  } catch (e) {
    return 'recently';
  }
}

// ---------------- View Switching (Recommendations vs Watchlist) ----------------

function switchToRecommendationsView() {
  state.currentView = 'recommendations';
  elements.tabRecommendations.classList.add('active');
  elements.tabWatchlist.classList.remove('active');
  elements.navWatchlistBtn.classList.remove('active');

  if (state.lastRecommendationMovies.length > 0) {
    const genres = Array.from(state.selectedGenres);
    elements.resultsHeading.textContent = genres.length 
      ? `Movies in ${genres.slice(0, 3).join(', ')}${genres.length > 3 ? '...' : ''}`
      : 'Curated Movies';
    elements.resultsSubtitle.textContent = 'Loaded based on your preferences. Click any movie to see details, like, review, and discover similar recommendations.';
    elements.resultsBadge.hidden = false;
    elements.resultsCountText.textContent = `${state.lastRecommendationMovies.length} movies`;
    renderMoviesGrid(state.lastRecommendationMovies);
  } else {
    loadPreferenceRecommendations();
  }
}

async function switchToWatchlistView() {
  state.currentView = 'watchlist';
  elements.tabWatchlist.classList.add('active');
  elements.tabRecommendations.classList.remove('active');
  elements.navWatchlistBtn.classList.add('active');

  scrollToResults();

  if (!state.currentUser) {
    elements.resultsHeading.textContent = 'My Watchlist';
    elements.resultsSubtitle.textContent = 'Log in to view, save, and manage your personalized movie collection.';
    elements.resultsBadge.hidden = true;
    elements.recsGrid.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">🔒</div>
        <h3 class="empty-state-title">Sign In Required</h3>
        <p class="empty-state-desc">
          Please click the <strong>Log-In</strong> button in the navigation bar to access your personal Watchlist.
        </p>
      </div>
    `;
    return;
  }

  showLoading(true);
  hideNotification();

  try {
    const res = await fetch(`${API_BASE}/user/watchlist`, {
      headers: getAuthHeaders(),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const movies = data.movies || [];
    state.userWatchlist = new Set(movies.map(m => Number(m.movie_id)));
    updateWatchlistBadges();

    elements.resultsHeading.textContent = 'My Watchlist';
    elements.resultsSubtitle.textContent = 'Your saved collection of movies retrieved from MongoDB Atlas. Click any movie to see details, like, and review!';
    elements.resultsBadge.hidden = false;
    elements.resultsCountText.textContent = `${movies.length} saved`;

    if (movies.length === 0) {
      elements.recsGrid.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">🔖</div>
          <h3 class="empty-state-title">Your Watchlist is Empty</h3>
          <p class="empty-state-desc">
            You haven't added any movies yet! Browse recommendations above and click the bookmark icon 🔖 to save films here for later.
          </p>
        </div>
      `;
    } else {
      renderMoviesGrid(movies);
    }
  } catch (err) {
    showNotification(`Could not fetch watchlist: ${err.message}`, 'error');
  } finally {
    showLoading(false);
  }
}

// ---------------- Event Listeners ----------------

function attachEventListeners() {
  // Auth Controls
  if (elements.btnDevLogin) {
    elements.btnDevLogin.addEventListener('click', handleDevLogin);
  }
  elements.btnSignout.addEventListener('click', handleSignout);

  // Watchlist Navigation Tabs
  elements.navWatchlistBtn.addEventListener('click', switchToWatchlistView);
  elements.tabWatchlist.addEventListener('click', switchToWatchlistView);
  elements.tabRecommendations.addEventListener('click', switchToRecommendationsView);

  // Modal Action Buttons (Watchlist & Likes)
  elements.modalWatchlistBtn.addEventListener('click', () => {
    if (state.currentModalMovieId) {
      toggleWatchlist(state.currentModalMovieId);
    }
  });

  elements.modalLikeBtn.addEventListener('click', () => {
    if (state.currentModalMovieId) {
      toggleLike(state.currentModalMovieId);
    }
  });

  // Modal Community Feedback Form
  elements.movieFeedbackForm.addEventListener('submit', handleFeedbackSubmit);

  // Toggle genre chips
  elements.genreChipsContainer.addEventListener('click', (e) => {
    const chip = e.target.closest('.genre-chip');
    if (!chip) return;
    const genre = chip.dataset.genre;

    if (state.selectedGenres.has(genre)) {
      state.selectedGenres.delete(genre);
      chip.classList.remove('active');
    } else {
      state.selectedGenres.add(genre);
      chip.classList.add('active');
    }
  });

  // Discover button click - triggers search and scrolls to results
  elements.discoverBtn.addEventListener('click', () => {
    switchToRecommendationsView();
    loadPreferenceRecommendations(true);
  });

  // Search button click
  elements.getRecsBtn.addEventListener('click', () => {
    const query = elements.searchInput.value.trim();
    if (query) {
      switchToRecommendationsView();
      executeTitleSearch(query);
    }
  });

  // Search input typing
  elements.searchInput.addEventListener('input', handleSearchInput);
  elements.searchInput.addEventListener('keydown', handleSearchKeydown);

  // Clear search
  elements.clearSearchBtn.addEventListener('click', () => {
    elements.searchInput.value = '';
    elements.clearSearchBtn.hidden = true;
    hideSuggestions();
  });

  // Close suggestions when clicking outside
  document.addEventListener('click', (e) => {
    if (!e.target.closest('#search-section')) {
      hideSuggestions();
    }
  });

  // Click on movie in grid to open detail modal
  elements.recsGrid.addEventListener('click', (e) => {
    // If clicked on the overlay bookmark button, handled separately
    if (e.target.closest('.card-btn-watchlist')) return;

    const card = e.target.closest('.movie-card');
    if (!card || !card.dataset.movieId) return;
    openMovieDetails(Number(card.dataset.movieId));
  });

  // Modal close listeners
  elements.modalCloseBtn.addEventListener('click', closeMovieDetails);
  elements.modalOverlay.addEventListener('click', (e) => {
    if (e.target === elements.modalOverlay) {
      closeMovieDetails();
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && elements.modalOverlay.classList.contains('open')) {
      closeMovieDetails();
    }
  });

  // Notification close
  elements.notificationClose.addEventListener('click', hideNotification);
}

// ---------------- Preference-Based Recommendations ----------------

async function loadPreferenceRecommendations(shouldScroll = false) {
  showLoading(true);
  hideNotification();

  const genres = Array.from(state.selectedGenres);
  const minRating = parseFloat(elements.ratingSelect.value) || 0.0;
  const limit = parseInt(elements.limitSelect.value, 10) || 18;

  try {
    const res = await fetch(`${API_BASE}/recommend/preferences`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        genres: genres,
        min_rating: minRating,
        limit: limit,
      })
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    state.lastRecommendationMovies = data.movies || [];

    elements.resultsHeading.textContent = genres.length 
      ? `Movies in ${genres.slice(0, 3).join(', ')}${genres.length > 3 ? '...' : ''}`
      : 'Curated Movies';

    elements.resultsSubtitle.textContent = (data.source === 'mongodb_atlas')
      ? 'Loaded directly from MongoDB Atlas based on your preferences. Click any movie to see details, like, review, and discover similar recommendations.'
      : 'Loaded from CineML dataset. Click any movie to see details, like, review, and discover similar recommendations.';

    elements.resultsBadge.hidden = false;
    elements.resultsCountText.textContent = `${data.count} movies`;

    renderMoviesGrid(data.movies);

    if (shouldScroll) {
      scrollToResults();
    }
  } catch (err) {
    showNotification(`Could not fetch recommendations: ${err.message}`, 'error');
  } finally {
    showLoading(false);
  }
}

// ---------------- Render Movies Grid ----------------

function renderMoviesGrid(movies) {
  elements.recsGrid.innerHTML = '';

  if (!movies || movies.length === 0) {
    elements.recsGrid.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">🔍</div>
        <h3 class="empty-state-title">No Movies Found</h3>
        <p class="empty-state-desc">Try selecting different genres or lowering the minimum rating filter.</p>
      </div>
    `;
    return;
  }

  movies.forEach(movie => {
    const card = document.createElement('div');
    card.className = 'movie-card';
    card.dataset.movieId = movie.movie_id;

    const inWatchlist = state.userWatchlist.has(Number(movie.movie_id));
    const posterSrc = movie.poster_url || `https://via.placeholder.com/300x450/192233/f0f6fc?text=${encodeURIComponent(movie.title)}`;
    const ratingDisplay = movie.vote_average ? `⭐ ${Number(movie.vote_average).toFixed(1)}` : '⭐ N/A';
    const genresDisplay = (movie.genres || []).slice(0, 2).join(' • ');

    card.innerHTML = `
      <div class="movie-card-poster-wrapper">
        <button 
          type="button" 
          class="card-overlay-btn card-btn-watchlist ${inWatchlist ? 'in-watchlist' : ''}" 
          data-movie-id="${movie.movie_id}" 
          title="${inWatchlist ? 'Remove from Watchlist' : 'Add to Watchlist'}"
          aria-label="${inWatchlist ? 'Remove from Watchlist' : 'Add to Watchlist'}"
        >
          ${inWatchlist ? '🔖 Saved' : '🔖'}
        </button>
        <img class="movie-card-poster" src="${posterSrc}" alt="${escapeHtml(movie.title)}" loading="lazy" onerror="this.onerror=null; this.src='https://via.placeholder.com/300x450/141923/e50914?text=${encodeURIComponent(movie.title)}';" />
        <span class="movie-rating-badge">${ratingDisplay}</span>
      </div>
      <div class="movie-card-body">
        <h3 class="movie-card-title">${escapeHtml(movie.title)}</h3>
        <p class="movie-card-genres">${escapeHtml(genresDisplay)}</p>
      </div>
    `;

    // Watchlist quick toggle button on card overlay
    const watchlistBtn = card.querySelector('.card-btn-watchlist');
    if (watchlistBtn) {
      watchlistBtn.addEventListener('click', (e) => {
        toggleWatchlist(movie.movie_id, e);
      });
    }

    elements.recsGrid.appendChild(card);
  });
}

// ---------------- Open Movie Details & "You Might Also Like" ----------------

async function openMovieDetails(movieId) {
  try {
    const numId = Number(movieId);
    state.currentModalMovieId = numId;

    // Reset feedback form & ratings for new movie
    elements.feedbackCommentInput.value = '';
    setStarRating(10);

    // Show modal in loading state first
    elements.modalTitle.textContent = 'Loading details...';
    elements.modalOverview.textContent = 'Fetching movie document from MongoDB Atlas & computing nearest neighbor similarity...';
    elements.modalGenres.innerHTML = '';
    elements.modalRecsTrack.innerHTML = '<div style="color:var(--text-muted); padding:1rem;">Calculating similar movies...</div>';
    elements.modalFeedbackList.innerHTML = '<div style="color:var(--text-muted); padding:0.6rem 0; font-size:0.85rem;">Loading community reviews...</div>';
    elements.modalOverlay.classList.add('open');

    // Fetch movie details with authorization header so backend returns user_liked status
    const res = await fetch(`${API_BASE}/movie/${numId}?recs_limit=10`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`Movie details failed: HTTP ${res.status}`);

    const data = await res.json();
    const movie = data.movie || data;
    const recs = data.you_might_also_like || [];

    // Populate Modal Info
    elements.modalTitle.textContent = movie.title;
    elements.modalPoster.src = movie.poster_url || '';
    elements.modalPoster.alt = movie.title;
    elements.modalRating.textContent = movie.vote_average ? `⭐ ${Number(movie.vote_average).toFixed(1)}` : '⭐ 7.0';
    elements.modalYear.textContent = movie.release_date ? movie.release_date.split('-')[0] : 'Feature Film';
    elements.modalOverview.textContent = movie.overview || 'No synopsis provided for this title.';

    // Populate Genres
    elements.modalGenres.innerHTML = '';
    (movie.genres || []).forEach(g => {
      const tag = document.createElement('span');
      tag.className = 'modal-genre-tag';
      tag.textContent = g;
      elements.modalGenres.appendChild(tag);
    });

    // Populate Modal Watchlist & Like State
    state.currentModalLikes = data.total_likes || 0;
    state.currentModalUserLiked = !!data.user_liked;
    updateModalActionButtons();

    // Populate Feedback Reviews
    renderFeedbackList(data.feedbacks || []);

    // Populate "You Might Also Like" E-Commerce Bar
    elements.modalRecsTrack.innerHTML = '';
    if (recs.length === 0) {
      elements.modalRecsTrack.innerHTML = '<div style="color:var(--text-muted); padding:1rem;">No similar recommendations found for this title.</div>';
    } else {
      recs.forEach(rec => {
        const item = document.createElement('div');
        item.className = 'rec-item-card';
        item.dataset.movieId = rec.movie_id;

        const posterUrl = rec.poster_url || `https://via.placeholder.com/150x225/192233/f0f6fc?text=${encodeURIComponent(rec.title)}`;
        const recRating = rec.vote_average ? `⭐ ${Number(rec.vote_average).toFixed(1)}` : '';

        item.innerHTML = `
          <img class="rec-item-poster" src="${posterUrl}" alt="${escapeHtml(rec.title)}" loading="lazy" onerror="this.onerror=null; this.src='https://via.placeholder.com/150x225/141923/e50914?text=${encodeURIComponent(rec.title)}';" />
          <div class="rec-item-info">
            <h4 class="rec-item-title">${escapeHtml(rec.title)}</h4>
            <span class="rec-item-rating">${recRating}</span>
          </div>
        `;

        // Clicking a recommendation smoothly loads that recommendation in the modal!
        item.addEventListener('click', (e) => {
          e.stopPropagation();
          openMovieDetails(rec.movie_id);
        });

        elements.modalRecsTrack.appendChild(item);
      });
    }

  } catch (err) {
    showNotification(`Could not load movie: ${err.message}`, 'error');
  }
}

function closeMovieDetails() {
  elements.modalOverlay.classList.remove('open');
  state.currentModalMovieId = null;
}

// ---------------- Title Search Autocomplete ----------------

function handleSearchInput(e) {
  const query = e.target.value.trim();
  elements.clearSearchBtn.hidden = !query;
  elements.getRecsBtn.disabled = !query;

  if (state.debounceTimer) clearTimeout(state.debounceTimer);

  if (query.length < 2) {
    hideSuggestions();
    return;
  }

  state.debounceTimer = setTimeout(() => {
    fetchSuggestions(query);
  }, 250);
}

async function fetchSuggestions(query) {
  if (state.currentSearchAbortController) {
    state.currentSearchAbortController.abort();
  }
  state.currentSearchAbortController = new AbortController();

  try {
    const res = await fetch(`${API_BASE}/movies?q=${encodeURIComponent(query)}&limit=6`, {
      signal: state.currentSearchAbortController.signal,
    });
    if (!res.ok) return;
    const data = await res.json();
    renderSuggestions(data.results);
  } catch (err) {
    if (err.name !== 'AbortError') {
      console.warn('Search error:', err);
    }
  }
}

function renderSuggestions(movies) {
  if (!movies || movies.length === 0) {
    hideSuggestions();
    return;
  }

  elements.suggestionsDropdown.innerHTML = '';
  movies.forEach(m => {
    const item = document.createElement('div');
    item.className = 'suggestion-item';
    item.dataset.movieId = m.movie_id;
    item.textContent = m.title;

    item.addEventListener('click', () => {
      elements.searchInput.value = m.title;
      hideSuggestions();
      openMovieDetails(m.movie_id);
    });

    elements.suggestionsDropdown.appendChild(item);
  });

  elements.suggestionsDropdown.hidden = false;
}

function hideSuggestions() {
  elements.suggestionsDropdown.hidden = true;
  elements.suggestionsDropdown.innerHTML = '';
}

function handleSearchKeydown(e) {
  if (e.key === 'Escape') {
    hideSuggestions();
  } else if (e.key === 'Enter') {
    e.preventDefault();
    const query = elements.searchInput.value.trim();
    if (query) {
      switchToRecommendationsView();
      executeTitleSearch(query);
    }
  }
}

// ---------------- Execute Title Search ----------------

async function executeTitleSearch(query) {
  const trimmed = (query || '').trim();
  if (!trimmed) return;

  hideSuggestions();
  showLoading(true);
  hideNotification();

  try {
    const res = await fetch(`${API_BASE}/movies?q=${encodeURIComponent(trimmed)}&limit=18`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    state.lastRecommendationMovies = data.results || [];

    elements.resultsHeading.textContent = `Search Results for "${trimmed}"`;
    elements.resultsSubtitle.textContent = `Found ${data.count} movies matching "${trimmed}". Click any movie to view full details, like, review, and see similar recommendations.`;
    elements.resultsBadge.hidden = false;
    elements.resultsCountText.textContent = `${data.count} movies`;

    renderMoviesGrid(data.results);
    scrollToResults();
  } catch (err) {
    showNotification(`Could not search movies: ${err.message}`, 'error');
  } finally {
    showLoading(false);
  }
}

// ---------------- Scroll Helper ----------------

function scrollToResults() {
  const section = document.getElementById('recommendations-section');
  if (section) {
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// ---------------- Utilities ----------------

function showLoading(show) {
  elements.loadingSpinner.hidden = !show;
  if (show) {
    elements.initialEmptyState.hidden = true;
  }
}

function showNotification(msg, type = 'info') {
  elements.notificationMsg.textContent = msg;
  elements.notificationBanner.className = `notification-banner ${type}`;
  elements.notificationBanner.hidden = false;
}

function hideNotification() {
  elements.notificationBanner.hidden = true;
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
