import json
import os
import re
import time
from pathlib import Path
from urllib.parse import quote_plus

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize


DATA_DIR = Path("data/raw")
MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"
CACHE_DIR = Path("data/cache")
POSTER_CACHE_PATH = CACHE_DIR / "tmdb_posters.json"


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def info_heading(text: str, help_text: str, level: str = "###") -> None:
    st.markdown(
        f'{level} <span title="{_esc(help_text)}">{_esc(text)} ⓘ</span>',
        unsafe_allow_html=True,
    )


def metric_card(container, label: str, value: str, help_text: str, delta: str | None = None) -> None:
    delta_html = (
        f'<div style="font-size:0.78rem;color:#6b7280;margin-top:2px;">Δ { _esc(delta) }</div>'
        if delta
        else ""
    )
    container.markdown(
        f"""
<div title="{_esc(help_text)}" style="border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;min-height:84px;">
  <div style="font-size:0.8rem;color:#6b7280;">{_esc(label)} ⓘ</div>
  <div style="font-size:1.35rem;font-weight:700;line-height:1.2;">{_esc(value)}</div>
  {delta_html}
</div>
""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not MOVIES_PATH.exists() or not RATINGS_PATH.exists():
        raise FileNotFoundError(
            "Missing dataset files. Expected data/raw/movies.csv and data/raw/ratings.csv."
        )

    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    year_pattern = re.compile(r"\((\d{4})\)\s*$")
    movies["year"] = (
        movies["title"]
        .astype(str)
        .apply(lambda x: int(year_pattern.search(x).group(1)) if year_pattern.search(x) else np.nan)
    )
    movies["genres"] = movies["genres"].fillna("Unknown")
    movies["genre_list"] = movies["genres"].str.split("|")

    agg = (
        ratings.groupby("movieId")["rating"]
        .agg(avg_rating="mean", rating_count="count")
        .reset_index()
    )

    merged = movies.merge(agg, left_on="movieId", right_on="movieId", how="left")
    merged["avg_rating"] = merged["avg_rating"].fillna(0.0)
    merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)

    c = merged["avg_rating"].mean()
    m = merged["rating_count"].quantile(0.75)
    merged["weighted_score"] = (
        (merged["rating_count"] / (merged["rating_count"] + m)) * merged["avg_rating"]
        + (m / (m + merged["rating_count"])) * c
    )
    merged["search_text"] = (
        merged["title"].astype(str).str.lower() + " " + merged["genres"].astype(str).str.lower()
    )

    return merged, ratings, movies


@st.cache_resource(show_spinner=False)
def build_similarity_index(movies_df: pd.DataFrame):
    corpus = (
        movies_df["title"].astype(str).str.replace(r"\(\d{4}\)$", "", regex=True)
        + " "
        + movies_df["genres"].astype(str).str.replace("|", " ", regex=False)
    )
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    matrix = vectorizer.fit_transform(corpus)
    cosine_sim = linear_kernel(matrix, matrix)

    n_components = 128 if matrix.shape[1] > 128 else max(16, matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent = normalize(svd.fit_transform(matrix))

    movie_id_to_idx = pd.Series(movies_df.index, index=movies_df["movieId"]).to_dict()
    idx_to_movie_id = pd.Series(movies_df["movieId"].values, index=movies_df.index).to_dict()
    return cosine_sim, latent, movie_id_to_idx, idx_to_movie_id


def init_feedback_state() -> None:
    if "liked_movies" not in st.session_state:
        st.session_state.liked_movies = set()
    if "disliked_movies" not in st.session_state:
        st.session_state.disliked_movies = set()


def update_feedback(movie_id: int, action: str) -> None:
    liked = st.session_state.liked_movies
    disliked = st.session_state.disliked_movies
    if action == "like":
        liked.add(movie_id)
        disliked.discard(movie_id)
    elif action == "dislike":
        disliked.add(movie_id)
        liked.discard(movie_id)
    elif action == "clear":
        liked.discard(movie_id)
        disliked.discard(movie_id)


def apply_filters(
    df: pd.DataFrame,
    selected_genres: list[str],
    year_min: int,
    year_max: int,
    min_avg_rating: float,
    min_votes: int,
) -> pd.DataFrame:
    out = df.copy()
    if selected_genres:
        out = out[out["genre_list"].apply(lambda g: any(x in g for x in selected_genres))]

    out = out[
        (out["year"].fillna(year_min).astype(float) >= year_min)
        & (out["year"].fillna(year_max).astype(float) <= year_max)
        & (out["avg_rating"] >= min_avg_rating)
        & (out["rating_count"] >= min_votes)
    ]
    return out


def attach_rank_score(df: pd.DataFrame, strategy: str, wr: float, ar: float, vr: float) -> pd.DataFrame:
    out = df.copy()
    vote_norm = (out["rating_count"] - out["rating_count"].min()) / (
        (out["rating_count"].max() - out["rating_count"].min()) + 1e-9
    )
    if strategy == "Weighted score (balanced)":
        out["rank_score"] = out["weighted_score"]
    elif strategy == "Highest average rating":
        out["rank_score"] = out["avg_rating"]
    elif strategy == "Most popular (votes)":
        out["rank_score"] = out["rating_count"]
    else:
        out["rank_score"] = wr * out["weighted_score"] + ar * out["avg_rating"] + vr * vote_norm
    return out


def apply_feedback_boost(df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["feedback_boost"] = 0.0

    liked = st.session_state.liked_movies
    disliked = st.session_state.disliked_movies
    if not liked and not disliked:
        out["personal_score"] = out["rank_score"]
        return out

    out.loc[out["movieId"].isin(liked), "feedback_boost"] += 0.6
    out.loc[out["movieId"].isin(disliked), "feedback_boost"] -= 0.8

    liked_genres, disliked_genres = set(), set()
    if liked:
        for gl in source_df[source_df["movieId"].isin(liked)]["genre_list"]:
            liked_genres.update(gl if isinstance(gl, list) else [])
    if disliked:
        for gl in source_df[source_df["movieId"].isin(disliked)]["genre_list"]:
            disliked_genres.update(gl if isinstance(gl, list) else [])

    if liked_genres:
        out.loc[out["genre_list"].apply(lambda g: any(x in g for x in liked_genres)), "feedback_boost"] += 0.12
    if disliked_genres:
        out.loc[out["genre_list"].apply(lambda g: any(x in g for x in disliked_genres)), "feedback_boost"] -= 0.15

    out["personal_score"] = out["rank_score"] + out["feedback_boost"]
    return out


def _normalize_title_for_tmdb(title: str) -> str:
    # Remove trailing year and common separators for better TMDB match quality.
    base = re.sub(r"\s*\(\d{4}\)\s*$", "", str(title))
    return base.replace("`", "").strip()


def _placeholder_poster(title: str) -> str:
    return f"https://placehold.co/300x450/111827/E5E7EB?text={quote_plus(str(title)[:40])}"


def _load_poster_cache_from_disk() -> dict[str, str]:
    if POSTER_CACHE_PATH.exists():
        try:
            return json.loads(POSTER_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_poster_cache_to_disk(cache: dict[str, str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    POSTER_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def init_poster_cache_state() -> None:
    if "poster_cache" not in st.session_state:
        st.session_state.poster_cache = _load_poster_cache_from_disk()


def _tmdb_search_poster(api_key: str, title: str, year: int | None = None) -> str | None:
    endpoint = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": api_key,
        "query": _normalize_title_for_tmdb(title),
        "include_adult": "false",
        "language": "en-US",
        "page": 1,
    }
    if year and int(year) > 1800:
        params["year"] = int(year)

    resp = requests.get(endpoint, params=params, timeout=8)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    for movie in results:
        poster_path = movie.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w342{poster_path}"
    return None


def poster_url(movie_id: int, title: str, year: int | None = None) -> str:
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key:
        return _placeholder_poster(title)

    cache_key = str(movie_id)
    cache = st.session_state.poster_cache
    if cache_key in cache:
        return cache[cache_key]

    try:
        url = _tmdb_search_poster(api_key, title, year)
        if not url:
            # Retry without year constraint for noisy titles.
            time.sleep(0.05)
            url = _tmdb_search_poster(api_key, title, None)
        cache[cache_key] = url if url else _placeholder_poster(title)
        _save_poster_cache_to_disk(cache)
        return cache[cache_key]
    except Exception:
        cache[cache_key] = _placeholder_poster(title)
        _save_poster_cache_to_disk(cache)
        return cache[cache_key]


def explanation_text(movie: pd.Series, seed_title: str | None = None) -> str:
    if seed_title:
        return f"AI match: similar style and themes to **{seed_title}**."
    avg = float(movie.get("avg_rating", 0.0))
    votes = int(movie.get("rating_count", 0))
    if avg >= 4.2 and votes >= 1000:
        return "AI signal: highly rated and consistently liked by many viewers."
    if votes >= 2000:
        return "AI signal: popular title with strong audience engagement."
    return "AI signal: strong content/genre match under current ranking settings."


def render_movie_cards(df: pd.DataFrame, card_count: int, section_key: str, seed_title: str | None = None):
    rows = df.head(card_count)
    if rows.empty:
        st.info("No movies found for current settings.")
        return

    cols_per_row = 5
    for start in range(0, len(rows), cols_per_row):
        row_slice = rows.iloc[start : start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (_, item) in zip(cols, row_slice.iterrows()):
            with col:
                st.image(
                    poster_url(
                        movie_id=int(item["movieId"]),
                        title=item["title"],
                        year=int(item["year"]) if pd.notna(item["year"]) else None,
                    ),
                    use_column_width=True,
                )
                st.markdown(f"**{item['title']}**")
                st.caption(
                    f"⭐ {item['avg_rating']:.2f} | 🗳️ {int(item['rating_count'])} | 📅 {int(item['year']) if pd.notna(item['year']) else 'N/A'}"
                )
                st.caption(item["genres"])
                st.caption(explanation_text(item, seed_title))

                movie_id = int(item["movieId"])
                c1, c2, c3 = st.columns(3)
                if c1.button("👍", key=f"{section_key}_like_{movie_id}"):
                    update_feedback(movie_id, "like")
                if c2.button("👎", key=f"{section_key}_dislike_{movie_id}"):
                    update_feedback(movie_id, "dislike")
                if c3.button("↺", key=f"{section_key}_clear_{movie_id}"):
                    update_feedback(movie_id, "clear")


@st.cache_data(show_spinner=True)
def evaluate_models(
    ratings: pd.DataFrame,
    merged: pd.DataFrame,
    cosine_sim: np.ndarray,
    latent_vectors: np.ndarray,
    movie_id_to_idx: dict[int, int],
    idx_to_movie_id: dict[int, int],
    k: int = 10,
    max_users: int = 250,
) -> dict:
    valid_users = ratings.groupby("userId").size()
    valid_users = valid_users[valid_users >= 8].index.tolist()
    if not valid_users:
        return {"error": "Not enough user history to evaluate."}

    users = valid_users[:max_users]
    popularity_rank = merged.sort_values("weighted_score", ascending=False)["movieId"].tolist()

    eval_count = 0
    coverage_pop, coverage_lexical, coverage_semantic, coverage_hybrid = set(), set(), set(), set()
    ks = [5, 10, 15, 20]
    per_k = {
        "pop": {x: {"hits": 0, "rr_sum": 0.0, "ndcg_sum": 0.0} for x in ks},
        "lexical": {x: {"hits": 0, "rr_sum": 0.0, "ndcg_sum": 0.0} for x in ks},
        "semantic": {x: {"hits": 0, "rr_sum": 0.0, "ndcg_sum": 0.0} for x in ks},
        "hybrid": {x: {"hits": 0, "rr_sum": 0.0, "ndcg_sum": 0.0} for x in ks},
    }
    max_k = max(max(ks), k)

    for user_id in users:
        hist = ratings[ratings["userId"] == user_id].sort_values("timestamp")
        if len(hist) < 8:
            continue
        holdout = hist.iloc[-1]
        train_hist = hist.iloc[:-1]
        seen = set(train_hist["movieId"].tolist())
        true_movie = int(holdout["movieId"])

        pop_recs = [m for m in popularity_rank if m not in seen][:max_k]
        coverage_pop.update(pop_recs[:k])
        for kval in ks:
            preds = pop_recs[:kval]
            if true_movie in preds:
                rank = preds.index(true_movie) + 1
                per_k["pop"][kval]["hits"] += 1
                per_k["pop"][kval]["rr_sum"] += 1.0 / rank
                per_k["pop"][kval]["ndcg_sum"] += 1.0 / np.log2(rank + 1)

        liked = train_hist[train_hist["rating"] >= 4.0]
        seed_movie = int((liked.iloc[-1] if not liked.empty else train_hist.iloc[-1])["movieId"])
        seed_idx = movie_id_to_idx.get(seed_movie)
        if seed_idx is not None:
            sim_scores = list(enumerate(cosine_sim[seed_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:300]
            content_recs = []
            for idx, _ in sim_scores:
                mid = idx_to_movie_id.get(idx)
                if mid is None or mid in seen:
                    continue
                content_recs.append(mid)
                if len(content_recs) >= max_k:
                    break
            coverage_lexical.update(content_recs[:k])
            for kval in ks:
                preds = content_recs[:kval]
                if true_movie in preds:
                    rank = preds.index(true_movie) + 1
                    per_k["lexical"][kval]["hits"] += 1
                    per_k["lexical"][kval]["rr_sum"] += 1.0 / rank
                    per_k["lexical"][kval]["ndcg_sum"] += 1.0 / np.log2(rank + 1)

            # After: semantic embedding similarity baseline
            sem_scores = sorted(list(enumerate(latent_vectors @ latent_vectors[seed_idx])), key=lambda x: float(x[1]), reverse=True)[1:300]
            sem_recs = []
            for idx, _ in sem_scores:
                mid = idx_to_movie_id.get(idx)
                if mid is None or mid in seen:
                    continue
                sem_recs.append(mid)
                if len(sem_recs) >= max_k:
                    break
            coverage_semantic.update(sem_recs[:k])
            for kval in ks:
                preds = sem_recs[:kval]
                if true_movie in preds:
                    rank = preds.index(true_movie) + 1
                    per_k["semantic"][kval]["hits"] += 1
                    per_k["semantic"][kval]["rr_sum"] += 1.0 / rank
                    per_k["semantic"][kval]["ndcg_sum"] += 1.0 / np.log2(rank + 1)

            # After: hybrid (semantic + popularity) baseline
            pop_rank_map = {mid: i for i, mid in enumerate(popularity_rank)}
            sem_rank_map = {mid: i for i, mid in enumerate(sem_recs)}
            hybrid_candidates = []
            for mid in set(sem_recs + pop_recs):
                sem_r = sem_rank_map.get(mid, 10_000)
                pop_r = pop_rank_map.get(mid, 10_000)
                # Lower is better: weighted rank fusion
                score = 0.45 * sem_r + 0.55 * pop_r
                hybrid_candidates.append((mid, score))
            hybrid_candidates.sort(key=lambda x: x[1])
            hybrid_recs = [mid for mid, _ in hybrid_candidates[:max_k]]

            coverage_hybrid.update(hybrid_recs[:k])
            for kval in ks:
                preds = hybrid_recs[:kval]
                if true_movie in preds:
                    rank = preds.index(true_movie) + 1
                    per_k["hybrid"][kval]["hits"] += 1
                    per_k["hybrid"][kval]["rr_sum"] += 1.0 / rank
                    per_k["hybrid"][kval]["ndcg_sum"] += 1.0 / np.log2(rank + 1)
        eval_count += 1

    if eval_count == 0:
        return {"error": "No users available for evaluation split."}

    curve_rows = []
    for key, name in [
        ("pop", "Before: Popularity"),
        ("lexical", "Before: Lexical"),
        ("semantic", "After: Semantic AI"),
        ("hybrid", "After: Hybrid AI"),
    ]:
        for kval in ks:
            s = per_k[key][kval]
            curve_rows.append(
                {
                    "Model": name,
                    "K": kval,
                    "HitRate@K": s["hits"] / eval_count,
                    "MRR@K": s["rr_sum"] / eval_count,
                    "NDCG@K": s["ndcg_sum"] / eval_count,
                }
            )
    curve_df = pd.DataFrame(curve_rows)
    selected_k = min(ks, key=lambda x: abs(x - k))
    summary_df = curve_df[curve_df["K"] == selected_k].copy()
    catalog_size = merged["movieId"].nunique()
    summary_df["CatalogCoverage"] = [
        len(coverage_pop) / max(catalog_size, 1),
        len(coverage_lexical) / max(catalog_size, 1),
        len(coverage_semantic) / max(catalog_size, 1),
        len(coverage_hybrid) / max(catalog_size, 1),
    ]
    lift = {}
    after = summary_df[summary_df["Model"] == "After: Hybrid AI"].head(1)
    before_set = summary_df[summary_df["Model"].isin(["Before: Popularity", "Before: Lexical"])]
    if not after.empty and not before_set.empty:
        for col in ["HitRate@K", "MRR@K", "NDCG@K", "CatalogCoverage"]:
            b = float(before_set[col].max())  # best non-AI baseline
            a = float(after.iloc[0][col])
            lift[col] = ((a - b) / b * 100.0) if b > 0 else np.nan

    return {
        "users_evaluated": eval_count,
        "selected_k": selected_k,
        "summary_df": summary_df,
        "curve_df": curve_df,
        "lift_percent": lift,
    }


def format_result_table(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    score_col = "personal_score" if "personal_score" in df.columns else "rank_score"
    out = df[["title", "year", "genres", "avg_rating", "rating_count", score_col]].head(top_n).copy()
    out.columns = ["Title", "Year", "Genres", "Avg Rating", "Votes", "Score"]
    out["Avg Rating"] = out["Avg Rating"].round(2)
    out["Score"] = out["Score"].round(3)
    return out


def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
    st.title("🎬 Movie Recommender")
    st.caption("AI-driven discovery with semantic similarity, personalized feedback, and explainable recommendations.")

    try:
        merged, ratings, movies_only = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    init_feedback_state()
    init_poster_cache_state()
    movies_indexed = movies_only.reset_index(drop=True)
    cosine_sim, latent_vectors, movie_id_to_idx, idx_to_movie_id = build_similarity_index(movies_indexed)

    all_genres = sorted({g for gl in merged["genre_list"] for g in gl if isinstance(g, str) and g.strip()})
    years = merged["year"].dropna().astype(int)
    min_year = int(years.min()) if len(years) else 1900
    max_year = int(years.max()) if len(years) else 2100

    st.sidebar.header("Filters")
    selected_genres = st.sidebar.multiselect("Genres", options=all_genres)
    year_min, year_max = st.sidebar.slider("Release Year", min_year, max_year, (min_year, max_year))
    min_avg = st.sidebar.slider("Minimum Average Rating", 0.0, 5.0, 3.5, 0.1)
    min_votes = st.sidebar.slider("Minimum Votes", 0, 5000, 200, 50)

    st.sidebar.header("Ranking Controls")
    ranking_strategy = st.sidebar.selectbox(
        "Sort results by",
        ["Weighted score (balanced)", "Highest average rating", "Most popular (votes)", "Custom blend"],
    )
    w_weighted = st.sidebar.slider("Weighted score weight", 0.0, 1.0, 0.5, 0.05)
    w_avg = st.sidebar.slider("Average rating weight", 0.0, 1.0, 0.35, 0.05)
    w_votes = st.sidebar.slider("Popularity weight", 0.0, 1.0, 0.15, 0.05)

    st.sidebar.header("Personalization")
    st.sidebar.caption("Use 👍 / 👎 on cards to personalize in-session recommendations.")
    st.sidebar.write(f"Liked: {len(st.session_state.liked_movies)} | Disliked: {len(st.session_state.disliked_movies)}")
    if st.sidebar.button("Reset feedback"):
        st.session_state.liked_movies = set()
        st.session_state.disliked_movies = set()
        st.rerun()

    st.sidebar.header("Posters")
    if os.getenv("TMDB_API_KEY", "").strip():
        st.sidebar.success("TMDB posters enabled")
    else:
        st.sidebar.info("Set TMDB_API_KEY to load real posters")
    if st.sidebar.button("Clear poster cache"):
        st.session_state.poster_cache = {}
        _save_poster_cache_to_disk({})
        st.rerun()

    filtered = apply_filters(merged, selected_genres, year_min, year_max, min_avg, min_votes)
    filtered = attach_rank_score(filtered, ranking_strategy, w_weighted, w_avg, w_votes)
    filtered = apply_feedback_boost(filtered, merged)

    tab_discover, tab_search, tab_similar, tab_eval = st.tabs(
        ["Top Picks", "Search Movies", "Find Similar Movies", "Evaluation Dashboard"]
    )

    with tab_discover:
        st.subheader("Top Rated Movies")
        top_rated = filtered.sort_values(["personal_score", "avg_rating"], ascending=False)
        render_movie_cards(top_rated, card_count=10, section_key="top")

        st.subheader("Critically Acclaimed")
        acclaimed = filtered[(filtered["avg_rating"] >= 4.2) & (filtered["rating_count"] >= 800)]
        acclaimed = acclaimed.sort_values(["personal_score", "avg_rating"], ascending=False)
        if acclaimed.empty:
            st.info("No critically acclaimed titles match current filters.")
        else:
            render_movie_cards(acclaimed, card_count=10, section_key="acclaimed")

    with tab_search:
        st.subheader("Search by title, genre, or keyword")
        query = st.text_input("Search", placeholder="e.g. toy story, action, sci-fi, comedy")
        n_results = st.slider("Results to show", 5, 100, 25, 5)
        searched = filtered[filtered["search_text"].str.contains(query.lower().strip(), na=False)] if query.strip() else filtered
        if searched.empty:
            st.warning("No movies found for that search + filter combination.")
        else:
            searched = searched.sort_values(["personal_score", "avg_rating", "rating_count"], ascending=False)
            render_movie_cards(searched, card_count=min(10, n_results), section_key="search")

    with tab_similar:
        st.subheader("Get recommendations similar to a movie")
        movie_options = filtered.sort_values("title")["title"].tolist()
        if not movie_options:
            st.warning("No movies available with current filters.")
        else:
            selected_title = st.selectbox("Select a movie", options=movie_options)
            k = st.slider("How many recommendations?", 5, 30, 12, 1)
            movie_row = movies_only[movies_only["title"] == selected_title].head(1)
            if movie_row.empty:
                st.info("Selected movie not found in source catalog.")
            else:
                movie_id = int(movie_row.iloc[0]["movieId"])
                idx = movie_id_to_idx.get(movie_id)
                if idx is None:
                    st.info("No similarity index found for selected movie.")
                else:
                    sims = latent_vectors @ latent_vectors[idx]
                    sim_scores = sorted(list(enumerate(sims)), key=lambda x: float(x[1]), reverse=True)[1:250]
                    candidate_idx = [i for i, _ in sim_scores]
                    candidates = movies_indexed.iloc[candidate_idx][["movieId", "title", "genres"]].copy()
                    candidates = candidates.merge(
                        merged[["movieId", "year", "avg_rating", "rating_count", "weighted_score", "genre_list"]],
                        on="movieId",
                        how="left",
                    )
                    candidates = apply_filters(candidates, selected_genres, year_min, year_max, min_avg, min_votes)
                    candidates = attach_rank_score(candidates, ranking_strategy, w_weighted, w_avg, w_votes)
                    candidates = apply_feedback_boost(candidates, merged)
                    candidates = candidates.sort_values(["personal_score", "avg_rating", "rating_count"], ascending=False)

                    if candidates.empty:
                        st.warning("No similar movies found with current filters.")
                    else:
                        render_movie_cards(candidates, card_count=min(10, k), section_key="similar", seed_title=selected_title)

    with tab_eval:
        info_heading("Eval Dashboard", "Business-friendly checks for recommendation quality.")
        st.caption("Simple, non-technical quality checks using held-out user behavior.")
        k_eval = st.slider("Top-K size", 5, 25, 10, 1, help="How many recommendations are shown per user.")
        users_eval = st.slider("Sample users", 50, 1000, 250, 50, help="How many users to include.")

        with st.spinner("Computing evaluation metrics..."):
            eval_result = evaluate_models(
                ratings=ratings,
                merged=merged,
                cosine_sim=cosine_sim,
                latent_vectors=latent_vectors,
                movie_id_to_idx=movie_id_to_idx,
                idx_to_movie_id=idx_to_movie_id,
                k=k_eval,
                max_users=users_eval,
            )

        if "error" in eval_result:
            st.error(eval_result["error"])
        else:
            summary_df = eval_result["summary_df"]
            curve_df = eval_result["curve_df"]
            selected_k = eval_result["selected_k"]

            info_heading(f"Snapshot (Top {selected_k})", "Quick quality comparison at selected list size.")
            cards = st.columns(4)
            metric_card(cards[0], "Users", f"{eval_result['users_evaluated']}", "Number of users used in evaluation.")
            metric_card(
                cards[1],
                "Hit",
                f"{summary_df['HitRate@K'].max():.3f}",
                "How often users get at least one good match in the list.",
                delta=f"{(summary_df['HitRate@K'].max()-summary_df['HitRate@K'].min()):.3f}",
            )
            metric_card(
                cards[2],
                "Rank",
                f"{summary_df['MRR@K'].max():.3f}",
                "How high the best match appears in the list (higher is better).",
            )
            metric_card(
                cards[3],
                "Variety",
                f"{summary_df['CatalogCoverage'].max():.3f}",
                "How broad the recommendations are across all movies.",
            )

            lift = eval_result.get("lift_percent", {})
            if lift:
                lift_cols = st.columns(4)
                metric_card(
                    lift_cols[0],
                    "Lift: Hit",
                    f"{lift.get('HitRate@K', np.nan):.1f}%",
                    "Percent lift of Hybrid AI over the best non-AI baseline on Hit.",
                )
                metric_card(
                    lift_cols[1],
                    "Lift: Rank",
                    f"{lift.get('MRR@K', np.nan):.1f}%",
                    "Percent lift of Hybrid AI over the best non-AI baseline on Rank.",
                )
                metric_card(
                    lift_cols[2],
                    "Lift: Quality",
                    f"{lift.get('NDCG@K', np.nan):.1f}%",
                    "Percent lift of Hybrid AI over the best non-AI baseline on Quality.",
                )
                metric_card(
                    lift_cols[3],
                    "Lift: Variety",
                    f"{lift.get('CatalogCoverage', np.nan):.1f}%",
                    "Percent lift of Hybrid AI over the best non-AI baseline on Variety.",
                )

            st.caption("Hover labels marked with ⓘ for quick definitions.")

            info_heading("Performance by List Size", "How recommendation quality changes as list size grows.")
            metric_map = {"Hit": "HitRate@K", "Rank": "MRR@K", "Quality": "NDCG@K"}
            metric_choice_label = st.selectbox(
                "Metric",
                list(metric_map.keys()),
                help="Hit: at least one good match. Rank: good match appears earlier. Quality: overall ranking quality.",
            )
            metric_choice = metric_map[metric_choice_label]
            curve_chart = (
                alt.Chart(curve_df)
                .mark_line(point=True, strokeWidth=3)
                .encode(
                    x=alt.X("K:Q", title="K"),
                    y=alt.Y(f"{metric_choice}:Q", title=metric_choice_label),
                    color=alt.Color("Model:N"),
                    tooltip=["Model", "K", "HitRate@K", "MRR@K", "NDCG@K"],
                )
                .properties(height=320)
            )
            st.altair_chart(curve_chart, use_container_width=True)

            info_heading("Variety Comparison", "Compares how many different movies each approach surfaces.")
            coverage_chart = (
                alt.Chart(summary_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Model:N"),
                    y=alt.Y("CatalogCoverage:Q", title="Variety"),
                    color=alt.Color("Model:N", legend=None),
                    tooltip=["Model", "CatalogCoverage"],
                )
                .properties(height=280)
            )
            st.altair_chart(coverage_chart, use_container_width=True)


if __name__ == "__main__":
    main()
