import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _make_text(genres, overview):
    # Pandas NaN becomes float('nan'), so protect against non-strings
    if not isinstance(genres, str):
        genres = ""
    if not isinstance(overview, str):
        overview = ""
    return (genres.strip() + " " + overview.strip()).strip()


def build_user_profile(watched_df):
    df = watched_df.copy()

    df["text"] = df.apply(
        lambda r: _make_text(r.get("genres"), r.get("overview")),
        axis=1
    )

    # keep only watched items that have TMDb ids and usable text
    df = df[df["tmdb_id"].notna()]
    df = df[df["text"].str.len() > 0]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    item_matrix = vectorizer.fit_transform(df["text"])

    return {"vectorizer": vectorizer, "item_matrix": item_matrix, "watched_df": df}


def recommend_from_candidates(profile, candidates_df, already_watched_tmdb_ids, top_k=5):
    vec = profile["vectorizer"]
    watched_df = profile["watched_df"]
    watched_mat = profile["item_matrix"]

    # weights from watch_count (default 1)
    weights = watched_df["watch_count"].fillna(1).astype(float).values
    weights = weights / weights.sum()

    # user vector: weighted sum (convert to normal numpy array, 2D)
    user_vec = (watched_mat.multiply(weights.reshape(-1, 1))).sum(axis=0)
    user_vec = np.asarray(user_vec).reshape(1, -1)

    cand = candidates_df.copy()
    cand = cand[cand["tmdb_id"].notna()]
    cand["tmdb_id"] = cand["tmdb_id"].astype(int)

    # remove already watched
    cand = cand[~cand["tmdb_id"].isin(already_watched_tmdb_ids)]

    cand["text"] = cand.apply(
        lambda r: _make_text(r.get("genres"), r.get("overview")),
        axis=1
    )
    cand = cand[cand["text"].str.len() > 0]

    cand_mat = vec.transform(cand["text"])
    sims = cosine_similarity(user_vec, cand_mat).flatten()
    cand["score"] = sims

    out = cand.sort_values("score", ascending=False).head(top_k)
    return out[["tmdb_id", "tmdb_name", "genres", "score"]]
