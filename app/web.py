import os
import sys
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on the Python path so `import src...` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tmdb_client import TMDbClient
from src.ingest_netflix import load_netflix_history, summarize_history
from src.enrich_watched import enrich_top_watched
from src.build_candidates import build_candidate_pool
from src.recommender import build_user_profile, recommend_from_candidates

load_dotenv()

st.set_page_config(page_title="NextWatch Recommender", layout="wide")
st.title("🎬 NextWatch Recommender")
st.write("Upload a Netflix CSV **or** type shows manually, then get 5 recommendations.")


def get_client() -> TMDbClient:
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        st.error("TMDB_API_KEY not found. Create a .env file with TMDB_API_KEY=YOUR_KEY")
        st.stop()
    return TMDbClient(api_key=api_key)


client = get_client()

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")
seed_n = st.sidebar.slider("Seed shows (how many watched titles to use)", 3, 20, 10)
per_seed = st.sidebar.slider("Similar shows per seed", 10, 50, 20)

# Only used for CSV mode:
top_n_csv = st.sidebar.slider("CSV: how many top watched to use", 5, 30, 20)

# ---------- Input mode ----------
mode = st.radio(
    "How do you want to provide watched titles?",
    ["Type shows manually (no CSV required)", "Upload Netflix Viewing Activity CSV"],
)

watched_tmdb = None

# ---------- Mode A: Manual entry ----------
if mode == "Type shows manually (no CSV required)":
    st.subheader("Enter shows you’ve watched (one per line)")
    text = st.text_area(
        "Titles",
        height=220,
        placeholder="The Big Bang Theory\nSuits\nBreaking Bad\nThe Rookie",
    )

    titles = [line.strip() for line in text.splitlines() if line.strip()]
    st.caption(f"Titles entered: {len(titles)}")

    if st.button("Generate recommendations"):
        if len(titles) == 0:
            st.warning("Please enter at least 1 title.")
            st.stop()

        rows = []
        with st.spinner("Matching your titles to TMDb..."):
            for t in titles:
                hit = client.search_tv(t)
                if not hit:
                    rows.append(
                        {
                            "series_title": t,
                            "watch_count": 1,
                            "tmdb_id": None,
                            "tmdb_name": None,
                            "genres": None,
                            "overview": None,
                            "status": "NOT_FOUND",
                        }
                    )
                    continue

                tv_id = int(hit["id"])
                details = client.get_tv_details(tv_id)
                genres = [g.get("name") for g in details.get("genres", []) if g.get("name")]

                rows.append(
                    {
                        "series_title": t,
                        "watch_count": 1,
                        "tmdb_id": tv_id,
                        "tmdb_name": details.get("name"),
                        "genres": ", ".join(genres),
                        "overview": details.get("overview"),
                        "status": "OK",
                    }
                )

                time.sleep(0.15)

        watched_tmdb = pd.DataFrame(rows)

        st.subheader("Matched titles")
        st.dataframe(watched_tmdb[["series_title", "tmdb_name", "status"]], use_container_width=True)

# ---------- Mode B: CSV upload ----------
else:
    st.subheader("Upload Netflix Viewing Activity CSV")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        # Save to a temp file so we can reuse your existing pipeline
        os.makedirs("data/processed", exist_ok=True)
        tmp_path = "data/processed/_uploaded_netflix.csv"
        pd.read_csv(uploaded).to_csv(tmp_path, index=False)

        history = load_netflix_history(tmp_path)
        summary = summarize_history(history, top_n=top_n_csv)

        st.subheader("Top watched series (from your CSV)")
        st.dataframe(summary, use_container_width=True)

        if st.button("Generate recommendations"):
            with st.spinner("Enriching your top watched titles with TMDb metadata..."):
                watched_tmdb = enrich_top_watched(client, tmp_path, top_n=top_n_csv)

            st.subheader("Matched titles")
            st.dataframe(
                watched_tmdb[["series_title", "tmdb_name", "status", "watch_count"]].head(30),
                use_container_width=True,
            )
    else:
        st.info("CSV is optional. If you don’t have it, switch to manual entry above.")

# ---------- Common recommendation pipeline ----------
if watched_tmdb is not None:
    ok_watched = watched_tmdb[watched_tmdb["status"] == "OK"].copy()

    if ok_watched.empty:
        st.error("None of your titles matched on TMDb. Try different spellings.")
        st.stop()

    # Adjust seeds to available number of matched titles
    seed_n_eff = min(seed_n, len(ok_watched))

    with st.spinner("Building candidate pool from similar shows..."):
        candidates = build_candidate_pool(client, watched_tmdb, seed_n=seed_n_eff, per_seed=per_seed)

    already = set(ok_watched["tmdb_id"].dropna().astype(int).tolist())

    with st.spinner("Ranking candidates (TF-IDF + cosine similarity)..."):
        profile = build_user_profile(watched_tmdb)
        recs = recommend_from_candidates(profile, candidates, already_watched_tmdb_ids=already, top_k=5)

    st.subheader("✅ Your next 5 shows")
    st.dataframe(recs[["tmdb_name", "genres", "score"]], use_container_width=True)
