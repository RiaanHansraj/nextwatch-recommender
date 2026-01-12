import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.tmdb_client import TMDbClient
from src.enrich_watched import enrich_top_watched
from src.build_candidates import build_candidate_pool
from src.recommender import build_user_profile, recommend_from_candidates
from src.ingest_netflix import load_netflix_history, summarize_history

load_dotenv()

st.set_page_config(page_title="NextWatch Recommender", layout="wide")
st.title("🎬 NextWatch Recommender (Web App)")
st.write("Upload your Netflix viewing history CSV and get 5 next-watch recommendations.")

api_key = os.getenv("TMDB_API_KEY")
if not api_key:
    st.error("TMDB_API_KEY not found. Create a .env file with TMDB_API_KEY=YOUR_KEY")
    st.stop()

client = TMDbClient(api_key=api_key)

# Sidebar settings
st.sidebar.header("Settings")
top_n = st.sidebar.slider("Top watched to use", min_value=5, max_value=30, value=20)
seed_n = st.sidebar.slider("Seeds (top watched) to fetch similar shows from", min_value=5, max_value=20, value=10)
per_seed = st.sidebar.slider("Similar shows per seed", min_value=10, max_value=40, value=20)

uploaded = st.file_uploader("Upload Netflix Viewing Activity CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# Save uploaded CSV to a temp file so we can reuse existing functions
os.makedirs("data/processed", exist_ok=True)
tmp_path = "data/processed/_uploaded_netflix.csv"
pd.read_csv(uploaded).to_csv(tmp_path, index=False)

# Show top watched summary
history = load_netflix_history(tmp_path)
summary = summarize_history(history, top_n=top_n)

st.subheader("Top watched series (from your CSV)")
st.dataframe(summary, use_container_width=True)

if st.button("Generate recommendations"):
    with st.spinner("Matching your watched shows to TMDb..."):
        watched_tmdb = enrich_top_watched(client, tmp_path, top_n=top_n)

    ok_watched = watched_tmdb[watched_tmdb["status"] == "OK"]
    if ok_watched.empty:
        st.error("Could not match your watched shows to TMDb. Try lowering 'Top watched to use'.")
        st.stop()

    with st.spinner("Building candidate pool from similar shows..."):
        candidates = build_candidate_pool(client, watched_tmdb, seed_n=seed_n, per_seed=per_seed)

    already = set(ok_watched["tmdb_id"].dropna().astype(int).tolist())
    profile = build_user_profile(watched_tmdb)
    recs = recommend_from_candidates(profile, candidates, already_watched_tmdb_ids=already, top_k=5)

    st.subheader("✅ Your next 5 shows")
    st.dataframe(recs[["tmdb_name", "genres", "score"]], use_container_width=True)
