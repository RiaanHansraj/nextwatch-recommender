import argparse
import os

import pandas as pd
from dotenv import load_dotenv

from src.ingest_netflix import load_netflix_history, summarize_history
from src.tmdb_client import TMDbClient
from src.enrich_watched import enrich_top_watched
from src.build_candidates import build_candidate_pool
from src.recommender import build_user_profile, recommend_from_candidates


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="nextwatch",
        description="Recommend next shows based on Netflix viewing history (content-based).",
    )

    parser.add_argument("--history", type=str, help="Path to Netflix viewing history CSV")
    parser.add_argument("--test-tmdb", type=str, help="Test TMDb search for a show title")
    parser.add_argument("--enrich", action="store_true", help="Enrich top watched shows with TMDb metadata")
    parser.add_argument("--recommend", action="store_true", help="Generate 5 recommendations")
    args = parser.parse_args()

    api_key = os.getenv("TMDB_API_KEY")

    # 1) TMDb test
    if args.test_tmdb:
        if not api_key:
            print("ERROR: TMDB_API_KEY not found. Create a .env file with TMDB_API_KEY=...")
            return
        client = TMDbClient(api_key=api_key)
        hit = client.search_tv(args.test_tmdb)
        if not hit:
            print("No TMDb results found.")
            return
        details = client.get_tv_details(int(hit["id"]))
        print("Top match:")
        print(f"  title: {details.get('name')}")
        print(f"  first_air_date: {details.get('first_air_date')}")
        print(f"  genres: {[g['name'] for g in details.get('genres', [])]}")
        return

    # 2) Default message
    if not args.history and not args.enrich and not args.recommend:
        print("✅ OK. Try one of these:")
        print("  python -m app --history data/raw/netflix_viewing.csv")
        print("  python -m app --history data/raw/netflix_viewing.csv --enrich")
        print("  python -m app --recommend")
        print('  python -m app --test-tmdb "The Big Bang Theory"')
        return

    # 3) Just summarize Netflix CSV
    if args.history and not args.enrich and not args.recommend:
        history = load_netflix_history(args.history)
        summary = summarize_history(history, top_n=20)
        print("\nTop watched series (approx):")
        print(summary.to_string(index=False))
        return

    # 4) Enrich watched -> save watched_tmdb.csv
    if args.enrich:
        if not api_key:
            print("ERROR: TMDB_API_KEY not found. Create a .env file with TMDB_API_KEY=...")
            return
        if not args.history:
            print("ERROR: --history is required with --enrich")
            return

        client = TMDbClient(api_key=api_key)
        df = enrich_top_watched(client, args.history, top_n=20)

        os.makedirs("data/processed", exist_ok=True)
        out_path = "data/processed/watched_tmdb.csv"
        df.to_csv(out_path, index=False)

        print(f"\nSaved: {out_path}")
        print("\nPreview:")
        print(df.head(10).to_string(index=False))
        return

    # 5) Recommend (uses data/processed/watched_tmdb.csv)
    if args.recommend:
        if not api_key:
            print("ERROR: TMDB_API_KEY not found. Create a .env file with TMDB_API_KEY=...")
            return

        watched_path = "data/processed/watched_tmdb.csv"
        if not os.path.exists(watched_path):
            print("ERROR: missing data/processed/watched_tmdb.csv. Run:")
            print("  python -m app --history data/raw/netflix_viewing.csv --enrich")
            return

        watched_tmdb = pd.read_csv(watched_path)
        client = TMDbClient(api_key=api_key)

        candidates = build_candidate_pool(client, watched_tmdb, seed_n=10, per_seed=20)
        already = set(watched_tmdb.dropna(subset=["tmdb_id"])["tmdb_id"].astype(int).tolist())

        profile = build_user_profile(watched_tmdb)
        recs = recommend_from_candidates(profile, candidates, already_watched_tmdb_ids=already, top_k=5)

        print("\nTop 5 recommendations:")
        print(recs.to_string(index=False))
        return


if __name__ == "__main__":
    main()
