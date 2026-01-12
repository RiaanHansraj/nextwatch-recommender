import argparse
import os
import pandas as pd
from src.providers import extract_provider_names, is_on_requested_services
import time

from dotenv import load_dotenv

from src.ingest_netflix import load_netflix_history, summarize_history
from src.tmdb_client import TMDbClient
from src.enrich_watched import enrich_top_watched

from src.recommender import build_user_profile, recommend_from_candidates
from src.build_candidates import build_candidate_pool

def main() -> None:
    load_dotenv()  # reads .env

    parser = argparse.ArgumentParser(
        prog="nextwatch",
        description="Recommend next shows from Netflix/Prime based on viewing history.",
    )

    parser.add_argument("--history", type=str, help="Path to Netflix viewing history CSV")
    parser.add_argument("--test-tmdb", type=str, help="Test TMDb search for a show title")
    parser.add_argument("--enrich", action="store_true", help="Enrich top watched shows with TMDb metadata")
    parser.add_argument("--recommend", action="store_true", help="Generate 5 recommendations (no provider filter yet)")
    parser.add_argument("--netflix", action="store_true", help="Filter recommendations to Netflix")
    parser.add_argument("--prime", action="store_true", help="Filter recommendations to Prime Video")
    parser.add_argument("--region", type=str, default=None, help="Region code like ZA (default from .env or ZA)")


    args = parser.parse_args()

    # TMDb test mode
    if args.test_tmdb:
        api_key = os.getenv("TMDB_API_KEY")
        region = os.getenv("TMDB_REGION", "ZA")
        if not api_key:
            print("ERROR: TMDB_API_KEY not found. Create a .env file with TMDB_API_KEY=...")
            return

        client = TMDbClient(api_key=api_key, region=region)
        hit = client.search_tv(args.test_tmdb)
        if not hit:
            print("No TMDb results found.")
            return

        tv_id = hit["id"]
        details = client.get_tv_details(tv_id)

        print("Top match:")
        print(f"  title: {details.get('name')}")
        print(f"  first_air_date: {details.get('first_air_date')}")
        print(f"  genres: {[g['name'] for g in details.get('genres', [])]}")
        return

    # Enrich mode
    if args.enrich:
        api_key = os.getenv("TMDB_API_KEY")
        region = os.getenv("TMDB_REGION", "ZA")
        if not api_key:
            print("ERROR: TMDB_API_KEY not found in .env")
            return
        if not args.history:
            print("ERROR: --history is required with --enrich")
            return

        client = TMDbClient(api_key=api_key, region=region)
        df = enrich_top_watched(client, args.history, top_n=20)

        out_path = "data/processed/watched_tmdb.csv"
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(out_path, index=False)

        print(f"\nSaved: {out_path}")
        print("\nPreview:")
        print(df.head(10).to_string(index=False))
        return

    if args.recommend:
        api_key = os.getenv("TMDB_API_KEY")
        region = os.getenv("TMDB_REGION", "ZA")
        if not api_key:
            print("ERROR: TMDB_API_KEY not found in .env")
            return

        watched_path = "data/processed/watched_tmdb.csv"
        if not os.path.exists(watched_path):
            print("ERROR: missing data/processed/watched_tmdb.csv. Run --enrich first.")
            return

        watched_tmdb = pd.read_csv(watched_path)
        client = TMDbClient(api_key=api_key, region=region)

        candidates = build_candidate_pool(client, watched_tmdb, seed_n=10, per_seed=20)
        already = set(watched_tmdb.dropna(subset=["tmdb_id"])["tmdb_id"].astype(int).tolist())

        profile = build_user_profile(watched_tmdb)
        recs = recommend_from_candidates(profile, candidates, already_watched_tmdb_ids=already, top_k=5)

                # Provider filter defaults: if neither flag is set, allow both
        allow_netflix = args.netflix or (not args.netflix and not args.prime)
        allow_prime = args.prime or (not args.netflix and not args.prime)

        region = args.region or os.getenv("TMDB_REGION", "ZA")

        filtered_rows = []
        for _, r in recs.iterrows():
            tv_id = int(r["tmdb_id"])
            payload = client.get_tv_watch_providers(tv_id)
            provider_names = extract_provider_names(payload, region)

            ok, matched = is_on_requested_services(provider_names, allow_netflix, allow_prime)
            if ok:
                row = r.to_dict()
                row["available_on"] = ", ".join(matched)
                filtered_rows.append(row)

            time.sleep(0.2)

        if not filtered_rows:
            print("\nNo recommendations found on the selected services in region:", region)
            print("Try rerunning with a larger candidate pool or without provider filtering.")
            return

        out = pd.DataFrame(filtered_rows)
        print("\nTop recommendations (filtered):")
        print(out.to_string(index=False))
        return


    # Default: Netflix CSV summary mode
    if not args.history:
        print("✅ OK: project runs. Use --history, --test-tmdb, or --enrich.")
        return

    history = load_netflix_history(args.history)
    summary = summarize_history(history, top_n=20)

    print("\nTop watched series (approx):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
