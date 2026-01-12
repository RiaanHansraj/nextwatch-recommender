import argparse
import os

from dotenv import load_dotenv

from src.ingest_netflix import load_netflix_history, summarize_history
from src.tmdb_client import TMDbClient
from src.enrich_watched import enrich_top_watched


def main() -> None:
    load_dotenv()  # reads .env

    parser = argparse.ArgumentParser(
        prog="nextwatch",
        description="Recommend next shows from Netflix/Prime based on viewing history.",
    )

    parser.add_argument("--history", type=str, help="Path to Netflix viewing history CSV")
    parser.add_argument("--test-tmdb", type=str, help="Test TMDb search for a show title")
    parser.add_argument("--enrich", action="store_true", help="Enrich top watched shows with TMDb metadata")

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
