import argparse
import os

from dotenv import load_dotenv

from src.ingest_netflix import load_netflix_history, summarize_history
from src.tmdb_client import TMDbClient


def main() -> None:
    load_dotenv()  # reads .env into environment variables

    parser = argparse.ArgumentParser(
        prog="nextwatch",
        description="Recommend next shows from Netflix/Prime based on viewing history.",
    )
    parser.add_argument("--history", type=str, help="Path to Netflix viewing history CSV")
    parser.add_argument("--test-tmdb", type=str, help="Test TMDb search for a show title")
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

    # Netflix CSV summary mode (what you already have)
    if not args.history:
        print("✅ OK: project runs. Use --history to summarize Netflix CSV or --test-tmdb to test TMDb.")
        return

    history = load_netflix_history(args.history)
    summary = summarize_history(history, top_n=20)

    print("\nTop watched series (approx):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
