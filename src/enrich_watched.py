import time
import pandas as pd

from src.ingest_netflix import load_netflix_history, summarize_history
from src.tmdb_client import TMDbClient


def enrich_top_watched(
    client: TMDbClient,
    history_csv_path: str,
    top_n: int = 20,
    sleep_seconds: float = 0.25,
) -> pd.DataFrame:
    history = load_netflix_history(history_csv_path)
    top = summarize_history(history, top_n=top_n)

    rows = []
    for _, row in top.iterrows():
        title = str(row["series_title"]).strip()
        watch_count = int(row["watch_count"])

        hit = client.search_tv(title)
        if not hit:
            rows.append(
                {
                    "series_title": title,
                    "watch_count": watch_count,
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
        overview = details.get("overview")

        rows.append(
            {
                "series_title": title,
                "watch_count": watch_count,
                "tmdb_id": tv_id,
                "tmdb_name": details.get("name"),
                "genres": ", ".join(genres),
                "overview": overview,
                "status": "OK",
            }
        )

        time.sleep(sleep_seconds)

    return pd.DataFrame(rows)
