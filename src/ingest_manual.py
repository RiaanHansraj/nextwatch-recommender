import pandas as pd
from src.tmdb_client import TMDbClient


def build_watched_from_titles(client: TMDbClient, titles: list[str]) -> pd.DataFrame:
    rows = []
    for t in titles:
        title = (t or "").strip()
        if not title:
            continue

        hit = client.search_tv(title)
        if not hit:
            rows.append(
                {
                    "series_title": title,
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
                "series_title": title,
                "watch_count": 1,
                "tmdb_id": tv_id,
                "tmdb_name": details.get("name"),
                "genres": ", ".join(genres),
                "overview": details.get("overview"),
                "status": "OK",
            }
        )

    return pd.DataFrame(rows)
