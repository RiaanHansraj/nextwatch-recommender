import time
import pandas as pd
from src.tmdb_client import TMDbClient


def build_candidate_pool(
    client: TMDbClient,
    watched_tmdb_df: pd.DataFrame,
    seed_n: int = 10,
    per_seed: int = 20,
    sleep_seconds: float = 0.25,
) -> pd.DataFrame:
    """
    For top seed_n watched shows, fetch up to per_seed similar shows.
    Then fetch details for each candidate.
    """
    seeds = watched_tmdb_df[watched_tmdb_df["status"] == "OK"].head(seed_n)
    candidate_ids: set[int] = set()

    for _, row in seeds.iterrows():
        tv_id = int(row["tmdb_id"])
        results = client.get_similar_tv(tv_id)
        for hit in results[:per_seed]:
            if "id" in hit:
                candidate_ids.add(int(hit["id"]))
        time.sleep(sleep_seconds)

    rows = []
    for cid in sorted(candidate_ids):
        details = client.get_tv_details(cid)
        genres = [g.get("name") for g in details.get("genres", []) if g.get("name")]
        rows.append(
            {
                "tmdb_id": cid,
                "tmdb_name": details.get("name"),
                "genres": ", ".join(genres),
                "overview": details.get("overview"),
            }
        )
        time.sleep(sleep_seconds)

    return pd.DataFrame(rows)
