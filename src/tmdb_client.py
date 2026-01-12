import os
import requests


class TMDbClient:
    def __init__(self, api_key: str, region: str = "ZA"):
        self.api_key = api_key
        self.region = region
        self.base_url = "https://api.themoviedb.org/3"

    def search_tv(self, query: str, year: int | None = None) -> dict | None:
        params = {
            "api_key": self.api_key,
            "query": query,
            "include_adult": "false",
        }
        if year is not None:
            params["first_air_date_year"] = str(year)

        r = requests.get(f"{self.base_url}/search/tv", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if not results:
            return None
        return results[0]  # best match

    def get_tv_details(self, tv_id: int) -> dict:
        params = {"api_key": self.api_key}
        r = requests.get(f"{self.base_url}/tv/{tv_id}", params=params, timeout=20)
        r.raise_for_status()
        return r.json()
