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

    def get_similar_tv(self, tv_id: int, page: int = 1) -> list[dict]:
        params = {"api_key": self.api_key, "page": page}
        r = requests.get(f"{self.base_url}/tv/{tv_id}/similar", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("results", [])

    def get_tv_watch_providers(self, tv_id: int) -> dict:
        params = {"api_key": self.api_key}
        r = requests.get(f"{self.base_url}/tv/{tv_id}/watch/providers", params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def list_tv_providers(self, watch_region: str) -> list[dict]:
        params = {"api_key": self.api_key, "watch_region": watch_region}
        r = requests.get(f"{self.base_url}/watch/providers/tv", params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("results", [])

    def discover_tv(
        self,
        watch_region: str,
        with_watch_providers: str,
        with_watch_monetization_types: str = "flatrate",
        page: int = 1,
        sort_by: str = "popularity.desc",
    ) -> dict:
        params = {
            "api_key": self.api_key,
            "watch_region": watch_region,
            "with_watch_providers": with_watch_providers,  # e.g. "8|119"
            "with_watch_monetization_types": with_watch_monetization_types,
            "sort_by": sort_by,
            "page": page,
        }
        r = requests.get(f"{self.base_url}/discover/tv", params=params, timeout=20)
        r.raise_for_status()
        return r.json()
