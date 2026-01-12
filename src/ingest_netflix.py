import pandas as pd


def _pick_column(df: pd.DataFrame, possible_names: list[str]) -> str | None:
    # Try exact matches first, then case-insensitive matches
    for name in possible_names:
        if name in df.columns:
            return name

    lowered = {c.lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in lowered:
            return lowered[name.lower()]

    return None


def _to_series_title(raw_title: str) -> str:
    """
    Netflix often exports TV episode titles like:
      "Breaking Bad: Season 2: Grilled"
    We want the series name: "Breaking Bad"
    """
    t = (raw_title or "").strip()

    # Most common episode format has 2+ colons:
    if t.count(":") >= 2:
        return t.split(":")[0].strip()

    # Some locales use variations; keep a small heuristic:
    keywords = ["season", "episode", "series", "saison"]
    if ":" in t and any(k in t.lower() for k in keywords):
        return t.split(":")[0].strip()

    return t


def load_netflix_history(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    title_col = _pick_column(df, ["Title", "title"])
    if title_col is None:
        # Fallback: take the first column if Netflix format changes
        title_col = df.columns[0]

    date_col = _pick_column(df, ["Date", "Start Time", "date", "start time"])

    out = pd.DataFrame()
    out["raw_title"] = df[title_col].astype(str)

    if date_col is not None:
        out["watched_at"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        out["watched_at"] = pd.NaT

    out["series_title"] = out["raw_title"].apply(_to_series_title)
    return out


def summarize_history(history: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    summary = (
        history.groupby("series_title", dropna=False)
        .agg(
            watch_count=("series_title", "size"),
            last_watched=("watched_at", "max"),
        )
        .reset_index()
        .sort_values(["watch_count", "last_watched"], ascending=[False, False])
        .head(top_n)
    )
    return summary
