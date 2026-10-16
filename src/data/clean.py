import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect

from src.utils.config import load_config

DetectorFactory.seed = 0

VERSION_RE = re.compile(
    r"\b(remix|live|acoustic|remaster(?:ed)?|demo|instrumental|karaoke|"
    r"radio edit|version|edit|mono|stereo)\b"
)


def normalize_text(s) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    return re.sub(r"\s+", " ", s)


def strip_punct(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s)).strip()


def base_title(title) -> str:
    """Remove common release/version markers from a song title."""
    t = normalize_text(title)
    t = re.sub(
        r"[\(\[][^)\]]*[\)\]]",
        lambda m: "" if VERSION_RE.search(m.group(0)) else m.group(0),
        t,
    )
    t = re.sub(r"\s-\s.*" + VERSION_RE.pattern + r".*$", "", t)
    return strip_punct(t)


def clean_lyrics(s) -> str:
    s = str(s).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\[[^\]]*\]", " ", s)  # [Chorus], [Verse 1], etc.
    s = re.sub(r"\d*Embed\s*$", "", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip()


def parse_genres(raw) -> list:
    if raw is None or pd.isna(raw):
        return []
    return [normalize_text(p) for p in re.split(r"[;,/|]", str(raw)) if p.strip()]


def map_genres(raw_genres, gmap) -> list:
    mapped = set()
    for genre in raw_genres:
        value = gmap.get(genre)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        mapped.update(normalize_text(v) for v in values if str(v).strip())
    return sorted(mapped)


def detect_lang(text: str) -> str:
    try:
        return detect(text[:1000])
    except Exception:
        return "unknown"


def make_id(artist_norm: str, title_base: str) -> str:
    return hashlib.sha1(f"{artist_norm}|{title_base}".encode("utf-8")).hexdigest()[:12]


def _find_column(raw_df: pd.DataFrame, column_config: dict, canonical: str, aliases=()) -> str:
    """Find a raw column using config first, then dataset-compatible aliases."""
    candidates = []
    configured = column_config.get(canonical)
    if configured:
        candidates.append(configured)
    candidates.extend([canonical, *aliases])

    for candidate in dict.fromkeys(candidates):
        if candidate in raw_df.columns:
            return candidate

    available = ", ".join(map(str, raw_df.columns))
    names = ", ".join(dict.fromkeys(map(str, candidates)))
    raise ValueError(
        f"Cannot find column for '{canonical}'. Tried: {names}. "
        f"Available columns: {available}"
    )


def load_cleanable_columns(raw_file, column_config) -> pd.DataFrame:
    """Load the four fields needed by the cleaner.

    The current hapan dataset calls its genre field ``tag``. Older or future
    datasets may call it ``genre``; both names are supported here.
    """
    raw_df = pd.read_csv(raw_file)
    fields = {
        "title": (),
        "artist": (),
        "genre": ("tag",),
        "lyrics": (),
    }
    selected = {
        canonical: _find_column(raw_df, column_config, canonical, aliases)
        for canonical, aliases in fields.items()
    }
    out = raw_df[[selected[field] for field in fields]].copy()
    out.columns = list(fields)
    return out


def _write_raw_genre_counts(df: pd.DataFrame, results_dir: Path) -> Path:
    counts = df["raw_genres"].explode().dropna().value_counts()
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "raw_genre_counts.csv"
    counts.to_csv(path, header=["count"])
    return path


def main():
    cfg = load_config()
    d, cols = cfg["data"], cfg["data"].get("columns", {})
    results_dir = Path(cfg.get("paths", {}).get("results", "results"))
    drops = {}

    df = load_cleanable_columns(d["raw_file"], cols)
    raw_rows = len(df)

    # 1) Required raw fields.
    n = len(df)
    df = df.dropna(subset=["title", "artist", "lyrics"])
    drops["missing_fields"] = n - len(df)

    # 2) Clean lyrics and remove empty/short entries.
    for column in ("title", "artist", "lyrics", "genre"):
        df[column] = df[column].astype(str).str.strip()
    df["lyrics_text"] = df["lyrics"].map(clean_lyrics)
    n = len(df)
    df = df[df["lyrics_text"].ne("")]
    drops["empty_lyrics"] = n - len(df)

    df["n_words"] = df["lyrics_text"].str.split().str.len()
    n = len(df)
    df = df[df["n_words"] >= d["min_lyrics_words"]]
    drops["lyrics_too_short"] = n - len(df)

    # 3) Keep the longest non-variant copy of duplicate songs.
    df["artist_norm"] = df["artist"].map(lambda s: strip_punct(normalize_text(s)))
    df["title_base"] = df["title"].map(base_title)
    df["is_variant"] = df["title"].map(
        lambda t: strip_punct(normalize_text(t)) != base_title(t)
    )
    n = len(df)
    df = (
        df.sort_values(
            ["is_variant", "n_words", "title"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates(["artist_norm", "title_base"], keep="first")
    )
    drops["duplicates_or_variants"] = n - len(df)

    # 4) Map the dataset's tag/genre values to canonical genre labels.
    df["raw_genres"] = df["genre"].map(parse_genres)
    gmap_path = Path(d["genre_map"])
    if not gmap_path.exists():
        path = _write_raw_genre_counts(df, results_dir)
        print(f"Missing {gmap_path}; raw genre counts were written to {path}")
        return

    gmap = json.loads(gmap_path.read_text(encoding="utf-8"))
    if not isinstance(gmap, dict):
        raise ValueError(f"Genre map must be a JSON object: {gmap_path}")
    gmap = {normalize_text(k): v for k, v in gmap.items()}

    df["genres"] = df["raw_genres"].map(lambda genres: map_genres(genres, gmap))
    n = len(df)
    df = df[df["genres"].str.len() > 0]
    drops["no_mapped_genre"] = n - len(df)

    # 5) Add stable language/id fields and write the processed dataset.
    df["language"] = df["lyrics_text"].map(detect_lang)
    df["track_id"] = [
        make_id(artist, title) for artist, title in zip(df["artist_norm"], df["title_base"])
    ]
    out = (
        df[["track_id", "title", "artist", "genres", "lyrics_text", "language", "n_words"]]
        .sort_values("track_id")
        .reset_index(drop=True)
    )
    output_path = Path(d["output_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    write_report(out, raw_rows, drops, results_dir=results_dir)
    print(f"Finished: {raw_rows} -> {len(out)} tracks")


def write_report(df, raw_rows, drops, results_dir="results"):
    genre_counts = df["genres"].explode().value_counts()
    lines = [
        "# Data report",
        "",
        f"- Raw rows: {raw_rows}",
        *[f"- Dropped ({key}): {value}" for key, value in drops.items()],
        f"- Final tracks: {len(df)}",
        f"- Unique artists: {df['artist'].nunique()}",
        f"- Lyrics length (words): mean {df['n_words'].mean():.0f}, median {df['n_words'].median():.0f}",
        "",
        "## Tracks per genre (a track can have several)",
        *[f"- {genre}: {count}" for genre, count in genre_counts.items()],
        "",
        "## Language",
        *[f"- {language}: {count}" for language, count in df["language"].value_counts().items()],
    ]
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    (results_path / "data_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
