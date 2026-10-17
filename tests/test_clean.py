import json

import pandas as pd
import pytest

import src.data.clean as clean
from src.data.clean import base_title, clean_lyrics, map_genres, parse_genres


def test_base_title_strips_version_suffixes():
    assert base_title("Hello (Live at Wembley)") == base_title("Hello")
    assert base_title("Hello - Remastered 2011") == base_title("Hello")


def test_base_title_keeps_normal_parentheses():
    assert base_title("Hello (Goodbye)") != base_title("Hello")


def test_clean_lyrics_removes_section_tags():
    assert clean_lyrics("[Chorus]\nla la\n[Verse 1]\nso so") == "la la so so"


def test_parse_and_map_genres():
    raw = parse_genres("Alt Rock; Rock | Hard Rock")
    gmap = {"alt rock": "rock", "rock": "rock", "hard rock": "rock"}
    assert map_genres(raw, gmap) == ["rock"]


def _configure_pipeline(monkeypatch, tmp_path, rows, *, genre_map):
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed" / "tracks.parquet"
    results_path = tmp_path / "results"
    pd.DataFrame(rows).to_csv(raw_path, index=False)

    genre_map_path = tmp_path / "genre_map.json"
    if genre_map is not None:
        genre_map_path.write_text(json.dumps(genre_map), encoding="utf-8")

    config = {
        "paths": {"results": str(results_path)},
        "data": {
            "raw_file": str(raw_path),
            "output_file": str(output_path),
            "genre_map": str(genre_map_path),
            "columns": {},
            "min_lyrics_words": 1,
        },
    }
    monkeypatch.setattr(clean, "load_config", lambda: config)
    return output_path, results_path


def _lyrics():
    return "This is a sufficiently long lyric line for the cleaning pipeline. " * 5


def test_genre_validation_happens_before_deduplication(monkeypatch, tmp_path):
    output_path, _ = _configure_pipeline(
        monkeypatch,
        tmp_path,
        [
            {
                "title": "Shared Song",
                "artist": "Artist",
                "genre": "unmapped",
                "lyrics": _lyrics() * 2,
            },
            {
                "title": "Shared Song (Live)",
                "artist": "Artist",
                "genre": "rock",
                "lyrics": _lyrics(),
            },
        ],
        genre_map={"rock": "rock"},
    )

    clean.main()

    output = pd.read_parquet(output_path)
    assert len(output) == 1
    assert output.loc[0, "title"] == "Shared Song (Live)"
    assert output.loc[0, "genres"] == ["rock"]


def test_blank_artist_and_title_are_dropped(monkeypatch, tmp_path):
    output_path, _ = _configure_pipeline(
        monkeypatch,
        tmp_path,
        [
            {"title": "   ", "artist": "Artist", "genre": "rock", "lyrics": _lyrics()},
            {"title": "Valid title", "artist": " \t ", "genre": "rock", "lyrics": _lyrics()},
            {"title": "Valid title", "artist": "Artist", "genre": "rock", "lyrics": _lyrics()},
        ],
        genre_map={"rock": "rock"},
    )

    clean.main()

    output = pd.read_parquet(output_path)
    assert output[["title", "artist"]].to_dict("records") == [
        {"title": "Valid title", "artist": "Artist"}
    ]


def test_missing_genre_map_fails_and_writes_diagnostic(monkeypatch, tmp_path):
    output_path, results_path = _configure_pipeline(
        monkeypatch,
        tmp_path,
        [{"title": "Title", "artist": "Artist", "genre": "rock", "lyrics": _lyrics()}],
        genre_map=None,
    )

    with pytest.raises(FileNotFoundError, match="raw genre counts"):
        clean.main()

    assert not output_path.exists()
    assert (results_path / "raw_genre_counts.csv").exists()

from src.data.clean import safe_title_key, safe_artist_key, make_id

def test_punctuation_only_titles_have_different_keys():
    assert safe_title_key("!!!") == "!!!"
    assert safe_title_key("???") == "???"
    assert safe_title_key("!!!") != safe_title_key("???")


def test_punctuation_only_artists_have_different_keys():
    assert safe_artist_key("!!!") != safe_artist_key("???")


def test_track_ids_for_punctuation_titles_are_unique():
    artist = safe_artist_key("Artist A")

    id1 = make_id(artist, safe_title_key("!!!"))
    id2 = make_id(artist, safe_title_key("???"))

    assert id1 != id2

def test_punctuation_title_versions_share_key():
    assert clean.safe_title_key("!!!") == clean.safe_title_key("!!! (Live)")
    assert clean.safe_title_key("???") == clean.safe_title_key("??? (Remastered)")
    assert clean.safe_title_key("!!!") != clean.safe_title_key("???")

def test_punctuation_live_version_deduplicated(monkeypatch, tmp_path):
    output_path, _ = _configure_pipeline(
        monkeypatch,
        tmp_path,
        [
            {
                "title": "!!!",
                "artist": "Artist A",
                "genre": "pop",
                "lyrics": _lyrics(),
            },
            {
                "title": "!!! (Live)",
                "artist": "Artist A",
                "genre": "pop",
                "lyrics": _lyrics(),
            },
            {
                "title": "???",
                "artist": "Artist A",
                "genre": "pop",
                "lyrics": _lyrics(),
            },
        ],
        genre_map={"pop": "pop"},
    )

    clean.main()
    output = pd.read_parquet(output_path)

    assert len(output) == 2
    assert set(output["title"]) == {"!!!", "???"}
    assert output["track_id"].is_unique

def test_punctuation_titles_preserved_in_pipeline(monkeypatch, tmp_path):
    output_path, _ = _configure_pipeline(
        monkeypatch,
        tmp_path,
        [
            {
                "title": "!!!",
                "artist": "Artist A",
                "genre": "pop",
                "lyrics": _lyrics(),
            },
            {
                "title": "???",
                "artist": "Artist A",
                "genre": "pop",
                "lyrics": _lyrics(),
            },
        ],
        genre_map={"pop": "pop"},
    )

    clean.main()

    output = pd.read_parquet(output_path)

    assert len(output) == 2
    assert set(output["title"]) == {"!!!", "???"}
    assert output["track_id"].is_unique