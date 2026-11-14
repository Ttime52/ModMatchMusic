import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.clean import safe_artist_key
from src.data.split import split_dataset, write_split_outputs
from src.utils.config import load_config


def synthetic_tracks():
    rows = []
    for artist_idx in range(10):
        for song_idx in range(2):
            rows.append({
                "track_id": f"track-{artist_idx}-{song_idx}",
                "artist": f"artist-{artist_idx}",
                "genres": ["pop", "rock"],
                "title": f"song-{artist_idx}-{song_idx}",
            })
    return pd.DataFrame(rows)


def test_split_uses_config_ratios_and_preserves_rows():
    df = synthetic_tracks()
    splits, metadata = split_dataset(df, load_config()["data"]["split"])
    assert sum(len(part) for part in splits.values()) == len(df)
    assert metadata["song_counts"]["total"] == len(df)
    assert metadata["song_counts"]["train"] == len(splits["train"])
    assert metadata["song_counts"]["validation"] == len(splits["validation"])
    assert metadata["song_counts"]["test"] == len(splits["test"])
    assert metadata["seed_search_range"] == [0, 99]


def test_split_is_reproducible_and_disjoint():
    df = synthetic_tracks()
    first, first_meta = split_dataset(df, {"train": 0.8, "val": 0.1, "test": 0.1})
    second, second_meta = split_dataset(df, {"train": 0.8, "val": 0.1, "test": 0.1})
    assert first_meta["seed"] == second_meta["seed"]
    for name in first:
        assert set(first[name]["track_id"]) == set(second[name]["track_id"])
    assert set(first["train"]["artist"]).isdisjoint(first["validation"]["artist"])
    assert set(first["train"]["artist"]).isdisjoint(first["test"]["artist"])
    assert set(first["validation"]["artist"]).isdisjoint(first["test"]["artist"])


def test_split_isolates_artist_name_variations():
    rows = []
    artists = ["Taylor Swift", "taylor swift"] + [f"artist-{i}" for i in range(2, 12)]
    for artist_idx, artist in enumerate(artists):
        for song_idx in range(2):
            rows.append(
                {
                    "track_id": f"variant-track-{artist_idx}-{song_idx}",
                    "artist": artist,
                    "genres": ["pop"],
                    "title": f"song-{artist_idx}-{song_idx}",
                }
            )

    splits, metadata = split_dataset(
        pd.DataFrame(rows),
        {"train": 0.8, "val": 0.1, "test": 0.1},
        seed_search=[0],
    )
    normalized_artist_sets = [
        set(part["artist"].map(safe_artist_key)) for part in splits.values()
    ]
    assert all(
        normalized_artist_sets[i].isdisjoint(normalized_artist_sets[j])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    assert metadata["artist_counts"] == {
        name: len(normalized_artist_sets[index])
        for index, name in enumerate(("train", "validation", "test"))
    }
    taylor_splits = {
        name
        for name, part in splits.items()
        if part["artist"].map(safe_artist_key).eq("taylor swift").any()
    }
    assert len(taylor_splits) == 1


def test_split_preserves_genre_coverage():
    splits, metadata = split_dataset(synthetic_tracks(), {"train": 0.8, "val": 0.1, "test": 0.1})
    assert all({g for row in part["genres"] for g in row} == set(metadata["genres"]) for part in splits.values())


def test_custom_non_contiguous_seed_search_is_recorded_accurately():
    _, metadata = split_dataset(
        synthetic_tracks(),
        {"train": 0.8, "val": 0.1, "test": 0.1},
        seed_search=(seed for seed in (2, 5, 9)),
    )
    assert metadata["seed_search_range"] == {"values": [2, 5, 9]}


def test_outputs_and_metadata_are_written(tmp_path):
    splits, metadata = split_dataset(synthetic_tracks(), {"train": 0.8, "val": 0.1, "test": 0.1})
    write_split_outputs(splits, metadata, tmp_path)
    assert (tmp_path / "train.parquet").exists()
    assert (tmp_path / "val.parquet").exists()
    assert (tmp_path / "test.parquet").exists()
    assignments = pd.read_csv(tmp_path / "split_assignments.csv")
    assert list(assignments.columns) == ["track_id", "split"]
    assert len(assignments) == len(synthetic_tracks())
    assert assignments["track_id"].is_unique
    assert assignments["track_id"].tolist() == sorted(assignments["track_id"])
    assert set(assignments["split"]) == {"train", "validation", "test"}
    for name, part in splits.items():
        assert set(assignments.loc[assignments["split"].eq(name), "track_id"]) == set(
            part["track_id"]
        )
    saved = json.loads((tmp_path / "split_metadata.json").read_text())
    assert saved["song_counts"] == metadata["song_counts"]


def test_import_has_no_file_io_side_effects(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("import performed file I/O")

    monkeypatch.setattr(pd, "read_parquet", fail)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    importlib.reload(importlib.import_module("src.data.split"))


def test_invalid_ratios_fail():
    with pytest.raises(ValueError, match="sum to 1"):
        split_dataset(synthetic_tracks(), {"train": 0.7, "val": 0.1, "test": 0.1})
