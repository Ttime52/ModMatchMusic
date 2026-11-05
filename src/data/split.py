import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.utils.config import load_config


def _genres_in(df):
    return {genre for genres in df["genres"] for genre in genres}


def _seed_search_metadata(seed_candidates):
    if not seed_candidates:
        return []
    if seed_candidates == list(range(seed_candidates[0], seed_candidates[-1] + 1)):
        return [seed_candidates[0], seed_candidates[-1]]
    return {"values": seed_candidates}


def _split_once(df, ratios, seed):
    temporary_ratio = ratios["val"] + ratios["test"]
    if not 0 < temporary_ratio < 1:
        raise ValueError("Validation and test ratios must sum to a value between 0 and 1")
    first = GroupShuffleSplit(n_splits=1, test_size=temporary_ratio, random_state=seed)
    train_idx, temp_idx = next(first.split(X=df, groups=df["artist"]))
    train_df, temp_df = df.iloc[train_idx], df.iloc[temp_idx]
    second = GroupShuffleSplit(
        n_splits=1, test_size=ratios["test"] / temporary_ratio, random_state=seed
    )
    val_idx, test_idx = next(second.split(X=temp_df, groups=temp_df["artist"]))
    return train_df, temp_df.iloc[val_idx], temp_df.iloc[test_idx]


def split_dataset(df, split_config, seed_search=range(100)):
    """Return artist-isolated splits and metadata without performing I/O."""
    seed_candidates = list(seed_search)
    if not seed_candidates:
        raise ValueError("seed_search must contain at least one seed")
    ratios = {"train": float(split_config["train"]), "val": float(split_config["val"]), "test": float(split_config["test"])}
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test ratios must sum to 1")
    all_genres = _genres_in(df)
    best = None
    for seed in seed_candidates:
        train_df, val_df, test_df = _split_once(df, ratios, seed)
        actual = {"train": len(train_df) / len(df), "validation": len(val_df) / len(df), "test": len(test_df) / len(df)}
        coverage_ok = all(_genres_in(part) == all_genres for part in (train_df, val_df, test_df))
        error = sum(abs(actual[name] - ratios[key]) for name, key in (("train", "train"), ("validation", "val"), ("test", "test")))
        if coverage_ok and (best is None or error < best[0]):
            best = (error, seed, train_df, val_df, test_df, actual)
    if best is None:
        raise ValueError("No valid seed found in range")
    error, best_seed, train_df, val_df, test_df, actual = best
    parts = (train_df, val_df, test_df)
    artist_sets = [set(part["artist"]) for part in parts]
    if any(artist_sets[i] & artist_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Artist leakage between splits")
    track_sets = [set(part["track_id"]) for part in parts]
    if any(track_sets[i] & track_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Track leakage between splits")
    if not all(part["track_id"].is_unique for part in parts):
        raise AssertionError("Duplicate track_id in split")
    if sum(len(part) for part in parts) != len(df):
        raise AssertionError("Total song count mismatch")
    metadata = {
        "method": "GroupShuffleSplit", "seed": best_seed,
        "seed_search_range": _seed_search_metadata(seed_candidates),
        "target_ratio": {"train": ratios["train"], "validation": ratios["val"], "test": ratios["test"]},
        "actual_ratio": actual,
        "song_counts": {"total": len(df), "train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "artist_counts": {"train": len(artist_sets[0]), "validation": len(artist_sets[1]), "test": len(artist_sets[2])},
        "genres": sorted(all_genres), "ratio_error_percentage_points": error * 100,
    }
    return {"train": train_df, "validation": val_df, "test": test_df}, metadata


def write_split_outputs(splits, metadata, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits["train"].to_parquet(output_dir / "train.parquet", index=False)
    splits["validation"].to_parquet(output_dir / "val.parquet", index=False)
    splits["test"].to_parquet(output_dir / "test.parquet", index=False)
    with (output_dir / "split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    df = pd.read_parquet(data_cfg["output_file"])
    splits, metadata = split_dataset(df, data_cfg["split"])
    write_split_outputs(splits, metadata, cfg["paths"]["processed"])
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
