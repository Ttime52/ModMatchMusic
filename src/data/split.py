
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter
import json

# Load Dataset
df = pd.read_parquet("data/processed/tracks.parquet")

print("Total songs:", len(df))
print("Total artists:", df["artist"].nunique())

artist_counts = df.groupby("artist").size()
print("Artists with only one song:", (artist_counts == 1).sum())

# Get All Genres
all_genres = set()

for genres in df["genres"]:
    for genre in genres:
        all_genres.add(genre)

print("All genres:", all_genres)

# Find Best Seed
best_seed = None
best_error = float("inf")

for seed in range(100):

    # First split: Train 80% / Temporary 20%
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=seed
    )

    train_idx, temp_idx = next(
        splitter.split(X=df, groups=df["artist"])
    )

    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    # Second split: Validation 10% / Test 10%
    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=seed
    )

    val_idx, test_idx = next(
        splitter_2.split(X=temp_df, groups=temp_df["artist"])
    )

    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    # Calculate Split Ratios
    train_ratio = len(train_df) / len(df) * 100
    val_ratio = len(val_df) / len(df) * 100
    test_ratio = len(test_df) / len(df) * 100

    ratio_error = (
        abs(train_ratio - 80)
        + abs(val_ratio - 10)
        + abs(test_ratio - 10)
    )

    # Check Genre Coverage
    train_genres = set()
    val_genres = set()
    test_genres = set()

    for genres in train_df["genres"]:
        for genre in genres:
            train_genres.add(genre)

    for genres in val_df["genres"]:
        for genre in genres:
            val_genres.add(genre)

    for genres in test_df["genres"]:
        for genre in genres:
            test_genres.add(genre)

    genre_coverage_ok = (
        train_genres == all_genres
        and val_genres == all_genres
        and test_genres == all_genres
    )

    # Update Best Seed
    if genre_coverage_ok and ratio_error < best_error:
        best_seed = seed
        best_error = ratio_error


if best_seed is None:
    raise ValueError("No valid seed found in range(100)")

print("\nBest seed:", best_seed)
print(f"Best ratio error: {best_error:.2f}")



# Split Dataset Using Best Seed
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=best_seed
)

train_idx, temp_idx = next(
    splitter.split(X=df, groups=df["artist"])
)

train_df = df.iloc[train_idx]
temp_df = df.iloc[temp_idx]

splitter_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.5,
    random_state=best_seed
)

val_idx, test_idx = next(
    splitter_2.split(X=temp_df, groups=temp_df["artist"])
)

val_df = temp_df.iloc[val_idx]
test_df = temp_df.iloc[test_idx]

# Validate Artist Leakage
train_artists = set(train_df["artist"])
val_artists = set(val_df["artist"])
test_artists = set(test_df["artist"])

assert not (train_artists & val_artists), \
    "Artist leakage between Train and Validation"

assert not (train_artists & test_artists), \
    "Artist leakage between Train and Test"

assert not (val_artists & test_artists), \
    "Artist leakage between Validation and Test"

print("\nArtist leakage check: PASSED")

# Validate Track Leakage
train_tracks = set(train_df["track_id"])
val_tracks = set(val_df["track_id"])
test_tracks = set(test_df["track_id"])

assert not (train_tracks & val_tracks), \
    "Track leakage between Train and Validation"

assert not (train_tracks & test_tracks), \
    "Track leakage between Train and Test"

assert not (val_tracks & test_tracks), \
    "Track leakage between Validation and Test"

assert train_df["track_id"].is_unique, \
    "Duplicate track_id in Train"

assert val_df["track_id"].is_unique, \
    "Duplicate track_id in Validation"

assert test_df["track_id"].is_unique, \
    "Duplicate track_id in Test"

assert (
    len(train_df) + len(val_df) + len(test_df) == len(df)
), "Total song count mismatch"

print("Track leakage check: PASSED")


# Validate Genre Coverage
train_genres = set()
val_genres = set()
test_genres = set()

for genres in train_df["genres"]:
    for genre in genres:
        train_genres.add(genre)

for genres in val_df["genres"]:
    for genre in genres:
        val_genres.add(genre)

for genres in test_df["genres"]:
    for genre in genres:
        test_genres.add(genre)

assert (
    train_genres == val_genres == test_genres == all_genres
), "Genre coverage check failed"

print("Genre coverage check: PASSED")

#Genre Distribution
train_genre_counts = Counter()
val_genre_counts = Counter()
test_genre_counts = Counter()

for genres in train_df["genres"]:
    for genre in genres:
        train_genre_counts[genre] += 1

for genres in val_df["genres"]:
    for genre in genres:
        val_genre_counts[genre] += 1

for genres in test_df["genres"]:
    for genre in genres:
        test_genre_counts[genre] += 1

print("\nTrain genres:", train_genre_counts)
print("Validation genres:", val_genre_counts)
print("Test genres:", test_genre_counts)

#Final Split Summary
train_ratio = len(train_df) / len(df) * 100
val_ratio = len(val_df) / len(df) * 100
test_ratio = len(test_df) / len(df) * 100

print("\n========== SPLIT SUMMARY ==========")
print(f"Best seed: {best_seed}")
print(f"Train:      {len(train_df)} songs ({train_ratio:.2f}%)")
print(f"Validation: {len(val_df)} songs ({val_ratio:.2f}%)")
print(f"Test:       {len(test_df)} songs ({test_ratio:.2f}%)")
print(f"Ratio error: {best_error:.2f}")
print("===================================")

#เอาลงไฟล์จริง

from pathlib import Path

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_parquet(output_dir / "train.parquet", index=False)
val_df.to_parquet(output_dir / "val.parquet", index=False)
test_df.to_parquet(output_dir / "test.parquet", index=False)

print("Dataset splits saved successfully")

#สร้าง Dictionary สำหรับ Metadata
metadata = {
    "method": "GroupShuffleSplit",
    "seed": best_seed,
    "seed_search_range": [0, 99],
    "target_ratio": {
        "train": 0.8,
        "validation": 0.1,
        "test": 0.1
    },
    "actual_ratio": {
        "train": train_ratio / 100,
        "validation": val_ratio / 100,
        "test": test_ratio / 100
    },
    "song_counts": {
        "total": len(df),
        "train": len(train_df),
        "validation": len(val_df),
        "test": len(test_df)
    },
    "artist_counts": {
        "train": len(train_artists),
        "validation": len(val_artists),
        "test": len(test_artists)
    },
    "genres": sorted(all_genres),
    "ratio_error_percentage_points": best_error
}

#บันทึกเป็น JSON
metadata_path = output_dir / "split_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print("Split metadata saved successfully!")
