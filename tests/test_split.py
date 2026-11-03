import json
import pandas as pd
from pathlib import Path

#ตรวจว่าไฟล์ Parquet ที่บันทึกไว้มีจำนวนเพลงตรงกับ Metadata
def test_split_song_counts():
    data_dir = Path("data/processed")

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    with open(data_dir / "split_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    assert len(train_df) == metadata["song_counts"]["train"]
    assert len(val_df) == metadata["song_counts"]["validation"]
    assert len(test_df) == metadata["song_counts"]["test"]

#ตรวจ Artist Leakage
def test_no_artist_leakage():
    data_dir = Path("data/processed")

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    train_artists = set(train_df["artist"])
    val_artists = set(val_df["artist"])
    test_artists = set(test_df["artist"])

    assert train_artists.isdisjoint(val_artists) #isdisjoint เป็น Method ของ Python set ที่ใช้ตรวจว่า Set สองชุดไม่มีสมาชิกซ้ำกันเลยหรือไม่
    assert train_artists.isdisjoint(test_artists)
    assert val_artists.isdisjoint(test_artists)

#ตรวจว่า ทั้ง 3 Split มีครบทุก Genre ที่อยู่ใน Dataset ต้นฉบับ
def test_genre_coverage():
    data_dir = Path("data/processed")

    df = pd.read_parquet(data_dir / "tracks.parquet")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    all_genres = set()
    for genres in df["genres"]:
        for genre in genres:
            all_genres.add(genre)

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

    assert train_genres == all_genres
    assert val_genres == all_genres
    assert test_genres == all_genres

# ตรวจว่า ไม่มีเพลงเดียวกันอยู่ข้าม Train, Validation และ Test โดยใช้ track_id
def test_no_track_leakage():
    data_dir = Path("data/processed")

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    train_tracks = set(train_df["track_id"])
    val_tracks = set(val_df["track_id"])
    test_tracks = set(test_df["track_id"])

    assert train_tracks.isdisjoint(val_tracks)
    assert train_tracks.isdisjoint(test_tracks)
    assert val_tracks.isdisjoint(test_tracks)

    assert train_df["track_id"].is_unique
    assert val_df["track_id"].is_unique
    assert test_df["track_id"].is_unique

#อ่าน Dataset ต้นฉบับและ Seed จาก Metadata แล้วแบ่งข้อมูลใหม่ จากนั้นเปรียบเทียบ track_id กับไฟล์ Parquet ที่บันทึกไว้
from sklearn.model_selection import GroupShuffleSplit

def test_split_reproducibility():
    data_dir = Path("data/processed")

    df = pd.read_parquet(data_dir / "tracks.parquet")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    with open(data_dir / "split_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    seed = metadata["seed"]

    # แบ่งข้อมูลใหม่ด้วย Seed ที่บันทึกไว้
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=seed
    )

    train_idx, temp_idx = next(
        splitter.split(X=df, groups=df["artist"])
    )

    new_train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=seed
    )

    val_idx, test_idx = next(
        splitter_2.split(X=temp_df, groups=temp_df["artist"])
    )

    new_val_df = temp_df.iloc[val_idx]
    new_test_df = temp_df.iloc[test_idx]

    # TODO: เปรียบเทียบ Track ID ของข้อมูลใหม่กับไฟล์เดิม
    assert set(new_train_df["track_id"]) == set(train_df["track_id"])
    assert set(new_val_df["track_id"]) == set(val_df["track_id"])
    assert set(new_test_df["track_id"]) == set(test_df["track_id"])