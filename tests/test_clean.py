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