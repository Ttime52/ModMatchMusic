from src.utils.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert "paths" in cfg