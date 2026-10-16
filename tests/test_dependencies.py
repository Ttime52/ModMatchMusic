from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_dependencies_are_present_in_lockfile():
    requirements = {
        line.strip().split("==", 1)[0].lower()
        for line in (PROJECT_ROOT / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    locked = {
        line.strip().split("==", 1)[0].lower()
        for line in (PROJECT_ROOT / "requirements.lock.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "langdetect" in requirements
    assert "langdetect" in locked
