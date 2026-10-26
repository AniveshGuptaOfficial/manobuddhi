from backend.config import PROJECT_ROOT, Config


def test_env_override(monkeypatch):
    monkeypatch.setenv("MB_K_FAST_SAMPLES", "7")
    assert Config().k_fast_samples == 7


def test_data_dir_exists():
    assert (PROJECT_ROOT / "data").is_dir()
