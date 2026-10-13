"""Settings honor isolated workspace paths without import-time filesystem writes."""
from pathlib import Path
from src.config import Settings


def test_settings_environment_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("HMM_DATA_DIR", str(tmp_path / "data"))
    settings = Settings(_env_file=None)
    assert settings.data_dir == tmp_path / "data"
    assert isinstance(settings.models_dir, Path)
    assert not settings.data_dir.exists()
