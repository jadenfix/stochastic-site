"""Basic placeholder tests for the project."""
from src.config import settings


def test_settings_directories_exist():
    assert settings.data_dir.exists()
    assert settings.cache_dir.exists()
    assert settings.models_dir.exists()
    assert settings.reports_dir.exists()
