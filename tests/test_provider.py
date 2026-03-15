import os
import json
import pytest
from unittest.mock import patch, MagicMock
from concept_dfs.provider import (
    get_api_key,
    save_api_key,
    ensure_api_key,
    get_saved_model,
    save_model,
    resolve_model,
    select_model,
    _load_auth,
    _save_auth,
    MissingAPIKeyError,
    PROVIDERS,
    AUTH_FILE,
)
import concept_dfs.provider as provider_module


@pytest.fixture(autouse=True)
def isolate_auth(tmp_path, monkeypatch):
    """Redirect auth storage to a temp directory for all tests."""
    auth_dir = tmp_path / "concept-dfs"
    auth_file = auth_dir / "auth.json"
    monkeypatch.setattr(provider_module, "CONFIG_DIR", auth_dir)
    monkeypatch.setattr(provider_module, "AUTH_FILE", auth_file)
    # Clear any provider env vars
    for info in PROVIDERS.values():
        monkeypatch.delenv(info.env_key, raising=False)
    monkeypatch.delenv("CONCEPT_DFS_MODEL", raising=False)
    yield auth_file


# ── API Key Tests ──


def test_get_api_key_from_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key-123")
    assert get_api_key("gemini") == "env-key-123"


def test_get_api_key_from_auth_file(isolate_auth):
    save_api_key("gemini", "saved-key-456")
    assert get_api_key("gemini") == "saved-key-456"


def test_env_overrides_auth_file(monkeypatch, isolate_auth):
    save_api_key("gemini", "saved-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
    assert get_api_key("gemini") == "env-key"


def test_get_api_key_unknown_provider():
    assert get_api_key("nonexistent") is None


def test_save_and_load_multiple_keys(isolate_auth):
    save_api_key("gemini", "gkey")
    save_api_key("openrouter", "orkey")

    assert get_api_key("gemini") == "gkey"
    assert get_api_key("openrouter") == "orkey"


def test_auth_file_permissions(isolate_auth):
    save_api_key("gemini", "test-key")
    auth_file = isolate_auth
    assert auth_file.exists()
    mode = auth_file.stat().st_mode & 0o777
    assert mode == 0o600


def test_ensure_api_key_raises_when_missing(isolate_auth):
    with pytest.raises(MissingAPIKeyError) as exc_info:
        ensure_api_key("openrouter")
    assert exc_info.value.provider_id == "openrouter"
    assert exc_info.value.env_key == "OPENROUTER_API_KEY"


def test_ensure_api_key_uses_existing(monkeypatch, isolate_auth):
    monkeypatch.setenv("GOOGLE_API_KEY", "existing-key")
    key = ensure_api_key("gemini")
    assert key == "existing-key"


def test_ensure_api_key_unknown_provider(isolate_auth):
    with pytest.raises(ValueError, match="Unknown provider"):
        ensure_api_key("nonexistent-provider")


@patch("concept_dfs.provider.Prompt.ask")
def test_force_auth(mock_ask, monkeypatch, isolate_auth):
    monkeypatch.setenv("CONCEPT_DFS_MODEL", "gemini:gemini-2.5-pro")
    mock_ask.return_value = "new-forced-key"

    # Should prompt for gemini key and save it
    from concept_dfs.provider import force_auth

    force_auth()

    assert get_api_key("gemini") == "new-forced-key"
    assert os.environ["GOOGLE_API_KEY"] == "new-forced-key"


# ── Model Selection Tests ──


def test_resolve_model_from_env(monkeypatch, isolate_auth):
    monkeypatch.setenv("CONCEPT_DFS_MODEL", "openai:gpt-4o")
    assert resolve_model() == "openai:gpt-4o"


def test_resolve_model_from_saved(isolate_auth):
    save_model("openrouter:google/gemini-2.5-pro")
    assert resolve_model() == "openrouter:google/gemini-2.5-pro"


def test_resolve_model_env_overrides_saved(monkeypatch, isolate_auth):
    save_model("gemini:gemini-2.5-pro")
    monkeypatch.setenv("CONCEPT_DFS_MODEL", "openai:gpt-4o")
    assert resolve_model() == "openai:gpt-4o"


@patch("concept_dfs.provider.Prompt.ask")
def test_select_model_interactive(mock_ask, isolate_auth):
    # First call: provider number (2 = openrouter), second call: model name (accept default)
    provider_ids = list(PROVIDERS.keys())
    mock_ask.side_effect = ["2", PROVIDERS[provider_ids[1]].default_model]

    result = select_model()

    expected_provider = provider_ids[1]  # openrouter
    expected_model = PROVIDERS[expected_provider].default_model
    assert result == f"{expected_provider}:{expected_model}"
    assert get_saved_model() == result


@patch("concept_dfs.provider.Prompt.ask")
def test_select_model_custom_model(mock_ask, isolate_auth):
    # Pick openrouter (provider 2), then type a custom model
    mock_ask.side_effect = ["2", "anthropic/claude-3-opus"]

    result = select_model()

    assert result == "openrouter:anthropic/claude-3-opus"
    assert get_saved_model() == result


def test_save_and_get_model(isolate_auth):
    assert get_saved_model() is None
    save_model("gemini:gemini-2.5-pro")
    assert get_saved_model() == "gemini:gemini-2.5-pro"
