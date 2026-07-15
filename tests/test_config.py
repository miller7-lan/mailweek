from __future__ import annotations

import stat

from mailweek.config import ConfigStore
from mailweek.schemas import AccountConfig


def test_config_round_trip_and_permissions(tmp_path) -> None:
    path = tmp_path / ".mailweek" / "config.toml"
    store = ConfigStore(path)
    account = AccountConfig(
        name="work",
        email="me@example.com",
        host="imap.example.com",
        username="me@example.com",
    )
    store.add_account(account)

    loaded = store.load()
    assert loaded.active_account == "work"
    assert loaded.accounts["work"].host == "imap.example.com"
    assert loaded.accounts["work"].provider == "auto"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_environment_overrides_ollama_host(tmp_path, monkeypatch) -> None:
    path = tmp_path / "config.toml"
    store = ConfigStore(path)
    store.save(store.load())
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9999")
    assert store.load().ollama_host == "http://127.0.0.1:9999"
