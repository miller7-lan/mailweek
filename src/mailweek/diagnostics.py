from __future__ import annotations

import stat
from typing import Any

from .config import ConfigStore, SecretStore
from .ollama_client import OllamaAdapter
from .providers import provider_id, provider_label


def run_doctor(
    config_store: ConfigStore, secrets: SecretStore, ollama: OllamaAdapter
) -> dict[str, Any]:
    config = config_store.load()
    path = config_store.path
    config_mode: str | None = None
    if path.exists():
        config_mode = oct(stat.S_IMODE(path.stat().st_mode))
    accounts: list[dict[str, Any]] = []
    for name, account in config.accounts.items():
        try:
            password, source = secrets.get(account)
            password_available = bool(password)
        except Exception:
            source = "unavailable"
            password_available = False
        accounts.append(
            {
                "name": name,
                "email": account.email,
                "provider_id": provider_id(
                    account.provider,
                    email=account.email,
                    host=account.host,
                ),
                "provider": provider_label(
                    account.provider,
                    email=account.email,
                    host=account.host,
                ),
                "host": account.host,
                "active": name == config.active_account,
                "password_available": password_available,
                "password_source": source,
            }
        )
    ollama_error: str | None = None
    models: list[str] = []
    try:
        models = ollama.installed_models()
    except Exception as exc:
        ollama_error = str(exc)
    configured_model = config.default_model
    ready = bool(config.accounts) and ollama_error is None and configured_model in models
    missing: list[str] = []
    if not config.accounts:
        missing.append("运行 mailweek accounts add work 配置邮箱，并在隐藏提示中输入应用专用密码")
    if ollama_error:
        missing.append("启动本地 Ollama")
    elif configured_model not in models:
        missing.append(f"运行 mailweek models pull {configured_model}")
    return {
        "ok": True,
        "ready": ready,
        "config": {
            "path": str(path),
            "exists": path.exists(),
            "mode": config_mode,
            "secure_mode": config_mode in {None, "0o600"},
        },
        "accounts": accounts,
        "active_account": config.active_account,
        "ollama": {
            "host": config.ollama_host,
            "reachable": ollama_error is None,
            "error": ollama_error,
            "models": models,
            "configured_model": configured_model,
            "configured_model_installed": configured_model in models,
        },
        "missing_steps": missing,
    }
