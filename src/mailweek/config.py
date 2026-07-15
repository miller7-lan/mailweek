from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Protocol

import keyring
import tomli_w

from .errors import ConfigError, SecretError
from .schemas import AccountConfig, AppConfig


def default_config_path() -> Path:
    override = os.environ.get("MAILWEEK_CONFIG")
    return Path(override).expanduser() if override else Path.home() / ".mailweek" / "config.toml"


class ConfigStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_config_path()

    def load(self) -> AppConfig:
        if not self.path.exists():
            host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
            return AppConfig(ollama_host=host)
        try:
            with self.path.open("rb") as handle:
                raw = tomllib.load(handle)
            config = AppConfig.model_validate(raw)
        except Exception as exc:
            raise ConfigError(
                "config_invalid",
                f"无法读取配置：{self.path}",
                "修复 TOML 格式，或先备份后删除该文件并重新运行 mailweek init。",
            ) from exc
        if override_host := os.environ.get("OLLAMA_HOST"):
            config.ollama_host = override_host
        return config

    def save(self, config: AppConfig) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.path.parent.chmod(0o700)
            payload = config.model_dump(mode="json", exclude_none=True)
            with self.path.open("wb") as handle:
                tomli_w.dump(payload, handle)
            self.path.chmod(0o600)
        except OSError as exc:
            raise ConfigError(
                "config_write_failed", f"无法保存配置：{self.path}", str(exc)
            ) from exc

    def add_account(self, account: AccountConfig, *, make_active: bool = True) -> AppConfig:
        config = self.load()
        config.accounts[account.name] = account
        if make_active or config.active_account is None:
            config.active_account = account.name
        self.save(config)
        return config

    def remove_account(self, name: str) -> AppConfig:
        config = self.load()
        if name not in config.accounts:
            raise ConfigError("account_not_found", f"未找到邮箱账户：{name}")
        del config.accounts[name]
        if config.active_account == name:
            config.active_account = next(iter(config.accounts), None)
        self.save(config)
        return config

    def set_active_account(self, name: str) -> AppConfig:
        config = self.load()
        if name not in config.accounts:
            raise ConfigError("account_not_found", f"未找到邮箱账户：{name}")
        config.active_account = name
        self.save(config)
        return config

    def set_default_model(self, model: str) -> AppConfig:
        config = self.load()
        config.default_model = model
        self.save(config)
        return config


class SecretStore(Protocol):
    def get(self, account: AccountConfig) -> tuple[str | None, str]: ...

    def set(self, account: AccountConfig, password: str) -> None: ...

    def delete(self, account: AccountConfig) -> None: ...


class KeyringSecretStore:
    service_prefix = "mailweek"

    @staticmethod
    def _service(account: AccountConfig) -> str:
        return f"mailweek:{account.name}"

    def get(self, account: AccountConfig) -> tuple[str | None, str]:
        if value := os.environ.get("MAILWEEK_IMAP_PASSWORD"):
            return value, "env"
        try:
            return keyring.get_password(self._service(account), account.username), "keychain"
        except Exception as exc:
            raise SecretError(
                "keychain_read_failed",
                f"无法从 macOS 钥匙串读取账户 {account.name} 的密码。",
                str(exc),
            ) from exc

    def set(self, account: AccountConfig, password: str) -> None:
        try:
            keyring.set_password(self._service(account), account.username, password)
        except Exception as exc:
            raise SecretError(
                "keychain_write_failed",
                f"无法把账户 {account.name} 的密码写入 macOS 钥匙串。",
                str(exc),
            ) from exc

    def delete(self, account: AccountConfig) -> None:
        try:
            keyring.delete_password(self._service(account), account.username)
        except keyring.errors.PasswordDeleteError:
            return
        except Exception as exc:
            raise SecretError(
                "keychain_delete_failed",
                f"无法删除账户 {account.name} 的钥匙串密码。",
                str(exc),
            ) from exc
