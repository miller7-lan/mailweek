from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from mailweek.cli import _open_review_item, _slash_help, app, main
from mailweek.config import ConfigStore, KeyringSecretStore
from mailweek.mail import MailService
from mailweek.schemas import (
    AccountConfig,
    EmailClassification,
    EmailContent,
    Theme,
    ToolExecutionResult,
)
from mailweek.tools import SessionContext, ToolRegistry, ToolSpec

runner = CliRunner()


def test_help_lists_agent_and_composable_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ask" in result.stdout
    assert "tools" in result.stdout
    assert "accounts" in result.stdout


def test_tools_list_has_stable_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MAILWEEK_CONFIG", str(tmp_path / "config.toml"))
    result = runner.invoke(app, ["--json", "tools", "list"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    names = {tool["name"] for tool in payload["tools"]}
    assert "emails.search" in names
    assert "reviews.generate" in names
    assert "reviews.summarize" in names
    assert all("shell" not in name for name in names)


def test_accounts_providers_lists_supported_sources(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MAILWEEK_CONFIG", str(tmp_path / "config.toml"))

    result = runner.invoke(app, ["--json", "accounts", "providers"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert {item["id"] for item in payload["providers"]} >= {
        "qq",
        "gmail",
        "outlook",
        "icloud",
        "custom",
    }


def test_accounts_list_detects_provider_for_legacy_account(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("MAILWEEK_CONFIG", str(config_path))
    ConfigStore(config_path).add_account(
        AccountConfig(
            name="work",
            email="user@qq.com",
            host="imap.qq.com",
            username="user@qq.com",
        )
    )

    result = runner.invoke(app, ["--json", "accounts", "list"])

    assert result.exit_code == 0, result.output
    account = json.loads(result.stdout)["accounts"][0]
    assert account["provider_id"] == "qq"
    assert account["provider"] == "QQ邮箱"


def test_accounts_add_uses_provider_preset_and_hides_password(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    captured: list[tuple[str, str]] = []
    monkeypatch.setenv("MAILWEEK_CONFIG", str(config_path))
    monkeypatch.setattr(
        KeyringSecretStore,
        "set",
        lambda _self, account, password: captured.append((account.name, password)),
    )

    result = runner.invoke(
        app,
        [
            "--json",
            "accounts",
            "add",
            "personal",
            "--email",
            "person@gmail.com",
            "--provider",
            "gmail",
            "--password-stdin",
            "--yes",
        ],
        input="private-app-password\n",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["provider"] == "Gmail"
    assert payload["host"] == "imap.gmail.com"
    assert "private-app-password" not in result.stdout
    assert captured == [("personal", "private-app-password")]
    account = ConfigStore(config_path).load().accounts["personal"]
    assert account.provider == "gmail"
    assert account.host == "imap.gmail.com"


def test_repl_help_discovers_account_add_and_provider_commands() -> None:
    help_text = _slash_help()

    assert "直接输入自然语言" in help_text
    assert "常用操作" in help_text
    assert "输入编号" in help_text
    assert "/list 待处理" in help_text
    assert "/account add" in help_text
    assert "/account providers" in help_text


def test_doctor_json_works_without_account_or_running_ollama(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MAILWEEK_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
    result = runner.invoke(app, ["--json", "doctor"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["ready"] is False
    assert payload["ollama"]["reachable"] is False


def test_main_turns_usage_errors_into_stable_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["mailweek", "--json", "nonexistent"])

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": False,
        "error": {
            "code": "cli_usage_error",
            "message": "No such command 'nonexistent'.",
            "hint": "运行 mailweek --help 查看命令格式。",
        },
    }


def test_models_benchmark_uses_nonzero_exit_for_quality_failure(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MAILWEEK_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setattr(
        "mailweek.cli.run_classifier_benchmark",
        lambda *_args, **_kwargs: {
            "ok": False,
            "kind": "synthetic_email_classification",
            "expected_matches": 3,
            "samples": 5,
        },
    )

    result = runner.invoke(app, ["--json", "models", "benchmark", "--suite"])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_models_benchmark_passes_named_case_to_quality_runner(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MAILWEEK_CONFIG", str(tmp_path / "config.toml"))
    seen: list[str | None] = []

    def fake_benchmark(*_args, **kwargs):
        seen.append(kwargs.get("case"))
        return {"ok": True, "case": kwargs.get("case")}

    monkeypatch.setattr("mailweek.cli.run_classifier_benchmark", fake_benchmark)

    result = runner.invoke(
        app,
        ["--json", "models", "benchmark", "--case", "invoice_record"],
    )

    assert result.exit_code == 0
    assert seen == ["invoice_record"]


def test_models_benchmark_rejects_unknown_case_as_usage_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MAILWEEK_CONFIG", str(tmp_path / "config.toml"))

    result = runner.invoke(
        app,
        ["--json", "models", "benchmark", "--case", "missing_case"],
    )

    assert result.exit_code == 2
    assert "unknown benchmark case" in result.output


def test_main_propagates_benchmark_quality_failure_exit_code(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("MAILWEEK_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setattr(
        "mailweek.cli.run_classifier_benchmark",
        lambda *_args, **_kwargs: {"ok": False, "samples": 1},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["mailweek", "--json", "models", "benchmark", "--case", "invoice_record"],
    )

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_accounts_test_accepts_account_name_as_positional_argument(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("MAILWEEK_CONFIG", str(config_path))
    ConfigStore(config_path).add_account(
        AccountConfig(
            name="work",
            email="user@example.com",
            host="imap.example.com",
            username="user@example.com",
        )
    )
    monkeypatch.setattr(
        MailService,
        "test_connection",
        lambda _self, account: {"ok": True, "account": account.name},
    )

    result = runner.invoke(app, ["--json", "accounts", "test", "work"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"ok": True, "account": "work"}


def test_open_review_item_reads_registered_email_through_tool(capsys) -> None:
    class Args(BaseModel):
        uid: str
        account: str | None = None
        folder: str | None = None
        max_chars: int = 6000

    classification = EmailClassification(
        uid="42",
        subject="项目确认",
        sender="owner@example.com",
        priority_score=80,
        theme=Theme.WORK_PROJECT,
        summary="等待确认",
        importance_reason="需要回复",
        action_required=True,
        suggested_action="回复负责人",
        confidence=0.9,
    )
    session = SessionContext(active_account="work", model="fake")
    session.classifications["42"] = classification
    session.last_review_uids = ["42"]
    session.last_account = "work"
    session.last_folder = "INBOX"
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "emails.get_content",
            "read",
            Args,
            lambda _args: ToolExecutionResult(
                output=EmailContent(
                    uid="42",
                    subject="项目确认",
                    sender="owner@example.com",
                    body="完整的只读测试正文",
                    content_type="text/plain",
                    truncated=False,
                ).model_dump(mode="json"),
                sensitive=True,
            ),
        )
    )
    state = SimpleNamespace(
        runtime=SimpleNamespace(services=SimpleNamespace(session=session))
    )
    agent = SimpleNamespace(registry=registry)

    opened = _open_review_item(state, agent, 1)  # type: ignore[arg-type]

    assert opened is True
    output = capsys.readouterr().out
    assert "AI 判断与建议" in output
    assert "完整的只读测试正文" in output
