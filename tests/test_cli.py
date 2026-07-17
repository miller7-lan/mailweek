from __future__ import annotations

import json
import sys
from datetime import date
from types import SimpleNamespace

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic import BaseModel
from typer.testing import CliRunner

from mailweek.cli import (
    CLIState,
    _open_review_item,
    _quick_choice_command,
    _quick_key_bindings,
    _slash_help,
    app,
    main,
    run_repl,
)
from mailweek.config import ConfigStore, KeyringSecretStore
from mailweek.mail import MailService
from mailweek.schemas import (
    AccountConfig,
    AgentRunResult,
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
    assert "/quick" in help_text
    assert "/today" in help_text
    assert "3  自定义命令" in help_text


def test_quick_choice_accepts_mac_number_key_without_enter() -> None:
    with create_pipe_input() as pipe_input:
        session: PromptSession[str] = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=_quick_key_bindings(lambda: True),
        )
        pipe_input.send_text("1")

        assert session.prompt() == "1"


def test_number_keys_keep_multi_digit_mail_indices_outside_quick_mode() -> None:
    with create_pipe_input() as pipe_input:
        session: PromptSession[str] = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=_quick_key_bindings(lambda: False),
        )
        pipe_input.send_text("12\n")

        assert session.prompt() == "12"


def test_quick_presets_map_to_review_today_and_custom_command() -> None:
    custom_requests: list[str] = []

    def custom_command() -> str:
        custom_requests.append("asked")
        return "/list P0"

    assert _quick_choice_command("1", custom_command) == "/review"
    assert _quick_choice_command("2", custom_command) == "/today"
    assert _quick_choice_command("3", custom_command) == "/list P0"
    assert custom_requests == ["asked"]


def test_repl_custom_preset_runs_the_entered_command(monkeypatch) -> None:
    inputs = iter(("3", "找出今天需要处理的账单", "/exit"))
    prompts: list[str] = []

    class FakePromptSession:
        def __init__(self, **_kwargs) -> None:
            pass

        def prompt(self, *_args, **_kwargs) -> str:
            prompt = _args[0] if _args else ""
            prompts.append(str(getattr(prompt, "value", prompt)))
            return next(inputs)

    requests: list[str] = []

    class FakeAgent:
        events = SimpleNamespace(close=lambda: None)

        def run(self, prompt: str) -> AgentRunResult:
            requests.append(prompt)
            return AgentRunResult(answer="已完成")

    session = SessionContext(active_account=None, model="fake")
    runtime = SimpleNamespace(
        services=SimpleNamespace(session=session),
        close=lambda: None,
    )
    monkeypatch.setattr("mailweek.cli.PromptSession", FakePromptSession)
    monkeypatch.setattr("mailweek.cli._agent", lambda _state: FakeAgent())

    run_repl(CLIState(json_mode=False, runtime=runtime))  # type: ignore[arg-type]

    assert requests == ["找出今天需要处理的账单"]
    assert "mailweek:custom" in prompts[1]


def test_repl_today_preset_uses_an_exact_date_range(monkeypatch) -> None:
    inputs = iter(("2", "/exit"))

    class FakePromptSession:
        def __init__(self, **_kwargs) -> None:
            pass

        def prompt(self, *_args, **_kwargs) -> str:
            return next(inputs)

    requests: list[str] = []

    class FakeAgent:
        events = SimpleNamespace(close=lambda: None)

        def run(self, prompt: str) -> AgentRunResult:
            requests.append(prompt)
            return AgentRunResult(answer="已完成")

    session = SessionContext(active_account=None, model="fake")
    runtime = SimpleNamespace(
        services=SimpleNamespace(session=session),
        close=lambda: None,
    )
    monkeypatch.setattr("mailweek.cli.PromptSession", FakePromptSession)
    monkeypatch.setattr("mailweek.cli._agent", lambda _state: FakeAgent())
    monkeypatch.setattr("mailweek.cli._today", lambda: date(2026, 7, 18))

    run_repl(CLIState(json_mode=False, runtime=runtime))  # type: ignore[arg-type]

    assert requests == ["回顾 2026-07-18 到 2026-07-18 的邮件，按重要性排序。"]


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
