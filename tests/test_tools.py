from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, Field

from mailweek.errors import ToolError
from mailweek.ollama_client import NormalizedMessage
from mailweek.schemas import (
    AccountConfig,
    AppConfig,
    EmailContent,
    EmailHeader,
    SearchResult,
    ToolExecutionResult,
)
from mailweek.tools import (
    ApprovalGate,
    RuntimeServices,
    SearchEmailsArgs,
    SessionContext,
    ToolRegistry,
    ToolSpec,
    build_tool_registry,
)


class CountArgs(BaseModel):
    limit: int = Field(ge=1, le=500)


def test_search_defaults_to_recent_seven_days() -> None:
    parsed = SearchEmailsArgs.model_validate({})
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()

    assert parsed.date_to == today
    assert parsed.date_from == today - timedelta(days=6)


def test_registry_rejects_unknown_tools() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolError, match="未知"):
        registry.execute("shell.exec", {"command": "rm -rf /"})


def test_registry_validates_arguments_before_handler() -> None:
    called = False

    def handler(_args):
        nonlocal called
        called = True
        return ToolExecutionResult(output={"ok": True})

    registry = ToolRegistry()
    registry.register(ToolSpec("emails.search", "search", CountArgs, handler))
    with pytest.raises(ToolError) as error:
        registry.execute("emails.search", {"limit": 0})
    assert error.value.code == "invalid_tool_arguments"
    assert called is False


def test_approval_gate_blocks_large_read_when_confirmation_unavailable() -> None:
    def handler(args):
        return ToolExecutionResult(output={"limit": args.limit})

    def approval(args):
        return "allow large search?" if args.limit > 100 else None

    registry = ToolRegistry(approvals=ApprovalGate())
    registry.register(ToolSpec("emails.search", "search", CountArgs, handler, approval=approval))
    with pytest.raises(ToolError) as error:
        registry.execute("emails.search", {"limit": 101})
    assert error.value.code == "approval_required"


def test_registry_wraps_unexpected_handler_errors_without_leaking_details() -> None:
    def handler(_args):
        raise RuntimeError("password=do-not-leak")

    registry = ToolRegistry()
    registry.register(ToolSpec("emails.search", "search", CountArgs, handler))

    with pytest.raises(ToolError) as error:
        registry.execute("emails.search", {"limit": 1})

    assert error.value.code == "tool_execution_failed"
    assert "do-not-leak" not in str(error.value)


def test_reviews_generate_orchestrates_single_email_calls_and_programmatic_summary() -> None:
    account = AccountConfig(
        name="work",
        email="user@example.com",
        host="imap.example.com",
        username="user@example.com",
    )

    class FakeConfigStore:
        def load(self):
            return AppConfig(active_account="work", accounts={"work": account})

    class FakeMail:
        def search(self, _account, **_kwargs):
            return SearchResult(
                account="work",
                folder="INBOX",
                date_from=date(2026, 7, 6),
                date_to=date(2026, 7, 12),
                total_found=2,
                returned=2,
                truncated=False,
                emails=[
                    EmailHeader(uid="1", subject="紧急", sender="one@example.com"),
                    EmailHeader(uid="2", subject="资讯", sender="two@example.com"),
                ],
            )

        def get_content(self, _account, uid, **_kwargs):
            return EmailContent(
                uid=uid,
                subject="紧急" if uid == "1" else "资讯",
                sender=f"{uid}@example.com",
                body="不可信正文",
                content_type="text/plain",
                truncated=False,
            )

    class FakeOllama:
        def __init__(self):
            self.calls = []
            self.responses = [
                _decision_json(95, "账户安全", True),
                _decision_json(20, "新闻订阅", False),
            ]

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return NormalizedMessage(content=self.responses.pop(0), thinking="", tool_calls=[])

        def installed_models(self):
            return ["qwen3.5:4b", "test-model"]

    def _decision_json(score: int, theme: str, action: bool) -> str:
        return json.dumps(
            {
                "priority_score": score,
                "theme": theme,
                "summary": "测试摘要",
                "importance_reason": "测试原因",
                "action_required": action,
                "suggested_action": "立即处理" if action else None,
                "confidence": 0.9,
            },
            ensure_ascii=False,
        )

    ollama = FakeOllama()
    services = RuntimeServices(
        config_store=FakeConfigStore(),  # type: ignore[arg-type]
        secrets=object(),  # type: ignore[arg-type]
        mail=FakeMail(),  # type: ignore[arg-type]
        ollama=ollama,  # type: ignore[arg-type]
        session=SessionContext(active_account="work", model="test-model"),
    )
    events = []
    registry = build_tool_registry(
        services, events=lambda event, data: events.append((event, data))
    )

    result = registry.execute(
        "reviews.generate",
        {"date_from": "2026-07-06", "date_to": "2026-07-12", "limit": 10},
    )

    assert isinstance(result.output, dict)
    assert result.output["emails_analyzed"] == 2
    assert result.output["emails_failed"] == 0
    assert result.output["summary"]["priorities"]["P0"] == 1
    assert services.session.last_folder == "INBOX"
    assert services.session.last_review_uids == ["1", "2"]
    assert [item.uid for item in services.session.review_items()] == ["1", "2"]
    assert len(ollama.calls) == 2
    assert {call["model"] for call in ollama.calls} == {"qwen3.5:4b"}
    assert [data["current"] for event, data in events if event == "tool_progress"] == [1, 2]
