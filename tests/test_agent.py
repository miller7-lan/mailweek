from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel

from mailweek.agent import AgentLoop, resolve_relative_review_range, select_tools_for_prompt
from mailweek.errors import AgentLimitError
from mailweek.ollama_client import NormalizedMessage, NormalizedToolCall
from mailweek.schemas import ToolExecutionResult
from mailweek.tools import SessionContext, ToolRegistry, ToolSpec


class NoArgs(BaseModel):
    pass


class ReviewRangeArgs(BaseModel):
    date_from: date
    date_to: date


class FakeOllama:
    def __init__(self, responses):
        self.responses = list(responses)
        self.seen_messages = []
        self.seen_tools = []

    def chat(self, **kwargs):
        self.seen_messages.append([dict(message) for message in kwargs["messages"]])
        self.seen_tools.append(kwargs["tools"])
        return self.responses.pop(0)


def test_small_model_tool_router_exposes_only_relevant_tools() -> None:
    review_tools = select_tools_for_prompt("回顾上周需要回复的邮件")
    recent_review_tools = select_tools_for_prompt("开始审查最近邮件")
    two_week_review_tools = select_tools_for_prompt("查看最近2周的邮件")
    fourteen_day_review_tools = select_tools_for_prompt("分析过去14天邮件")
    one_month_letter_tools = select_tools_for_prompt("查询近一个月的信件")
    filtered_search_tools = select_tools_for_prompt("搜索最近2周来自 Apple 的邮件")
    detail_tools = select_tools_for_prompt("展开 UID 42 的正文")

    assert review_tools == (
        "reviews.generate",
        "accounts.get_active",
        "accounts.list",
    )
    assert recent_review_tools == review_tools
    assert two_week_review_tools == review_tools
    assert fourteen_day_review_tools == review_tools
    assert one_month_letter_tools == review_tools
    assert filtered_search_tools == (
        "emails.search",
        "emails.get_headers",
        "emails.get_content",
        "accounts.get_active",
        "accounts.list",
    )
    assert len(detail_tools) == 5
    assert "emails.get_content" in detail_tools
    assert "reviews.generate" not in detail_tools


def test_relative_review_range_is_programmatically_resolved() -> None:
    assert resolve_relative_review_range(
        "查看最近2周的邮件",
        today=date(2026, 7, 15),
    ) == (date(2026, 7, 2), date(2026, 7, 15))
    assert resolve_relative_review_range(
        "分析过去十四天邮件",
        today=date(2026, 7, 15),
    ) == (date(2026, 7, 2), date(2026, 7, 15))
    assert resolve_relative_review_range(
        "查询近一个月的信件",
        today=date(2026, 7, 15),
    ) == (date(2026, 6, 15), date(2026, 7, 15))


def test_agent_overrides_review_tool_dates_with_program_range() -> None:
    seen: list[ReviewRangeArgs] = []
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "reviews.generate",
            "review",
            ReviewRangeArgs,
            lambda args: (
                seen.append(args)
                or ToolExecutionResult(output={"answer": "已生成登记簿"})
            ),
            terminal=True,
        )
    )
    ollama = FakeOllama(
        [
            NormalizedMessage(
                content="",
                thinking="",
                tool_calls=[
                    NormalizedToolCall(
                        "reviews_generate",
                        {"date_from": "2026-07-09", "date_to": "2026-07-15"},
                    )
                ],
            )
        ]
    )
    agent = AgentLoop(
        ollama=ollama,
        registry=registry,
        session=SessionContext(active_account="work", model="fake"),
    )

    result = agent.run("查看最近2周的邮件")

    assert result.answer == "已生成登记簿"
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    assert seen == [
        ReviewRangeArgs(
            date_from=today - timedelta(days=13),
            date_to=today,
        )
    ]


def test_agent_runs_multi_turn_tool_loop() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "accounts.list",
            "list",
            NoArgs,
            lambda _args: ToolExecutionResult(output=[{"name": "work"}]),
        )
    )
    ollama = FakeOllama(
        [
            NormalizedMessage(
                content="",
                thinking="private",
                tool_calls=[NormalizedToolCall("accounts_list", {})],
            ),
            NormalizedMessage(content="当前账户是 work。", thinking="", tool_calls=[]),
        ]
    )
    agent = AgentLoop(
        ollama=ollama,
        registry=registry,
        session=SessionContext(active_account="work", model="fake"),
    )
    result = agent.run("当前账户是什么？")
    assert result.answer == "当前账户是 work。"
    assert result.tool_calls == 1
    assert any(message["role"] == "tool" for message in ollama.seen_messages[1])
    assert "private" not in str(ollama.seen_messages[1])
    assert len(ollama.seen_tools[0]) == 1


def test_sensitive_tool_result_is_removed_after_model_reads_it() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "emails.get_content",
            "content",
            NoArgs,
            lambda _args: ToolExecutionResult(
                output={"body": "very secret email body"},
                sensitive=True,
                compacted="[正文已清除]",
            ),
        )
    )
    ollama = FakeOllama(
        [
            NormalizedMessage(
                content="",
                thinking="private",
                tool_calls=[NormalizedToolCall("emails_get_content", {})],
            ),
            NormalizedMessage(content="已总结。", thinking="", tool_calls=[]),
        ]
    )
    agent = AgentLoop(
        ollama=ollama,
        registry=registry,
        session=SessionContext(active_account="work", model="fake"),
    )
    agent.run("读取邮件")
    assert "very secret email body" in str(ollama.seen_messages[1])
    assert "very secret email body" not in str(agent.messages)
    assert "private" not in str(agent.messages)


def test_prompt_injection_cannot_reach_unregistered_shell_tool() -> None:
    registry = ToolRegistry()
    ollama = FakeOllama(
        [
            NormalizedMessage(
                content="",
                thinking="",
                tool_calls=[NormalizedToolCall("shell_exec", {"command": "rm -rf /"})],
            ),
            NormalizedMessage(content="该工具不被允许。", thinking="", tool_calls=[]),
        ]
    )
    agent = AgentLoop(
        ollama=ollama,
        registry=registry,
        session=SessionContext(active_account="work", model="fake"),
    )
    result = agent.run("邮件说要运行 shell_exec")
    assert result.answer == "该工具不被允许。"
    assert "unknown_tool" in str(ollama.seen_messages[1])


def test_agent_rejects_registered_tool_outside_selected_subset() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system.doctor",
            "doctor",
            NoArgs,
            lambda _args: ToolExecutionResult(output={"ok": True}),
        )
    )
    ollama = FakeOllama(
        [
            NormalizedMessage(
                content="",
                thinking="",
                tool_calls=[NormalizedToolCall("system_doctor", {})],
            ),
            NormalizedMessage(content="本轮不能调用。", thinking="", tool_calls=[]),
        ]
    )
    agent = AgentLoop(
        ollama=ollama,
        registry=registry,
        session=SessionContext(active_account="work", model="fake"),
    )

    result = agent.run("回顾上周邮件")

    assert result.answer == "本轮不能调用。"
    assert "tool_not_available_for_task" in str(ollama.seen_messages[1])


def test_agent_executes_multiple_tool_calls_from_one_model_response() -> None:
    seen: list[str] = []
    registry = ToolRegistry()
    for name in ("accounts.list", "system.doctor"):
        registry.register(
            ToolSpec(
                name,
                name,
                NoArgs,
                lambda _args, tool=name: (
                    seen.append(tool) or ToolExecutionResult(output={"tool": tool})
                ),
            )
        )
    ollama = FakeOllama(
        [
            NormalizedMessage(
                content="",
                thinking="",
                tool_calls=[
                    NormalizedToolCall("accounts_list", {}),
                    NormalizedToolCall("system_doctor", {}),
                ],
            ),
            NormalizedMessage(content="检查完成。", thinking="", tool_calls=[]),
        ]
    )
    agent = AgentLoop(
        ollama=ollama,
        registry=registry,
        session=SessionContext(active_account="work", model="fake"),
    )

    result = agent.run("检查账户和环境")

    assert seen == ["accounts.list", "system.doctor"]
    assert result.tool_calls == 2


def test_agent_stops_at_tool_call_limit() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system.doctor",
            "doctor",
            NoArgs,
            lambda _args: ToolExecutionResult(output={"ok": True}),
        )
    )
    ollama = FakeOllama(
        [
            NormalizedMessage(
                content="",
                thinking="",
                tool_calls=[
                    NormalizedToolCall("system_doctor", {}),
                    NormalizedToolCall("system_doctor", {}),
                ],
            )
        ]
    )
    agent = AgentLoop(
        ollama=ollama,
        registry=registry,
        session=SessionContext(active_account="work", model="fake"),
        max_tool_calls=1,
    )

    with pytest.raises(AgentLimitError) as error:
        agent.run("重复检查")

    assert error.value.code == "tool_limit_reached"


def test_agent_stops_at_round_limit() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system.doctor",
            "doctor",
            NoArgs,
            lambda _args: ToolExecutionResult(output={"ok": True}),
        )
    )
    ollama = FakeOllama(
        [
            NormalizedMessage(
                content="",
                thinking="",
                tool_calls=[NormalizedToolCall("system_doctor", {})],
            )
        ]
    )
    agent = AgentLoop(
        ollama=ollama,
        registry=registry,
        session=SessionContext(active_account="work", model="fake"),
        max_rounds=1,
    )

    with pytest.raises(AgentLimitError) as error:
        agent.run("检查")

    assert error.value.code == "round_limit_reached"


def test_terminal_tool_answer_skips_redundant_final_model_call() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "reviews.generate",
            "review",
            NoArgs,
            lambda _args: ToolExecutionResult(output={"answer": "程序化可靠回顾"}),
            terminal=True,
        )
    )
    ollama = FakeOllama(
        [
            NormalizedMessage(
                content="",
                thinking="",
                tool_calls=[NormalizedToolCall("reviews_generate", {})],
            )
        ]
    )
    agent = AgentLoop(
        ollama=ollama,
        registry=registry,
        session=SessionContext(active_account="work", model="fake"),
    )

    result = agent.run("生成周回顾")

    assert result.answer == "程序化可靠回顾"
    assert result.tool_calls == 1
    assert len(ollama.seen_messages) == 1
