from __future__ import annotations

from io import StringIO

from rich.console import Console

from mailweek.cli import _parse_review_index
from mailweek.render import (
    RichEventRenderer,
    emit_agent_result,
    emit_email_detail,
    emit_startup_banner,
)
from mailweek.schemas import (
    AgentRunResult,
    EmailClassification,
    EmailContent,
    Priority,
    ReviewSummary,
    Theme,
)


def _classification() -> EmailClassification:
    return EmailClassification(
        uid="42",
        subject="请确认项目发布时间",
        sender="owner@example.com",
        priority_score=82,
        theme=Theme.WORK_PROJECT,
        summary="项目等待发布确认",
        importance_reason="本周五前需要回复",
        action_required=True,
        suggested_action="回复负责人并确认发布时间",
        confidence=0.94,
    )


def test_startup_banner_renders_wide_mail_icon_and_status() -> None:
    buffer = StringIO()
    console = Console(file=buffer, width=100, color_system=None)

    emit_startup_banner("qwen3.5:9b", "work", console=console)

    output = buffer.getvalue()
    assert "╭────────────╮" in output
    assert "MAILWEEK" in output
    assert "本地 · 只读 · Ollama 邮件 Agent" in output
    assert "qwen3.5:9b" in output
    assert "work" in output


def test_startup_banner_uses_compact_icon_in_narrow_terminal() -> None:
    buffer = StringIO()
    console = Console(file=buffer, width=50, color_system=None)

    emit_startup_banner("qwen3.5:9b", None, console=console)

    output = buffer.getvalue()
    assert "✉  MAILWEEK" in output
    assert "账户 未选择" in output
    assert "╭────────────╮" not in output


def test_agent_review_renders_registry_before_advice(capsys) -> None:
    item = _classification()
    summary = ReviewSummary(
        overview="共分类一封邮件",
        priorities={priority.value: int(priority is Priority.P1) for priority in Priority},
    )

    emit_agent_result(
        AgentRunResult(
            answer="模型的长回答不应在登记簿首页展开",
            account="work",
            emails_analyzed=1,
        ),
        json_mode=False,
        review_items=[item],
        review_summary=summary,
        account_source="QQ邮箱 · work · user@qq.com",
    )

    output = capsys.readouterr().out
    assert "邮件分类登记簿" in output
    assert "请确认项目发布时间" in output
    assert "待处理" in output
    assert "输入编号" in output
    assert "邮箱出处：QQ邮箱 · work · user@qq.com" in output
    assert "项目等待发布确认" not in output
    assert "模型的长回答不应" not in output


def test_email_detail_separates_ai_advice_and_read_only_body(capsys) -> None:
    emit_email_detail(
        1,
        _classification(),
        EmailContent(
            uid="42",
            subject="请确认项目发布时间",
            sender="owner@example.com",
            body="这是邮件的具体正文。",
            content_type="text/plain",
            truncated=False,
        ),
        account_source="QQ邮箱 · work · user@qq.com",
    )

    output = capsys.readouterr().out
    assert "AI 判断与建议" in output
    assert "回复负责人并确认发布时间" in output
    assert "邮件正文（只读）" in output
    assert "这是邮件的具体正文" in output
    assert "邮箱出处：QQ邮箱 · work · user@qq.com" in output
    assert "/back" in output


def test_non_terminal_progress_only_prints_milestones(capsys) -> None:
    renderer = RichEventRenderer(json_mode=False)
    renderer("tool_start", {"tool": "reviews.generate"})
    for current in (1, 2, 3):
        renderer(
            "tool_progress",
            {"tool": "reviews.generate", "current": current, "total": 3},
        )
    renderer(
        "tool_finish",
        {"tool": "reviews.generate", "elapsed": 1.2, "summary": "emails_analyzed=3"},
    )

    output = capsys.readouterr().out
    assert "分类登记 1/3" in output
    assert "分类登记 2/3" not in output
    assert "分类登记 3/3" in output
    assert "已登记 3 封邮件" in output


def test_review_index_parser_only_accepts_positive_numbers() -> None:
    assert _parse_review_index(" 3 ") == 3
    assert _parse_review_index("0") is None
    assert _parse_review_index("P0") is None
