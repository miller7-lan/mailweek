from __future__ import annotations

from io import StringIO

from rich.console import Console

from mailweek.cli import _parse_review_index
from mailweek.render import (
    RichEventRenderer,
    emit_accounts,
    emit_agent_result,
    emit_email_detail,
    emit_error,
    emit_quick_presets,
    emit_review_registry,
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
    assert "直接描述需求" in output
    assert "/review" in output


def test_startup_banner_promotes_safety_and_quick_actions() -> None:
    buffer = StringIO()
    console = Console(file=buffer, width=100, color_system=None)

    emit_startup_banner("qwen3.5:9b", "work", console=console)

    output = buffer.getvalue()
    assert "READ ONLY" in output
    assert "QUICK START" in output
    assert "审查 /review" in output
    assert "帮助 /help" in output


def test_startup_banner_uses_compact_icon_in_narrow_terminal() -> None:
    buffer = StringIO()
    console = Console(file=buffer, width=50, color_system=None)

    emit_startup_banner("qwen3.5:9b", None, console=console)

    output = buffer.getvalue()
    assert "✉  MAILWEEK" in output
    assert "账户 未选择" in output
    assert "╭────────────╮" not in output
    assert "先添加邮箱账户" in output
    assert "/account add" in output


def test_quick_presets_explain_global_one_key_choices() -> None:
    buffer = StringIO()
    console = Console(file=buffer, width=80, color_system=None)

    emit_quick_presets(console=console)

    output = buffer.getvalue()
    assert "GLOBAL PRESETS" in output
    assert "[1]" in output and "上周完整审查" in output
    assert "[2]" in output and "今日重点" in output
    assert "[3]" in output and "自定义命令" in output
    assert "无需回车" in output
    assert "/quick" in output


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
    assert "建议先处理 #1" in output
    assert "直接输入 1" in output
    assert "邮箱出处：QQ邮箱 · work · user@qq.com" in output
    assert "项目等待发布确认" not in output
    assert "模型的长回答不应" not in output


def test_registry_uses_action_markers_and_compact_columns_on_narrow_terminals() -> None:
    buffer = StringIO()
    console = Console(file=buffer, width=72, color_system=None)

    emit_review_registry(
        [(1, _classification())],
        summary=None,
        total=1,
        console=console,
    )

    output = buffer.getvalue()
    assert "ACTION QUEUE" in output
    assert "READ ONLY" in output
    assert "● 待处理" in output
    assert "请确认项目发布时间" in output
    assert "发件人" not in output


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
    assert "下一步：" in output
    assert "/back 返回登记簿" in output
    assert "输入其他编号切换邮件" in output
    assert "/list P0" in output


def test_email_detail_promotes_action_and_read_only_hierarchy(capsys) -> None:
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
    )

    output = capsys.readouterr().out
    assert "ACTION REQUIRED" in output
    assert "建议动作" in output
    assert "READ ONLY" in output
    assert "操作导航" in output


def test_empty_review_filter_explains_how_to_recover() -> None:
    buffer = StringIO()
    console = Console(file=buffer, width=100, color_system=None)

    emit_review_registry(
        [],
        summary=None,
        total=6,
        filter_label="P1",
        console=console,
    )

    output = buffer.getvalue()
    assert "当前筛选没有结果" in output
    assert "/list 返回全部" in output
    assert "/list 待处理" in output


def test_account_list_explains_switch_and_add_commands(capsys) -> None:
    emit_accounts(
        [
            {
                "name": "work",
                "provider": "QQ邮箱",
                "email": "user@qq.com",
                "host": "imap.qq.com",
                "active": True,
            }
        ],
        json_mode=False,
    )

    output = capsys.readouterr().out
    assert "下一步：" in output
    assert "/account 名称" in output
    assert "/account add" in output


def test_account_list_uses_vault_title_and_active_status(capsys) -> None:
    emit_accounts(
        [
            {
                "name": "work",
                "provider": "QQ邮箱",
                "email": "user@qq.com",
                "host": "imap.qq.com",
                "active": True,
            }
        ],
        json_mode=False,
    )

    output = capsys.readouterr().out
    assert "ACCOUNT VAULT" in output
    assert "● ACTIVE" in output
    assert "NEXT / 下一步" in output


def test_human_error_labels_recovery_hint_as_next_step(capsys) -> None:
    emit_error(
        {
            "ok": False,
            "error": {
                "message": "还没有配置邮箱账户。",
                "hint": "输入 /account add 新增账户。",
            },
        },
        json_mode=False,
    )

    output = capsys.readouterr().err
    assert "错误：还没有配置邮箱账户" in output
    assert "下一步：输入 /account add" in output


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


def test_tool_selection_explains_the_operation_in_user_language(capsys) -> None:
    renderer = RichEventRenderer(json_mode=False)

    renderer(
        "tools_selected",
        {"tools": ["accounts.get_active", "accounts.list", "reviews.generate"]},
    )

    output = capsys.readouterr().out
    assert "准备审查邮件" in output
    assert "生成优先级登记簿" in output
    assert "只读" in output
    assert "accounts.get_active" not in output


def test_long_review_operation_explains_wait_and_cancel(capsys) -> None:
    renderer = RichEventRenderer(json_mode=False)

    renderer("tool_start", {"tool": "reviews.generate"})

    output = capsys.readouterr().out
    assert "正在读取并分类" in output
    assert "Ctrl+C" in output


def test_model_route_explains_accuracy_fallback(capsys) -> None:
    renderer = RichEventRenderer(json_mode=False)

    renderer("model_route", {"model": "qwen3.5:4b", "fallback": "qwen3.5:9b"})

    output = capsys.readouterr().out
    assert "qwen3.5:4b" in output
    assert "qwen3.5:9b" in output
    assert "自动复核" in output


def test_event_renderer_uses_consistent_run_state_badges(capsys) -> None:
    renderer = RichEventRenderer(json_mode=False)

    renderer("tools_selected", {"tools": ["reviews.generate"]})
    renderer("tool_start", {"tool": "reviews.generate"})
    renderer("model_route", {"model": "qwen3.5:4b", "fallback": "qwen3.5:9b"})
    renderer(
        "tool_finish",
        {"tool": "reviews.generate", "elapsed": 1.2, "summary": "emails_analyzed=3"},
    )

    output = capsys.readouterr().out
    assert "PLAN" in output
    assert "RUNNING" in output
    assert "MODEL ROUTE" in output
    assert "COMPLETE" in output


def test_review_index_parser_only_accepts_positive_numbers() -> None:
    assert _parse_review_index(" 3 ") == 3
    assert _parse_review_index("0") is None
    assert _parse_review_index("P0") is None
