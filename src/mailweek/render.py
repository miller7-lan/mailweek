from __future__ import annotations

import json
from typing import Any

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from .schemas import (
    AgentRunResult,
    EmailClassification,
    EmailContent,
    EmailHeader,
    ReviewSummary,
)

PRIORITY_STYLES = {
    "P0": "bold red",
    "P1": "bold yellow",
    "P2": "cyan",
    "P3": "blue",
    "P4": "dim",
}

PRIORITY_BADGE_STYLES = {
    "P0": "bold white on red3",
    "P1": "bold black on yellow3",
    "P2": "bold black on cyan",
    "P3": "bold white on blue",
    "P4": "bold white on grey35",
}


def emit_startup_banner(
    model: str,
    account: str | None,
    *,
    console: Console | None = None,
) -> None:
    target = console or Console()
    account_label = account or "未选择"
    if account:
        quick_start = "直接描述需求  ·  审查 /review  ·  帮助 /help"
    else:
        quick_start = "下一步：先添加邮箱账户 /account add  ·  帮助 /help"

    if target.width < 72:
        compact = Text()
        compact.append("✉  MAILWEEK", style="bold bright_cyan")
        compact.append("\n本地只读 · Ollama 邮件 Agent", style="white")
        compact.append("\n● ", style="green")
        compact.append(f"模型 {model}", style="dim")
        compact.append("  ·  ◆ ", style="magenta")
        compact.append(f"账户 {account_label}", style="dim")
        compact.append("\n READ ONLY ", style="bold white on dark_green")
        compact.append("  QUICK START", style="bold bright_cyan")
        if account:
            compact.append("\n审查 /review  ·  帮助 /help", style="white")
            compact.append("\n也可以直接描述需求", style="dim")
        else:
            compact.append("\n下一步：先添加邮箱账户 /account add", style="white")
            compact.append("\nHELP  /help", style="dim")
        target.print(
            Panel(
                compact,
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
        return

    icon = Text(
        "\n".join(
            (
                "╭────────────╮",
                "│ ╲        ╱ │",
                "│  ╲______╱  │",
                "│  ╱      ╲  │",
                "╰────────────╯",
            )
        ),
        style="bold bright_cyan",
    )
    brand = Text()
    brand.append("MAILWEEK", style="bold bright_cyan")
    brand.append("  /  INBOX PRIORITY AGENT\n", style="dim")
    brand.append("本地 · 只读 · Ollama 邮件 Agent\n", style="white")
    brand.append("让重要邮件先被看见\n", style="dim italic")
    brand.append(" READ ONLY ", style="bold white on dark_green")
    brand.append("  ● MODEL ", style="bold green")
    brand.append(model, style="white")
    brand.append("  ◆ ACCOUNT ", style="bold magenta")
    brand.append(f"{account_label}\n", style="white")
    brand.append("QUICK START  ", style="bold bright_cyan")
    brand.append(quick_start, style="white")

    layout = Table.grid(padding=(0, 3), expand=True)
    layout.add_column(width=14, no_wrap=True)
    layout.add_column(ratio=1)
    layout.add_row(icon, brand)
    target.print(
        Panel(
            layout,
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def emit_quick_presets(*, console: Console | None = None) -> None:
    target = console or Console()
    choices = Text()
    choices.append(" [1] ", style="bold black on bright_cyan")
    choices.append("上周完整审查", style="bold white")
    choices.append("  上一个完整自然周", style="dim")
    choices.append("\n [2] ", style="bold black on bright_cyan")
    choices.append("今日重点", style="bold white")
    choices.append("      精确审查今天的邮件", style="dim")
    choices.append("\n [3] ", style="bold black on bright_cyan")
    choices.append("自定义命令", style="bold white")
    choices.append("    输入 / 命令或自然语言", style="dim")
    choices.append("\n\n⌨  Mac 数字键可直接选择 · 无需回车", style="bright_cyan")
    choices.append("\n" if target.width < 60 else "  ·  ", style="dim")
    choices.append("/quick 随时打开", style="dim")
    target.print(
        Panel(
            choices,
            title=(
                "[bold bright_cyan]GLOBAL PRESETS[/bold bright_cyan] "
                "[dim]· 全局快捷[/dim]"
            ),
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


class RichEventRenderer:
    def __init__(self, *, json_mode: bool = False) -> None:
        self.json_mode = json_mode
        self.console = Console(stderr=json_mode)
        self._progress: Progress | None = None
        self._progress_task: TaskID | None = None

    def _start_progress(self, total: int) -> None:
        if self._progress is not None:
            return
        self._progress = Progress(
            TextColumn(
                " [bold black on cyan] CLASSIFY [/bold black on cyan]"
                " [cyan]分类登记[/cyan]"
            ),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        self._progress.start()
        self._progress_task = self._progress.add_task("分类登记", total=total)

    def close(self) -> None:
        if self._progress is not None:
            self._progress.stop()
        self._progress = None
        self._progress_task = None

    @staticmethod
    def _tool_label(tool: object) -> str:
        return {
            "reviews.generate": "生成邮件登记簿",
            "emails.get_content": "读取邮件正文",
        }.get(str(tool), str(tool))

    @staticmethod
    def _operation_hint(tools: object) -> str:
        selected = {str(tool) for tool in tools} if isinstance(tools, list) else set()
        if "reviews.generate" in selected:
            return "准备审查邮件：读取邮件并生成优先级登记簿（只读）"
        if "system.doctor" in selected:
            return "准备检查 Mailweek、邮箱和 Ollama 的运行状态（只读）"
        if "folders.list" in selected:
            return "准备查看邮箱文件夹（只读）"
        if "emails.get_content" in selected:
            return "准备查找并读取相关邮件内容（只读）"
        if "emails.search" in selected:
            return "准备按条件搜索邮件（只读）"
        if any(tool.startswith("accounts.") for tool in selected):
            return "准备查看或切换当前会话账户（不会修改邮件）"
        return "准备处理请求（邮件操作保持只读）"

    def __call__(self, event: str, data: dict[str, Any]) -> None:
        if event == "tools_selected":
            self.console.print(
                "[bold black on bright_cyan] PLAN [/bold black on bright_cyan]  "
                f"[dim]{self._operation_hint(data.get('tools', []))}[/dim]"
            )
        elif event == "tool_start":
            self.close()
            self.console.print(
                "\n[bold black on cyan] RUNNING [/bold black on cyan]  "
                f"[bold]{self._tool_label(data['tool'])}[/bold]"
            )
            if data.get("tool") == "reviews.generate":
                self.console.print("  [dim]正在读取并分类，请稍候；按 Ctrl+C 可取消[/dim]")
        elif event == "model_route":
            fallback = (
                f" · 结果不完整时由 {data['fallback']} 自动复核"
                if data.get("fallback")
                else ""
            )
            self.console.print(
                "  [bold white on magenta] MODEL ROUTE [/bold white on magenta]  "
                f"[dim]分类模型 {data['model']}{fallback}[/dim]"
            )
        elif event == "tool_progress":
            current = int(data.get("current", 0) or 0)
            total = int(data.get("total", 0) or 0)
            if self.console.is_terminal and not self.json_mode:
                self._start_progress(total)
                if self._progress is not None and self._progress_task is not None:
                    self._progress.update(self._progress_task, completed=current, total=total)
            elif current in {1, total} or (current > 0 and current % 10 == 0):
                cached = "（缓存）" if data.get("cached") else ""
                self.console.print(f"  [dim]分类登记 {current}/{total}{cached}[/dim]")
        elif event == "tool_finish":
            self.close()
            elapsed = data.get("elapsed", 0)
            summary = str(data.get("summary", "完成"))
            if summary.startswith("emails_analyzed="):
                summary = f"已登记 {summary.partition('=')[2]} 封邮件"
            self.console.print(
                "  [bold black on green] COMPLETE [/bold black on green]  "
                f"{summary} [dim]· {elapsed}s[/dim]"
            )
        elif event in {"tool_error", "tool_cancelled"}:
            self.close()
            self.console.print(
                "  [bold white on red] FAILED [/bold white on red]  "
                f"{data.get('tool')}: {data.get('message')}"
            )


def emit_json(value: object) -> None:
    Console().print_json(json.dumps(value, ensure_ascii=False, default=str))


def _overview_text(
    total: int,
    summary: ReviewSummary | None,
    *,
    filter_label: str | None = None,
    account_source: str | None = None,
) -> Text:
    text = Text()
    text.append(" READ ONLY ", style="bold white on dark_green")
    text.append(f"  {total:02d} MAILS", style="bold bright_cyan")
    text.append(f"  ·  本次会话已登记 {total} 封邮件", style="bold")
    if summary:
        text.append("\n")
        for priority in ("P0", "P1", "P2", "P3", "P4"):
            if priority != "P0":
                text.append("  ")
            text.append(
                f" {priority} {summary.priorities.get(priority, 0)} ",
                style=PRIORITY_BADGE_STYLES[priority],
            )
    if filter_label:
        text.append(f"\n当前筛选：{filter_label}", style="dim")
    if account_source:
        text.append("\n邮箱出处：", style="dim")
        text.append(account_source, style="cyan")
    return text


def _registry_guidance(
    entries: list[tuple[int, EmailClassification]],
    *,
    filter_label: str | None,
    compact: bool = False,
) -> Text:
    guidance = Text()
    if not entries:
        guidance.append(" NEXT / 下一步：", style="bold black on bright_cyan")
        guidance.append(" ")
        guidance.append("当前筛选没有结果；输入 ")
        guidance.append("/list", style="bold")
        guidance.append(" 返回全部，或换用 ")
        guidance.append("/list P0", style="bold")
        guidance.append("、")
        guidance.append("/list 待处理", style="bold")
        guidance.append("。")
        return guidance

    action_entry = next(
        ((index, item) for index, item in entries if item.action_required),
        None,
    )
    index, item = action_entry or entries[0]
    guidance.append(" NEXT / 下一步：", style="bold black on bright_cyan")
    guidance.append(" ")
    if action_entry:
        guidance.append(f"建议先处理 #{index}", style="bold")
        guidance.append(f"（{item.priority.value} · 待处理）")
    else:
        guidance.append(f"从 #{index} 开始查看", style="bold")
    guidance.append("\n" if compact else "，")
    if compact:
        guidance.append(" OPEN ", style="bold white on color(24)")
        guidance.append(" ")
    guidance.append(f"直接输入 {index}", style="bold cyan")
    guidance.append(" 查看 AI 建议与正文。\n")
    guidance.append(" COMMANDS / 更多操作：", style="bold white on color(24)")
    guidance.append(" ")
    guidance.append("输入编号或 /open 编号查看详情", style="dim")
    guidance.append("\n" if compact else " · ", style="dim")
    if compact:
        guidance.append(" FILTERS ", style="bold white on color(24)")
        guidance.append(" ")
    guidance.append("/list P0", style="bold")
    guidance.append(" 紧急" if compact else " 只看紧急", style="dim")
    guidance.append(" · ", style="dim")
    guidance.append("/list 待处理", style="bold")
    guidance.append(" 需操作" if compact else " 只看需操作", style="dim")
    if filter_label:
        guidance.append(" · ", style="dim")
        guidance.append("/list", style="bold")
        guidance.append(" 返回全部", style="dim")
    guidance.append("\n GLOBAL / 全局：", style="bold bright_cyan")
    guidance.append("/quick", style="bold")
    guidance.append(" 快捷预设", style="dim")
    return guidance


def emit_review_registry(
    entries: list[tuple[int, EmailClassification]],
    *,
    summary: ReviewSummary | None,
    headers: dict[str, EmailHeader] | None = None,
    total: int | None = None,
    filter_label: str | None = None,
    account_source: str | None = None,
    console: Console | None = None,
) -> None:
    target = console or Console()
    target.print()
    target.print(
        Panel(
            _overview_text(
                total if total is not None else len(entries),
                summary,
                filter_label=filter_label,
                account_source=account_source,
            ),
            title="[bold bright_cyan]ACTION QUEUE[/bold bright_cyan] [dim]· 邮件分类登记簿[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    compact = target.width < 96
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold bright_cyan",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("编号", justify="center", width=5, no_wrap=True)
    table.add_column("优先", justify="center", width=6, no_wrap=True)
    if not compact:
        table.add_column("分类", width=10, no_wrap=True)
    table.add_column("状态", width=10, no_wrap=True)
    table.add_column("主题", ratio=3, overflow="fold")
    if not compact:
        table.add_column("发件人", ratio=2, overflow="ellipsis", no_wrap=True)
    header_map = headers or {}
    for index, item in entries:
        priority = item.priority.value
        action = Text(
            "● 待处理" if item.action_required else "○ 仅查看",
            style="bold yellow" if item.action_required else "dim cyan",
        )
        known_header = header_map.get(item.uid)
        sender = known_header.sender if known_header is not None else item.sender
        row = [
            Text(f"#{index:02d}", style="bold cyan"),
            Text(f" {priority} ", style=PRIORITY_BADGE_STYLES[priority]),
        ]
        if not compact:
            row.append(Text(item.theme.value))
        row.extend((action, Text(item.subject)))
        if not compact:
            row.append(Text(sender))
        table.add_row(*row)
    if entries:
        target.print(table)
    else:
        target.print(Panel("没有匹配当前筛选条件的邮件。", border_style="dim"))
    target.print(_registry_guidance(entries, filter_label=filter_label, compact=compact))


def emit_email_detail(
    index: int,
    classification: EmailClassification,
    content: EmailContent,
    *,
    header: EmailHeader | None = None,
    account_source: str | None = None,
    console: Console | None = None,
) -> None:
    target = console or Console()
    priority = classification.priority.value
    received_at = content.received_at or (header.received_at if header else None)
    attachments = content.attachments or (header.attachments if header else [])

    metadata = Text()
    metadata.append(classification.subject, style="bold")
    metadata.append("\n发件人：", style="dim")
    metadata.append(content.sender)
    metadata.append("\n时间：", style="dim")
    metadata.append(received_at.astimezone().strftime("%Y-%m-%d %H:%M") if received_at else "未知")
    metadata.append("\n附件：", style="dim")
    if attachments:
        metadata.append(", ".join(item.filename or item.content_type for item in attachments))
    else:
        metadata.append("无")
    if account_source:
        metadata.append("\n邮箱出处：", style="dim")
        metadata.append(account_source, style="cyan")

    advice = Text()
    advice.append(
        " ACTION REQUIRED " if classification.action_required else " REVIEW ONLY ",
        style="bold black on yellow3" if classification.action_required else "bold white on blue",
    )
    advice.append("  ")
    advice.append(f" {priority} ", style=PRIORITY_BADGE_STYLES[priority])
    advice.append(f"  SCORE {classification.priority_score}", style="dim")
    advice.append("\n分类：", style="dim")
    advice.append(classification.theme.value)
    advice.append("\n摘要：", style="dim")
    advice.append(classification.summary)
    advice.append("\n重要原因：", style="dim")
    advice.append(classification.importance_reason)
    advice.append("\n\n建议动作\n", style="bold bright_cyan")
    advice.append(
        classification.suggested_action
        or ("核实邮件内容并尽快处理。" if classification.action_required else "无需立即操作。"),
        style="bold white" if classification.action_required else "white",
    )
    if classification.due_at:
        advice.append("\n截止时间：", style="dim")
        advice.append(classification.due_at.astimezone().strftime("%Y-%m-%d %H:%M"))
    advice.append("\n置信度：", style="dim")
    advice.append(f"{classification.confidence:.0%}")

    body = Text(content.body)
    if content.truncated:
        body.append("\n\n正文超过安全读取上限，当前仅显示前 6000 个字符。", style="bold yellow")

    target.print()
    target.print(
        Panel(
            metadata,
            title=f"[bold]MESSAGE #{index:02d}[/bold] [dim]· 邮件 #{index}[/dim]",
            border_style=PRIORITY_STYLES[priority],
            padding=(1, 2),
        )
    )
    target.print(
        Panel(
            advice,
            title="[bold magenta]AI 判断与建议[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        )
    )
    target.print(
        Panel(
            body,
            title=(
                "[bold green]READ ONLY[/bold green] [dim]·[/dim] "
                "[bold cyan]邮件正文（只读）[/bold cyan]"
            ),
            border_style="cyan",
            padding=(1, 2),
        )
    )
    detail_guidance = Text()
    detail_guidance.append(" NAVIGATION / 操作导航 ", style="bold black on bright_cyan")
    detail_guidance.append("  下一步：", style="bold bright_cyan")
    detail_guidance.append("/back", style="bold")
    detail_guidance.append(" 返回登记簿\n", style="dim")
    detail_guidance.append(" COMMANDS ", style="bold white on color(24)")
    detail_guidance.append("  输入其他编号切换邮件 · ", style="dim")
    detail_guidance.append("/list P0", style="bold")
    detail_guidance.append(" 只看紧急邮件", style="dim")
    detail_guidance.append("\n GLOBAL / 全局：", style="bold bright_cyan")
    detail_guidance.append("/quick", style="bold")
    detail_guidance.append(" 快捷预设", style="dim")
    target.print(detail_guidance)


def emit_agent_result(
    result: AgentRunResult,
    *,
    json_mode: bool,
    review_items: list[EmailClassification] | None = None,
    review_summary: ReviewSummary | None = None,
    headers: dict[str, EmailHeader] | None = None,
    account_source: str | None = None,
) -> None:
    if json_mode:
        emit_json(result.model_dump(mode="json"))
        return
    console = Console()
    if review_items:
        emit_review_registry(
            list(enumerate(review_items, start=1)),
            summary=review_summary,
            headers=headers,
            total=len(review_items),
            account_source=account_source,
            console=console,
        )
    else:
        console.print()
        console.print(Markdown(result.answer))
    session = Text("\n")
    session.append(" SESSION ", style="bold white on color(24)")
    session.append(
        f"  工具调用 {result.tool_calls} 次 · 分析 {result.emails_analyzed} 封"
        f" · 账户 {result.account or '未选择'}",
        style="dim",
    )
    console.print(session)


def emit_error(error: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        emit_json(error)
        return
    detail = error.get("error", {})
    console = Console(stderr=True)
    message = Text()
    message.append(" ERROR ", style="bold white on red")
    message.append("  错误：", style="bold red")
    message.append(str(detail.get("message", "未知错误")))
    console.print(message)
    recovery = Text()
    recovery.append(" RECOVERY ", style="bold black on yellow3")
    recovery.append("  ")
    if hint := detail.get("hint"):
        recovery.append(f"下一步：{hint}", style="dim")
    else:
        recovery.append("下一步：输入 /help 查看可用操作。", style="dim")
    console.print(recovery)


def emit_value(value: object, *, json_mode: bool) -> None:
    if json_mode:
        emit_json(value)
    else:
        Console().print(Pretty(value, expand_all=False))


def _data_table(title: str) -> Table:
    return Table(
        title=title,
        title_style="bold bright_cyan",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold bright_cyan",
        show_lines=False,
        expand=True,
    )


def _next_bar() -> Text:
    bar = Text()
    bar.append(" NEXT / 下一步：", style="bold black on bright_cyan")
    bar.append(" ")
    return bar


def emit_accounts(accounts: list[dict[str, Any]], *, json_mode: bool) -> None:
    if json_mode:
        emit_json({"ok": True, "accounts": accounts})
        return
    table = _data_table("ACCOUNT VAULT · Mailweek 邮箱账户")
    table.add_column("名称", style="bold cyan")
    table.add_column("邮箱出处")
    table.add_column("邮箱")
    table.add_column("IMAP", style="dim")
    table.add_column("状态", justify="center", no_wrap=True)
    for account in accounts:
        table.add_row(
            str(account["name"]),
            str(account.get("provider", "未知")),
            str(account["email"]),
            str(account["host"]),
            Text(
                "● ACTIVE" if account.get("active") else "○ STANDBY",
                style="bold green" if account.get("active") else "dim",
            ),
        )
    console = Console()
    console.print(table)
    guidance = _next_bar()
    if accounts:
        guidance.append("/account 名称", style="bold")
        guidance.append(" 切换会话账户 · ", style="dim")
        guidance.append("/account add", style="bold")
        guidance.append(" 新增账户 · ", style="dim")
        guidance.append("/review", style="bold")
        guidance.append(" 开始审查", style="dim")
    else:
        guidance.append("还没有邮箱账户，请输入 ")
        guidance.append("/account add", style="bold")
        guidance.append("。")
    console.print(guidance)


def emit_providers(providers: list[dict[str, Any]], *, json_mode: bool) -> None:
    if json_mode:
        emit_json({"ok": True, "providers": providers})
        return
    table = _data_table("PROVIDER CATALOG · Mailweek 支持的邮箱出处")
    table.add_column("ID", style="cyan")
    table.add_column("邮箱出处")
    table.add_column("默认 IMAP")
    for provider in providers:
        table.add_row(
            str(provider["id"]),
            str(provider["name"]),
            str(provider.get("imap_host") or "手动输入"),
        )
    console = Console()
    console.print(table)
    guidance = _next_bar()
    guidance.append("输入 ")
    guidance.append("/account add", style="bold")
    guidance.append("，Mailweek 会自动识别邮箱出处并预填 IMAP。")
    console.print(guidance)


def emit_tools(tools: list[dict[str, Any]], *, json_mode: bool) -> None:
    if json_mode:
        emit_json({"ok": True, "tools": tools})
        return
    table = _data_table("READ ONLY TOOLKIT · Mailweek Agent 工具")
    table.add_column("工具", style="bold cyan")
    table.add_column("权限")
    table.add_column("说明")
    for tool in tools:
        table.add_row(tool["name"], Text("● READ", style="bold green"), tool["description"])
    console = Console()
    console.print(table)
    console.print(
        "[dim]这些工具都不会发送、删除、移动或标记邮件；"
        "直接描述需求即可由 Mailweek 选择合适工具。[/dim]"
    )
