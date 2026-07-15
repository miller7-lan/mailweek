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


def emit_startup_banner(
    model: str,
    account: str | None,
    *,
    console: Console | None = None,
) -> None:
    target = console or Console()
    account_label = account or "未选择"

    if target.width < 72:
        compact = Text()
        compact.append("✉  MAILWEEK", style="bold bright_cyan")
        compact.append("\n本地只读 · Ollama 邮件 Agent", style="white")
        compact.append("\n● ", style="green")
        compact.append(f"模型 {model}", style="dim")
        compact.append("  ·  ◆ ", style="magenta")
        compact.append(f"账户 {account_label}", style="dim")
        compact.append("\n/help 查看命令", style="dim")
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
    brand.append("MAILWEEK\n", style="bold bright_cyan")
    brand.append("本地 · 只读 · Ollama 邮件 Agent\n", style="white")
    brand.append("让重要邮件先被看见\n", style="dim italic")
    brand.append("● ", style="green")
    brand.append(f"模型 {model}", style="dim")
    brand.append("   ◆ ", style="magenta")
    brand.append(f"账户 {account_label}", style="dim")
    brand.append("\n/help 查看命令", style="dim")

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
            TextColumn("  [cyan]分类登记[/cyan]"),
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

    def __call__(self, event: str, data: dict[str, Any]) -> None:
        if event == "tools_selected":
            self.console.print(f"[dim]相关工具：{', '.join(data.get('tools', []))}[/dim]")
        elif event == "tool_start":
            self.close()
            self.console.print(f"\n[cyan]●[/cyan] {self._tool_label(data['tool'])}")
        elif event == "model_route":
            fallback = f" · 失败回退 {data['fallback']}" if data.get("fallback") else ""
            self.console.print(f"  [dim]分类模型 {data['model']}{fallback}[/dim]")
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
            self.console.print(f"  [green]✓[/green] {summary} · {elapsed}s")
        elif event in {"tool_error", "tool_cancelled"}:
            self.close()
            self.console.print(f"  [red]工具失败[/red] {data.get('tool')}: {data.get('message')}")


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
    text.append(f"本次会话已登记 {total} 封邮件", style="bold")
    if summary:
        for priority in ("P0", "P1", "P2", "P3", "P4"):
            text.append("  ·  ", style="dim")
            text.append(
                f"{priority} {summary.priorities.get(priority, 0)}",
                style=PRIORITY_STYLES[priority],
            )
    if filter_label:
        text.append(f"\n当前筛选：{filter_label}", style="dim")
    if account_source:
        text.append("\n邮箱出处：", style="dim")
        text.append(account_source, style="cyan")
    return text


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
            title="[bold cyan]邮件分类登记簿[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("编号", justify="center", width=4, no_wrap=True)
    table.add_column("优先", justify="center", width=6, no_wrap=True)
    table.add_column("分类", width=10, no_wrap=True)
    table.add_column("状态", width=8, no_wrap=True)
    table.add_column("主题", ratio=3, overflow="fold")
    table.add_column("发件人", ratio=2, overflow="ellipsis", no_wrap=True)
    header_map = headers or {}
    for index, item in entries:
        priority = item.priority.value
        action = Text(
            "待处理" if item.action_required else "仅查看",
            style="bold yellow" if item.action_required else "dim",
        )
        known_header = header_map.get(item.uid)
        sender = known_header.sender if known_header is not None else item.sender
        table.add_row(
            Text(str(index), style="bold cyan"),
            Text(priority, style=PRIORITY_STYLES[priority]),
            Text(item.theme.value),
            action,
            Text(item.subject),
            Text(sender),
        )
    if entries:
        target.print(table)
    else:
        target.print(Panel("没有匹配当前筛选条件的邮件。", border_style="dim"))
    target.print(
        "[dim]输入编号或 [bold]/open 编号[/bold] 查看 AI 建议与正文；"
        "[bold]/list[/bold] 返回全部；[bold]/list P0[/bold] 可筛选。[/dim]"
    )


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
    advice.append("优先级：", style="dim")
    advice.append(f"{priority} · {classification.priority_score}", style=PRIORITY_STYLES[priority])
    advice.append("\n分类：", style="dim")
    advice.append(classification.theme.value)
    advice.append("\n摘要：", style="dim")
    advice.append(classification.summary)
    advice.append("\n重要原因：", style="dim")
    advice.append(classification.importance_reason)
    advice.append("\nAI 建议：", style="dim")
    advice.append(
        classification.suggested_action
        or ("核实邮件内容并尽快处理。" if classification.action_required else "无需立即操作。")
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
            title=f"[bold]邮件 #{index}[/bold]",
            border_style=PRIORITY_STYLES[priority],
            padding=(1, 2),
        )
    )
    target.print(Panel(advice, title="[bold magenta]AI 判断与建议[/bold magenta]", padding=(1, 2)))
    target.print(
        Panel(
            body,
            title="[bold cyan]邮件正文（只读）[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    target.print("[dim]输入 [bold]/back[/bold] 返回登记簿，或直接输入其他编号。[/dim]")


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
    console.print(
        f"\n[dim]工具调用 {result.tool_calls} 次 · 分析 {result.emails_analyzed} 封"
        f" · 账户 {result.account or '未选择'}[/dim]"
    )


def emit_error(error: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        emit_json(error)
        return
    detail = error.get("error", {})
    console = Console(stderr=True)
    console.print(f"[red]错误：[/red]{detail.get('message', '未知错误')}")
    if hint := detail.get("hint"):
        console.print(f"[dim]{hint}[/dim]")


def emit_value(value: object, *, json_mode: bool) -> None:
    if json_mode:
        emit_json(value)
    else:
        Console().print(Pretty(value, expand_all=False))


def emit_accounts(accounts: list[dict[str, Any]], *, json_mode: bool) -> None:
    if json_mode:
        emit_json({"ok": True, "accounts": accounts})
        return
    table = Table(title="Mailweek 邮箱账户")
    table.add_column("名称")
    table.add_column("邮箱出处")
    table.add_column("邮箱")
    table.add_column("IMAP")
    table.add_column("默认")
    for account in accounts:
        table.add_row(
            str(account["name"]),
            str(account.get("provider", "未知")),
            str(account["email"]),
            str(account["host"]),
            "✓" if account.get("active") else "",
        )
    Console().print(table)


def emit_providers(providers: list[dict[str, Any]], *, json_mode: bool) -> None:
    if json_mode:
        emit_json({"ok": True, "providers": providers})
        return
    table = Table(title="Mailweek 支持的邮箱出处")
    table.add_column("ID", style="cyan")
    table.add_column("邮箱出处")
    table.add_column("默认 IMAP")
    for provider in providers:
        table.add_row(
            str(provider["id"]),
            str(provider["name"]),
            str(provider.get("imap_host") or "手动输入"),
        )
    Console().print(table)


def emit_tools(tools: list[dict[str, Any]], *, json_mode: bool) -> None:
    if json_mode:
        emit_json({"ok": True, "tools": tools})
        return
    table = Table(title="Mailweek Agent 工具")
    table.add_column("工具")
    table.add_column("权限")
    table.add_column("说明")
    for tool in tools:
        table.add_row(tool["name"], "只读", tool["description"])
    Console().print(table)
