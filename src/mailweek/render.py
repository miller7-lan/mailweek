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
    if account:
        next_step = "下一步：直接描述需求，或输入 /review 开始每周审查"
    else:
        next_step = "下一步：先添加邮箱账户：/account add"

    if target.width < 72:
        compact = Text()
        compact.append("✉  MAILWEEK", style="bold bright_cyan")
        compact.append("\n本地只读 · Ollama 邮件 Agent", style="white")
        compact.append("\n● ", style="green")
        compact.append(f"模型 {model}", style="dim")
        compact.append("  ·  ◆ ", style="magenta")
        compact.append(f"账户 {account_label}", style="dim")
        compact.append(f"\n{next_step}", style="bright_cyan")
        compact.append("\n/help 查看全部命令", style="dim")
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
    brand.append(f"\n{next_step}", style="bright_cyan")
    brand.append("\n/help 查看全部命令", style="dim")

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
            self.console.print(f"[dim]{self._operation_hint(data.get('tools', []))}[/dim]")
        elif event == "tool_start":
            self.close()
            self.console.print(f"\n[cyan]●[/cyan] {self._tool_label(data['tool'])}")
            if data.get("tool") == "reviews.generate":
                self.console.print("  [dim]正在读取并分类，请稍候；按 Ctrl+C 可取消[/dim]")
        elif event == "model_route":
            fallback = (
                f" · 结果不完整时由 {data['fallback']} 自动复核"
                if data.get("fallback")
                else ""
            )
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


def _registry_guidance(
    entries: list[tuple[int, EmailClassification]],
    *,
    filter_label: str | None,
) -> Text:
    guidance = Text()
    if not entries:
        guidance.append("下一步：", style="bold bright_cyan")
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
    guidance.append("下一步：", style="bold bright_cyan")
    if action_entry:
        guidance.append(f"建议先处理 #{index}", style="bold")
        guidance.append(f"（{item.priority.value} · 待处理），")
    else:
        guidance.append(f"从 #{index} 开始查看，", style="bold")
    guidance.append(f"直接输入 {index}", style="bold cyan")
    guidance.append(" 查看 AI 建议与正文。\n")
    guidance.append("更多操作：", style="dim")
    guidance.append("输入编号或 /open 编号查看详情", style="dim")
    guidance.append(" · ", style="dim")
    guidance.append("/list P0", style="bold")
    guidance.append(" 只看紧急", style="dim")
    guidance.append(" · ", style="dim")
    guidance.append("/list 待处理", style="bold")
    guidance.append(" 只看需操作", style="dim")
    if filter_label:
        guidance.append(" · ", style="dim")
        guidance.append("/list", style="bold")
        guidance.append(" 返回全部", style="dim")
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
    target.print(_registry_guidance(entries, filter_label=filter_label))


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
    detail_guidance = Text()
    detail_guidance.append("下一步：", style="bold bright_cyan")
    detail_guidance.append("/back", style="bold")
    detail_guidance.append(" 返回登记簿 · 输入其他编号切换邮件 · ", style="dim")
    detail_guidance.append("/list P0", style="bold")
    detail_guidance.append(" 只看紧急邮件", style="dim")
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
        console.print(f"[dim]下一步：{hint}[/dim]")
    else:
        console.print("[dim]下一步：输入 /help 查看可用操作。[/dim]")


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
    console = Console()
    console.print(table)
    if accounts:
        console.print(
            "[bold bright_cyan]下一步：[/bold bright_cyan]"
            "[bold]/account 名称[/bold] 切换会话账户 · "
            "[bold]/account add[/bold] 新增账户 · "
            "[bold]/review[/bold] 开始审查"
        )
    else:
        console.print(
            "[bold bright_cyan]下一步：[/bold bright_cyan]"
            "还没有邮箱账户，请输入 [bold]/account add[/bold]。"
        )


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
    console = Console()
    console.print(table)
    console.print(
        "[bold bright_cyan]下一步：[/bold bright_cyan]"
        "输入 [bold]/account add[/bold]，Mailweek 会自动识别邮箱出处并预填 IMAP。"
    )


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
    console = Console()
    console.print(table)
    console.print(
        "[dim]这些工具都不会发送、删除、移动或标记邮件；"
        "直接描述需求即可由 Mailweek 选择合适工具。[/dim]"
    )
