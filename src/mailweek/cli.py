from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import partial
from typing import Annotated, Any
from zoneinfo import ZoneInfo

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress
from typer._click.exceptions import ClickException

from .agent import AgentLoop
from .analysis import ClassificationSettings
from .benchmark import run_classifier_benchmark
from .diagnostics import run_doctor
from .errors import MailweekError
from .providers import (
    MailProvider,
    provider_catalog,
    provider_id,
    provider_label,
    resolve_provider,
    suggested_imap_host,
)
from .render import (
    RichEventRenderer,
    emit_accounts,
    emit_agent_result,
    emit_email_detail,
    emit_error,
    emit_providers,
    emit_quick_presets,
    emit_review_registry,
    emit_startup_banner,
    emit_tools,
    emit_value,
)
from .runtime import Runtime
from .schemas import AccountConfig, AgentRunResult, EmailContent
from .tools import ApprovalGate, build_tool_registry

app = typer.Typer(
    name="mailweek",
    help="本地 Ollama 驱动的只读邮件 CLI Agent。",
    no_args_is_help=False,
    invoke_without_command=True,
    pretty_exceptions_enable=False,
)
accounts_app = typer.Typer(help="配置和检查 IMAP 邮箱账户。")
models_app = typer.Typer(help="管理本地 Ollama 模型。")
tools_app = typer.Typer(help="发现 Agent 注册的安全工具。")
tool_app = typer.Typer(help="直接调用一个注册工具（只读逃生口）。")
emails_app = typer.Typer(help="执行确定性的只读邮件查询。")
app.add_typer(accounts_app, name="accounts")
app.add_typer(models_app, name="models")
app.add_typer(tools_app, name="tools")
app.add_typer(tool_app, name="tool")
app.add_typer(emails_app, name="emails")


@dataclass
class CLIState:
    json_mode: bool
    runtime: Runtime


def _state(ctx: typer.Context) -> CLIState:
    return ctx.ensure_object(CLIState)


def _confirmation(state: CLIState) -> Callable[[str], bool] | None:
    if state.json_mode or not sys.stdin.isatty():
        return None
    return lambda prompt: typer.confirm(prompt, default=False)


def _registry(state: CLIState):
    events = RichEventRenderer(json_mode=state.json_mode)
    return build_tool_registry(
        state.runtime.services,
        approvals=ApprovalGate(_confirmation(state)),
        events=events,
    )


def _agent(state: CLIState) -> AgentLoop:
    events = RichEventRenderer(json_mode=state.json_mode)
    registry = build_tool_registry(
        state.runtime.services,
        approvals=ApprovalGate(_confirmation(state)),
        events=events,
    )
    return AgentLoop(
        ollama=state.runtime.services.ollama,
        registry=registry,
        session=state.runtime.services.session,
        events=events,
    )


def _account_origin_label(
    state: CLIState,
    name: str | None,
    *,
    include_email: bool = True,
) -> str | None:
    if not name:
        return None
    config_store = getattr(state.runtime.services, "config_store", None)
    if config_store is None:
        return name
    account = config_store.load().accounts.get(name)
    if account is None:
        return name
    source = provider_label(
        account.provider,
        email=account.email,
        host=account.host,
    )
    parts = (source, name, account.email) if include_email else (source, name)
    return " · ".join(parts)


def _emit_agent_result_for_state(
    state: CLIState,
    result: AgentRunResult,
    *,
    show_registry: bool,
) -> None:
    session = state.runtime.services.session
    emit_agent_result(
        result,
        json_mode=state.json_mode,
        review_items=session.review_items() if show_registry else None,
        review_summary=session.last_summary if show_registry else None,
        headers=session.headers if show_registry else None,
        account_source=_account_origin_label(state, result.account),
    )


def _run_agent_and_render(
    state: CLIState,
    agent: AgentLoop,
    prompt: str,
    *,
    force_registry: bool = False,
) -> AgentRunResult:
    previous_order = tuple(state.runtime.services.session.last_review_uids)
    result = agent.run(prompt)
    current_order = tuple(state.runtime.services.session.last_review_uids)
    show_registry = bool(current_order) and (
        force_registry or current_order != previous_order or result.emails_analyzed > 0
    )
    _emit_agent_result_for_state(state, result, show_registry=show_registry)
    return result


def _handle(state: CLIState, action: Callable[[], Any]) -> Any:
    try:
        return action()
    except MailweekError as exc:
        emit_error(exc.as_dict(), json_mode=state.json_mode)
        raise typer.Exit(code=1) from exc


@app.callback()
def root(
    ctx: typer.Context,
    json_mode: Annotated[bool, typer.Option("--json", help="仅向 stdout 输出稳定 JSON。")] = False,
) -> None:
    """启动交互 Agent，或运行一个可组合子命令。"""
    state = CLIState(json_mode=json_mode, runtime=Runtime.create())
    ctx.obj = state
    if ctx.invoked_subcommand is None:
        if json_mode:
            emit_error(
                {
                    "ok": False,
                    "error": {
                        "code": "interactive_json_unsupported",
                        "message": "交互模式不支持 --json。",
                        "hint": '使用 mailweek --json ask "任务"。',
                    },
                },
                json_mode=True,
            )
            raise typer.Exit(code=2)
        run_repl(state)


@app.command()
def ask(
    ctx: typer.Context,
    prompt: Annotated[str, typer.Argument(help="交给 Agent 的任务")],
) -> None:
    """让 Agent 完成一个单轮只读邮件任务。"""
    state = _state(ctx)
    agent = _agent(state)
    _handle(state, lambda: _run_agent_and_render(state, agent, prompt))


def _last_week() -> tuple[date, date]:
    today = _today()
    this_monday = today - timedelta(days=today.weekday())
    return this_monday - timedelta(days=7), this_monday - timedelta(days=1)


def _today() -> date:
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


def _parse_date(value: str, option: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{option} 必须是 YYYY-MM-DD") from exc


@app.command()
def review(
    ctx: typer.Context,
    last_week: Annotated[bool, typer.Option("--last-week", help="回顾上一个完整自然周")] = False,
    date_from: Annotated[str | None, typer.Option("--from", help="开始日期 YYYY-MM-DD")] = None,
    date_to: Annotated[str | None, typer.Option("--to", help="结束日期 YYYY-MM-DD")] = None,
    limit: Annotated[int, typer.Option(min=1, max=500)] = 100,
) -> None:
    """用明确日期生成邮件回顾。"""
    state = _state(ctx)
    if last_week or (date_from is None and date_to is None):
        start, end = _last_week()
    elif date_from is not None and date_to is not None:
        start = _parse_date(date_from, "--from")
        end = _parse_date(date_to, "--to")
    else:
        raise typer.BadParameter("--from 和 --to 必须同时提供")
    prompt = (
        f"回顾 {start.isoformat()} 到 {end.isoformat()} 的邮件，最多 {limit} 封。"
        "按重要性排序，优先告诉我需要回复、存在风险或有截止日期的事项。"
    )
    agent = _agent(state)
    _handle(
        state,
        lambda: _run_agent_and_render(state, agent, prompt, force_registry=True),
    )


@app.command()
def doctor(ctx: typer.Context) -> None:
    """检查配置、钥匙串、IMAP 准备状态和 Ollama。"""
    state = _state(ctx)
    result = _handle(
        state,
        lambda: run_doctor(
            state.runtime.services.config_store,
            state.runtime.services.secrets,
            state.runtime.services.ollama,
        ),
    )
    emit_value(result, json_mode=state.json_mode)


@app.command("init")
def init_command(ctx: typer.Context) -> None:
    """显示首次配置状态和安全的下一步。"""
    state = _state(ctx)
    result = _handle(
        state,
        lambda: run_doctor(
            state.runtime.services.config_store,
            state.runtime.services.secrets,
            state.runtime.services.ollama,
        ),
    )
    emit_value(result, json_mode=state.json_mode)
    if not state.json_mode and result["missing_steps"]:
        Console().print("\n[bold]下一步[/bold]")
        for step in result["missing_steps"]:
            Console().print(f"- {step}")


@tools_app.command("list")
def tools_list(ctx: typer.Context) -> None:
    """列出 Agent 可以调用的全部工具。"""
    state = _state(ctx)
    items = [
        {"name": spec.name, "wire_name": spec.wire_name, "description": spec.description}
        for spec in _registry(state).specs()
    ]
    emit_tools(items, json_mode=state.json_mode)


@tools_app.command("describe")
def tools_describe(ctx: typer.Context, name: str) -> None:
    """查看一个工具的参数 Schema。"""
    state = _state(ctx)
    spec = _handle(state, lambda: _registry(state).resolve(name))
    emit_value(
        {
            "ok": True,
            "name": spec.name,
            "wire_name": spec.wire_name,
            "read_only": spec.read_only,
            "description": spec.description,
            "parameters": spec.args_model.model_json_schema(),
        },
        json_mode=state.json_mode,
    )


@tool_app.command("call")
def tool_call(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="例如 emails.search")],
    args_json: Annotated[str, typer.Option("--args-json", help="工具参数 JSON 对象")] = "{}",
) -> None:
    """直接调用注册工具；不会绕过白名单和参数校验。"""
    state = _state(ctx)
    try:
        arguments = json.loads(args_json)
        if not isinstance(arguments, dict):
            raise ValueError("args-json must be an object")
    except (json.JSONDecodeError, ValueError) as exc:
        raise typer.BadParameter(f"--args-json 不是有效 JSON 对象：{exc}") from exc
    result = _handle(state, lambda: _registry(state).execute(name, arguments))
    emit_value({"ok": True, "result": result.output}, json_mode=state.json_mode)


def _account_rows(state: CLIState) -> list[dict[str, Any]]:
    config = state.runtime.services.config_store.load()
    return [
        {
            "name": name,
            "email": account.email,
            "provider_id": provider_id(
                account.provider,
                email=account.email,
                host=account.host,
            ),
            "provider": provider_label(
                account.provider,
                email=account.email,
                host=account.host,
            ),
            "host": account.host,
            "folder": account.folder,
            "active": name == config.active_account,
        }
        for name, account in config.accounts.items()
    ]


@accounts_app.command("list")
def accounts_list(ctx: typer.Context) -> None:
    """列出账户，不显示任何秘密。"""
    state = _state(ctx)
    emit_accounts(_account_rows(state), json_mode=state.json_mode)


@accounts_app.command("providers")
def accounts_providers(ctx: typer.Context) -> None:
    """列出可自动配置的邮箱出处和 IMAP 主机。"""
    emit_providers(provider_catalog(), json_mode=_state(ctx).json_mode)


def _resolve_account_provider(
    requested: str | None,
    *,
    email: str,
    host: str | None,
) -> MailProvider:
    try:
        return resolve_provider(requested, email=email, host=host)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--provider") from exc


def _account_payload(account: AccountConfig) -> dict[str, Any]:
    source = provider_label(
        account.provider,
        email=account.email,
        host=account.host,
    )
    return {
        "ok": True,
        "account": account.name,
        "email": account.email,
        "provider_id": account.provider,
        "provider": source,
        "host": account.host,
        "active": True,
    }


def _persist_account(state: CLIState, account: AccountConfig, password: str) -> None:
    _handle(state, lambda: state.runtime.services.secrets.set(account, password))
    try:
        state.runtime.services.config_store.add_account(account, make_active=True)
    except Exception:
        state.runtime.services.secrets.delete(account)
        raise
    state.runtime.services.session.reset_analysis()
    state.runtime.services.session.active_account = account.name


@accounts_app.command("add")
def accounts_add(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument(help="账户短名称，如 work")] = None,
    email: Annotated[str | None, typer.Option("--email")] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="邮箱出处 ID；运行 accounts providers 查看"),
    ] = None,
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int, typer.Option(min=1, max=65535)] = 993,
    username: Annotated[str | None, typer.Option("--username")] = None,
    folder: Annotated[str, typer.Option("--folder")] = "INBOX",
    password_stdin: Annotated[bool, typer.Option("--password-stdin")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="确认写入账户和钥匙串")] = False,
) -> None:
    """新增 IMAP 账户并把应用专用密码保存到 macOS 钥匙串。"""
    state = _state(ctx)
    if state.json_mode:
        if not all((name, email, password_stdin, yes)):
            raise typer.BadParameter("JSON 模式必须提供 NAME、--email、--password-stdin 和 --yes")
        assert name is not None and email is not None
        selected_provider = _resolve_account_provider(
            provider,
            email=email,
            host=host,
        )
        host = host or suggested_imap_host(email, selected_provider)
        username = username or email
    else:
        name = name or typer.prompt("账户名称", default="work")
        email = email or typer.prompt("邮箱地址")
        detected_provider = resolve_provider("auto", email=email, host=host)
        provider = provider or typer.prompt(
            "邮箱出处",
            default=detected_provider.key,
        )
        selected_provider = _resolve_account_provider(
            provider,
            email=email,
            host=host,
        )
        host = host or typer.prompt(
            "IMAP 主机",
            default=suggested_imap_host(email, selected_provider),
        )
        username = username or typer.prompt("IMAP 用户名", default=email)
    if password_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = typer.prompt("IMAP 应用专用密码", hide_input=True)
    account = AccountConfig(
        name=name,
        email=email,
        provider=selected_provider.key,
        host=host,
        port=port,
        username=username,
        folder=folder,
    )
    if not yes and not typer.confirm(
        f"将 {selected_provider.label} 账户 {name} 写入全局配置，"
        "并把密码保存到 macOS 钥匙串？",
        default=False,
    ):
        raise typer.Abort()
    config = state.runtime.services.config_store.load()
    if (
        name in config.accounts
        and not yes
        and not typer.confirm(f"账户 {name} 已存在，确认覆盖？", default=False)
    ):
        raise typer.Abort()
    try:
        _persist_account(state, account, password)
    finally:
        password = ""
    emit_value(_account_payload(account), json_mode=state.json_mode)


def _suggest_account_name(state: CLIState) -> str:
    existing = state.runtime.services.config_store.load().accounts
    for candidate in ("personal", "work", "school", "mail"):
        if candidate not in existing:
            return candidate
    index = 2
    while f"mail{index}" in existing:
        index += 1
    return f"mail{index}"


def _interactive_add_account(state: CLIState) -> bool:
    console = Console()
    password = ""
    try:
        console.print(
            "[dim]邮箱出处：qq / gmail / outlook / icloud / 163 / 126 / custom[/dim]"
        )
        name = typer.prompt("账户名称", default=_suggest_account_name(state))
        email = typer.prompt("邮箱地址")
        detected = resolve_provider("auto", email=email)
        provider_key = typer.prompt("邮箱出处", default=detected.key)
        selected_provider = _resolve_account_provider(
            provider_key,
            email=email,
            host=None,
        )
        host = typer.prompt(
            "IMAP 主机",
            default=suggested_imap_host(email, selected_provider),
        )
        username = typer.prompt("IMAP 用户名", default=email)
        password = typer.prompt("IMAP 应用专用密码", hide_input=True)
        account = AccountConfig(
            name=name,
            email=email,
            provider=selected_provider.key,
            host=host,
            port=993,
            username=username,
            folder="INBOX",
        )
        config = state.runtime.services.config_store.load()
        if name in config.accounts and not typer.confirm(
            f"账户 {name} 已存在，确认覆盖？",
            default=False,
        ):
            console.print("[dim]已取消新增邮箱。[/dim]")
            return False
        if not typer.confirm(
            f"新增 {selected_provider.label} · {name}，并把密码保存到 macOS 钥匙串？",
            default=False,
        ):
            console.print("[dim]已取消新增邮箱。[/dim]")
            return False
        _persist_account(state, account, password)
        emit_value(_account_payload(account), json_mode=False)
        console.print("[dim]已切换到新账户；可输入 /review 开始只读分类。[/dim]")
        return True
    except (typer.Abort, EOFError, KeyboardInterrupt):
        console.print("[dim]已取消新增邮箱。[/dim]")
        return False
    finally:
        password = ""


@accounts_app.command("use")
def accounts_use(ctx: typer.Context, name: str, yes: bool = typer.Option(False, "--yes")) -> None:
    """修改全局默认账户；需要确认。"""
    state = _state(ctx)
    if not yes and (
        state.json_mode or not typer.confirm(f"把 {name} 设为全局默认账户？", default=False)
    ):
        raise typer.Abort()
    _handle(state, lambda: state.runtime.services.config_store.set_active_account(name))
    state.runtime.services.session.active_account = name
    emit_value({"ok": True, "active_account": name}, json_mode=state.json_mode)


@accounts_app.command("test")
def accounts_test(
    ctx: typer.Context,
    name: Annotated[
        str | None,
        typer.Argument(help="账户名称；省略时测试当前默认账户"),
    ] = None,
) -> None:
    """执行只读 IMAP 连接测试。"""
    state = _state(ctx)
    account = _handle(state, lambda: state.runtime.services.account(name))
    result = _handle(state, lambda: state.runtime.services.mail.test_connection(account))
    emit_value(result, json_mode=state.json_mode)


@accounts_app.command("remove")
def accounts_remove(
    ctx: typer.Context, name: str, yes: bool = typer.Option(False, "--yes")
) -> None:
    """删除账户配置及其钥匙串密码；需要确认。"""
    state = _state(ctx)
    account = _handle(state, lambda: state.runtime.services.account(name))
    if not yes and (state.json_mode or not typer.confirm(f"删除邮箱账户 {name}？", default=False)):
        raise typer.Abort()
    _handle(state, lambda: state.runtime.services.config_store.remove_account(name))
    _handle(state, lambda: state.runtime.services.secrets.delete(account))
    if state.runtime.services.session.active_account == name:
        state.runtime.services.session.active_account = None
    emit_value({"ok": True, "removed": name}, json_mode=state.json_mode)


@models_app.command("list")
def models_list(ctx: typer.Context) -> None:
    """列出本机 Ollama 模型。"""
    state = _state(ctx)
    models = _handle(state, state.runtime.services.ollama.installed_models)
    config = state.runtime.services.config_store.load()
    emit_value(
        {"ok": True, "models": models, "default_model": config.default_model},
        json_mode=state.json_mode,
    )


@models_app.command("use")
def models_use(ctx: typer.Context, model: str, yes: bool = typer.Option(False, "--yes")) -> None:
    """修改全局默认 Agent 模型；需要确认。"""
    state = _state(ctx)
    models = _handle(state, state.runtime.services.ollama.installed_models)
    if model not in models:
        raise typer.BadParameter(f"模型尚未安装：{model}")
    if not yes and (
        state.json_mode or not typer.confirm(f"把 {model} 设为全局默认模型？", default=False)
    ):
        raise typer.Abort()
    _handle(state, lambda: state.runtime.services.config_store.set_default_model(model))
    state.runtime.services.session.model = model
    emit_value({"ok": True, "default_model": model}, json_mode=state.json_mode)


@models_app.command("pull")
def models_pull(ctx: typer.Context, model: str, yes: bool = typer.Option(False, "--yes")) -> None:
    """从 Ollama 下载模型；需要确认。"""
    state = _state(ctx)
    if not yes and (
        state.json_mode or not typer.confirm(f"下载 Ollama 模型 {model}？", default=False)
    ):
        raise typer.Abort()
    if state.json_mode:
        _handle(state, lambda: state.runtime.services.ollama.pull(model))
    else:
        with Progress() as progress:
            task = progress.add_task(f"下载 {model}", total=None)

            def update(status: str, completed: int, total: int) -> None:
                progress.update(
                    task,
                    description=status,
                    total=total or None,
                    completed=completed,
                )

            _handle(state, lambda: state.runtime.services.ollama.pull(model, on_progress=update))
    emit_value({"ok": True, "model": model, "installed": True}, json_mode=state.json_mode)


@models_app.command("benchmark")
def models_benchmark(
    ctx: typer.Context,
    model: Annotated[str | None, typer.Option("--model")] = None,
    runs: Annotated[int, typer.Option(min=1, max=5)] = 1,
    body_chars: Annotated[int, typer.Option(min=200, max=5000)] = 1200,
    retry_body_chars: Annotated[int, typer.Option(min=200, max=5000)] = 900,
    num_ctx: Annotated[int, typer.Option(min=2048, max=16384)] = 4096,
    num_predict: Annotated[int, typer.Option(min=64, max=1024)] = 320,
    schema_in_prompt: Annotated[
        bool,
        typer.Option("--schema-in-prompt/--no-schema-in-prompt"),
    ] = False,
    suite: Annotated[
        bool,
        typer.Option("--suite", help="运行完整合成质量套件"),
    ] = False,
    case: Annotated[
        str | None,
        typer.Option("--case", help="仅运行一个命名合成质量用例"),
    ] = None,
) -> None:
    """用固定合成邮件测量分类耗时；不会读取真实邮箱。"""
    state = _state(ctx)
    selected_model = model or state.runtime.services.session.model
    settings = ClassificationSettings(
        body_chars=body_chars,
        retry_body_chars=retry_body_chars,
        num_ctx=num_ctx,
        num_predict=num_predict,
        include_schema_in_prompt=schema_in_prompt,
    )
    try:
        result = _handle(
            state,
            lambda: run_classifier_benchmark(
                state.runtime.services.ollama,
                model=selected_model,
                runs=runs,
                settings=settings,
                suite=suite,
                case=case,
            ),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--case") from exc
    emit_value(result, json_mode=state.json_mode)
    if not result.get("ok", False):
        raise typer.Exit(code=1)


@emails_app.command("list")
def emails_list(
    ctx: typer.Context,
    date_from: Annotated[str, typer.Option("--from")],
    date_to: Annotated[str, typer.Option("--to")],
    limit: Annotated[int, typer.Option(min=1, max=500)] = 100,
    account: str | None = typer.Option(None, "--account"),
    sender: str | None = typer.Option(None, "--sender"),
    subject: str | None = typer.Option(None, "--subject"),
) -> None:
    """确定性地搜索邮件元数据，不调用 Agent。"""
    state = _state(ctx)
    start = _parse_date(date_from, "--from")
    end = _parse_date(date_to, "--to")
    result = _handle(
        state,
        lambda: _registry(state).execute(
            "emails.search",
            {
                "date_from": start.isoformat(),
                "date_to": end.isoformat(),
                "limit": limit,
                "account": account,
                "sender": sender,
                "subject": subject,
            },
        ),
    )
    emit_value({"ok": True, "result": result.output}, json_mode=state.json_mode)


def _slash_help() -> str:
    return (
        "可以直接输入自然语言，例如：开始审查最近一周的信件\n\n"
        "全局快捷预设\n"
        "  1  上周完整审查（数字键直接选择）\n"
        "  2  今日重点\n"
        "  3  自定义命令或自然语言\n"
        "  /quick                     随时重新打开快捷预设\n\n"
        "常用操作\n"
        "  /review                     分类并登记上周邮件\n"
        "  /today                      分类并登记今天的邮件\n"
        "  输入编号 或 /open 编号       查看 AI 建议与邮件正文\n"
        "  /list                       返回完整登记簿\n"
        "  /list P0                    只看紧急邮件\n"
        "  /list 待处理                只看需要操作的邮件\n"
        "  /back                       从详情返回登记簿\n\n"
        "账户与模型\n"
        "  /account                    查看账户和邮箱出处\n"
        "  /account add                新增邮箱账户\n"
        "  /account providers          查看支持的邮箱出处\n"
        "  /account 名称               切换会话账户\n"
        "  /model [名称]               查看或切换会话模型\n\n"
        "诊断与会话\n"
        "  /status  /tools             检查状态、查看只读工具\n"
        "  /clear                      清除当前分析结果\n"
        "  /cancel                     查看如何取消运行中的任务\n"
        "  /exit                       退出 Mailweek\n\n"
        "提示：Mailweek 不会发送、删除、移动或标记邮件。"
    )


def _review_entries(
    state: CLIState,
    filter_text: str | None = None,
) -> list[tuple[int, Any]]:
    session = state.runtime.services.session
    entries = list(enumerate(session.review_items(), start=1))
    if not filter_text:
        return entries
    needle = filter_text.strip().casefold()
    return [
        (index, item)
        for index, item in entries
        if needle
        in " ".join(
            (
                item.priority.value,
                item.theme.value,
                item.subject,
                item.sender,
                "待处理" if item.action_required else "仅查看",
            )
        ).casefold()
    ]


def _show_review_registry(state: CLIState, filter_text: str | None = None) -> bool:
    session = state.runtime.services.session
    all_items = session.review_items()
    if not all_items:
        Console().print("[yellow]当前会话还没有邮件登记簿，请先输入 /review。[/yellow]")
        return False
    emit_review_registry(
        _review_entries(state, filter_text),
        summary=session.last_summary,
        headers=session.headers,
        total=len(all_items),
        filter_label=filter_text.strip() if filter_text else None,
        account_source=_account_origin_label(
            state,
            session.last_account or session.active_account,
        ),
    )
    return True


def _parse_review_index(value: str) -> int | None:
    cleaned = value.strip()
    if not cleaned.isdecimal():
        return None
    index = int(cleaned)
    return index if index > 0 else None


def _quick_key_bindings(is_active: Callable[[], bool]) -> KeyBindings:
    bindings = KeyBindings()
    active = Condition(is_active)

    def accept_choice(choice: str) -> Callable[[KeyPressEvent], None]:
        def handler(event: KeyPressEvent) -> None:
            event.current_buffer.insert_text(choice)
            if event.current_buffer.text == choice:
                event.current_buffer.validate_and_handle()

        return handler

    for choice in ("1", "2", "3"):
        bindings.add(choice, filter=active)(accept_choice(choice))
    return bindings


def _quick_choice_command(choice: str, read_custom: Callable[[], str]) -> str:
    if choice == "1":
        return "/review"
    if choice == "2":
        return "/today"
    if choice == "3":
        return read_custom().strip()
    return choice


def _open_review_item(state: CLIState, agent: AgentLoop, index: int) -> bool:
    session = state.runtime.services.session
    items = session.review_items()
    if not items:
        Console().print("[yellow]当前会话还没有邮件登记簿，请先输入 /review。[/yellow]")
        return False
    if index > len(items):
        Console().print(f"[red]编号超出范围：请输入 1–{len(items)}。[/red]")
        return False
    classification = items[index - 1]
    try:
        result = agent.registry.execute(
            "emails.get_content",
            {
                "uid": classification.uid,
                "account": session.last_account,
                "folder": session.last_folder,
                "max_chars": 6000,
            },
        )
    except MailweekError as exc:
        emit_error(exc.as_dict(), json_mode=False)
        return False
    if not isinstance(result.output, dict):
        Console().print("[red]邮件正文工具返回了无法识别的结果。[/red]")
        return False
    content = EmailContent.model_validate(result.output)
    result.output = None
    emit_email_detail(
        index,
        classification,
        content,
        header=session.headers.get(classification.uid),
        account_source=_account_origin_label(
            state,
            session.last_account or session.active_account,
        ),
    )
    return True


def _close_agent_events(agent: AgentLoop) -> None:
    close = getattr(agent.events, "close", None)
    if callable(close):
        close()


def run_repl(state: CLIState) -> None:
    console = Console()
    emit_startup_banner(
        state.runtime.services.session.model,
        _account_origin_label(
            state,
            state.runtime.services.session.active_account,
            include_email=False,
        ),
        console=console,
    )
    emit_quick_presets(console=console)
    agent = _agent(state)
    last_interrupt = 0.0
    selected_index: int | None = None
    quick_mode = True
    quick_key_active = False
    prompt_session: PromptSession[str] = PromptSession(
        key_bindings=_quick_key_bindings(lambda: quick_key_active)
    )
    while True:
        try:
            quick_context = quick_mode or (
                selected_index is None
                and not state.runtime.services.session.last_review_uids
            )
            quick_key_active = quick_context
            if quick_context:
                prompt = "<ansicyan>mailweek:quick&gt;</ansicyan> "
            elif selected_index is not None:
                prompt = f"<ansicyan>mailweek:mail#{selected_index}&gt;</ansicyan> "
            elif state.runtime.services.session.last_review_uids:
                prompt = "<ansicyan>mailweek:list&gt;</ansicyan> "
            else:
                prompt = "<ansicyan>mailweek&gt;</ansicyan> "
            text = prompt_session.prompt(HTML(prompt)).strip()
            if quick_context and text in {"1", "2", "3"}:
                quick_key_active = False
                text = _quick_choice_command(
                    text,
                    lambda: prompt_session.prompt(
                        HTML("<ansimagenta>mailweek:custom&gt;</ansimagenta> ")
                    ),
                ).strip()
                if not text:
                    console.print(
                        "[dim]已取消自定义输入；请重新选择 1 / 2 / 3。[/dim]"
                    )
                    continue
                quick_mode = False
            elif quick_context:
                quick_mode = False
        except EOFError:
            break
        except KeyboardInterrupt:
            _close_agent_events(agent)
            now = time.monotonic()
            if now - last_interrupt < 2:
                break
            last_interrupt = now
            console.print("[dim]已取消输入；再次 Ctrl+C 退出。[/dim]")
            continue
        if not text:
            continue
        numeric_index = _parse_review_index(text)
        if numeric_index is not None:
            if _open_review_item(state, agent, numeric_index):
                selected_index = numeric_index
            continue
        if text.startswith("/"):
            command, _, argument = text.partition(" ")
            if command in {"/exit", "/quit"}:
                break
            if command == "/help":
                console.print(_slash_help(), markup=False)
            elif command == "/quick":
                quick_mode = True
                emit_quick_presets(console=console)
            elif command == "/tools":
                tools_list_for_repl = [
                    {
                        "name": spec.name,
                        "description": spec.description,
                    }
                    for spec in agent.registry.specs()
                ]
                emit_tools(tools_list_for_repl, json_mode=False)
            elif command == "/status":
                result = _handle(
                    state,
                    lambda: run_doctor(
                        state.runtime.services.config_store,
                        state.runtime.services.secrets,
                        state.runtime.services.ollama,
                    ),
                )
                emit_value(result, json_mode=False)
                if result.get("ready"):
                    console.print(
                        "[bold bright_cyan]下一步：[/bold bright_cyan]"
                        "状态正常，可输入 [bold]/review[/bold] 开始审查。"
                    )
                else:
                    console.print(
                        "[bold bright_cyan]下一步：[/bold bright_cyan]"
                        "按上方提示完成缺失配置，再输入 [bold]/status[/bold] 复查。"
                    )
            elif command == "/account":
                account_action = argument.strip()
                if account_action == "add":
                    if _interactive_add_account(state):
                        agent.clear()
                        selected_index = None
                elif account_action == "providers":
                    emit_providers(provider_catalog(), json_mode=False)
                elif account_action:
                    account_name = account_action
                    _handle(
                        state,
                        partial(
                            agent.registry.execute,
                            "accounts.select_session",
                            {"account": account_name},
                        ),
                    )
                    console.print(
                        f"[green]✓[/green] 已切换到 [bold]{escape(account_name)}[/bold]"
                        "（仅当前会话）"
                    )
                    origin = _account_origin_label(state, account_name)
                    if origin:
                        console.print(f"[dim]邮箱出处：{origin}[/dim]")
                    console.print(
                        "[bold bright_cyan]下一步：[/bold bright_cyan]"
                        "原登记簿已清除；输入 [bold]/review[/bold] 审查这个账户。"
                    )
                    selected_index = None
                else:
                    emit_accounts(_account_rows(state), json_mode=False)
            elif command == "/model":
                if argument:
                    models = _handle(state, state.runtime.services.ollama.installed_models)
                    if argument.strip() not in models:
                        console.print(
                            f"[red]模型未安装：{escape(argument.strip())}[/red]"
                        )
                        available = (
                            "、".join(escape(model) for model in models)
                            if models
                            else "未检测到已安装模型"
                        )
                        console.print(
                            f"[dim]可用模型：{available}；用法：/model 模型名[/dim]"
                        )
                    else:
                        state.runtime.services.session.model = argument.strip()
                        agent.clear()
                        selected_index = None
                        console.print(
                            f"[green]✓[/green] 当前会话模型：{escape(argument.strip())}"
                        )
                        console.print(
                            "[bold bright_cyan]下一步：[/bold bright_cyan]"
                            "原登记簿已清除；输入 [bold]/review[/bold] 用新模型重新审查。"
                        )
                else:
                    console.print(
                        "当前会话模型："
                        f"{escape(state.runtime.services.session.model)}"
                    )
                    console.print(
                        "[dim]切换方式：/model 模型名；"
                        "查看已安装模型：在 Shell 运行 mailweek models list。[/dim]"
                    )
            elif command == "/review":
                start, end = _last_week()
                review_prompt = (
                    f"回顾 {start.isoformat()} 到 {end.isoformat()} 的邮件，按重要性排序。"
                )
                selected_index = None
                _handle(
                    state,
                    partial(
                        _run_agent_and_render,
                        state,
                        agent,
                        review_prompt,
                        force_registry=True,
                    ),
                )
            elif command == "/today":
                today = _today()
                review_prompt = (
                    f"回顾 {today.isoformat()} 到 {today.isoformat()} 的邮件，"
                    "按重要性排序。"
                )
                selected_index = None
                _handle(
                    state,
                    partial(
                        _run_agent_and_render,
                        state,
                        agent,
                        review_prompt,
                        force_registry=True,
                    ),
                )
            elif command == "/list":
                selected_index = None
                _show_review_registry(state, argument or None)
            elif command == "/open":
                index = _parse_review_index(argument)
                if index is None:
                    console.print("[yellow]用法：/open 编号，例如 /open 3[/yellow]")
                elif _open_review_item(state, agent, index):
                    selected_index = index
            elif command == "/back":
                selected_index = None
                _show_review_registry(state)
            elif command == "/clear":
                agent.clear()
                selected_index = None
                quick_mode = True
                console.print("[green]✓[/green] 当前会话上下文和登记簿已清除。")
                console.print(
                    "[bold bright_cyan]下一步：[/bold bright_cyan]"
                    "输入 [bold]/review[/bold] 重新生成登记簿，或直接描述新任务。"
                )
                emit_quick_presets(console=console)
            elif command == "/cancel":
                console.print(
                    "[dim]当前没有后台任务；审查或搜索运行时按 Ctrl+C 可立即取消。[/dim]"
                )
            else:
                console.print(f"[red]未知命令：{command}[/red]")
                console.print(_slash_help(), markup=False)
            continue
        try:
            previous_order = tuple(state.runtime.services.session.last_review_uids)
            _run_agent_and_render(state, agent, text)
            if tuple(state.runtime.services.session.last_review_uids) != previous_order:
                selected_index = None
        except KeyboardInterrupt:
            _close_agent_events(agent)
            console.print("[yellow]当前任务已取消。[/yellow]")
        except MailweekError as exc:
            emit_error(exc.as_dict(), json_mode=False)
    _close_agent_events(agent)
    state.runtime.close()
    console.print("[dim]会话已清除。[/dim]")


def main() -> None:
    json_mode = "--json" in sys.argv[1:]
    try:
        exit_code = app(prog_name="mailweek", standalone_mode=False)
        if isinstance(exit_code, int) and exit_code != 0:
            raise SystemExit(exit_code)
    except MailweekError as exc:
        emit_error(exc.as_dict(), json_mode=json_mode)
        raise SystemExit(1) from exc
    except ClickException as exc:
        if json_mode:
            emit_error(
                {
                    "ok": False,
                    "error": {
                        "code": "cli_usage_error",
                        "message": exc.format_message(),
                        "hint": "运行 mailweek --help 查看命令格式。",
                    },
                },
                json_mode=True,
            )
        else:
            exc.show()
        raise SystemExit(exc.exit_code) from exc
    except typer.Abort as exc:
        if json_mode:
            emit_error(
                {
                    "ok": False,
                    "error": {
                        "code": "operation_aborted",
                        "message": "操作已取消或缺少所需确认。",
                        "hint": "检查参数，并仅在明确授权后使用 --yes。",
                    },
                },
                json_mode=True,
            )
        raise SystemExit(1) from exc
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":
    main()
