from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ollama import Client

from .errors import ModelError


@dataclass(frozen=True)
class NormalizedToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class InferenceMetrics:
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration_ns: int = 0
    eval_count: int = 0
    eval_duration_ns: int = 0

    @staticmethod
    def _seconds(value: int) -> float:
        return round(value / 1_000_000_000, 3)

    @staticmethod
    def _rate(tokens: int, duration_ns: int) -> float | None:
        if tokens <= 0 or duration_ns <= 0:
            return None
        return round(tokens / (duration_ns / 1_000_000_000), 2)

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "total_seconds": self._seconds(self.total_duration_ns),
            "load_seconds": self._seconds(self.load_duration_ns),
            "prompt_tokens": self.prompt_eval_count,
            "prompt_seconds": self._seconds(self.prompt_eval_duration_ns),
            "prompt_tokens_per_second": self._rate(
                self.prompt_eval_count, self.prompt_eval_duration_ns
            ),
            "output_tokens": self.eval_count,
            "output_seconds": self._seconds(self.eval_duration_ns),
            "output_tokens_per_second": self._rate(self.eval_count, self.eval_duration_ns),
        }


@dataclass(frozen=True)
class NormalizedMessage:
    content: str
    thinking: str
    tool_calls: list[NormalizedToolCall]
    metrics: InferenceMetrics | None = None

    def as_assistant_dict(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.thinking:
            message["thinking"] = self.thinking
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        return message


def _get(value: object, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


class OllamaAdapter:
    def __init__(
        self,
        host: str,
        *,
        client: Client | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.client = client or Client(host=self.host, timeout=timeout)

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        think: bool = False,
        format_schema: dict[str, Any] | None = None,
        temperature: float = 0.1,
        num_ctx: int = 16384,
        num_predict: int | None = None,
    ) -> NormalizedMessage:
        options: dict[str, Any] = {"temperature": temperature, "num_ctx": num_ctx}
        if num_predict is not None:
            options["num_predict"] = num_predict
        try:
            response = self.client.chat(
                model=model,
                messages=messages,
                tools=tools,
                think=think,
                format=format_schema,
                stream=False,
                options=options,
            )
        except Exception as exc:
            raise ModelError(
                "ollama_chat_failed",
                f"Ollama 模型 {model} 调用失败。",
                f"确认 Ollama 正在运行且模型已下载。详情：{exc}",
            ) from exc
        raw_message = _get(response, "message", {})
        calls: list[NormalizedToolCall] = []
        for raw_call in _get(raw_message, "tool_calls", []) or []:
            function = _get(raw_call, "function", {})
            name = str(_get(function, "name", ""))
            arguments = _get(function, "arguments", {}) or {}
            if not isinstance(arguments, dict):
                arguments = {}
            if name:
                calls.append(NormalizedToolCall(name=name, arguments=arguments))
        return NormalizedMessage(
            content=str(_get(raw_message, "content", "") or ""),
            thinking=str(_get(raw_message, "thinking", "") or ""),
            tool_calls=calls,
            metrics=InferenceMetrics(
                total_duration_ns=int(_get(response, "total_duration", 0) or 0),
                load_duration_ns=int(_get(response, "load_duration", 0) or 0),
                prompt_eval_count=int(_get(response, "prompt_eval_count", 0) or 0),
                prompt_eval_duration_ns=int(
                    _get(response, "prompt_eval_duration", 0) or 0
                ),
                eval_count=int(_get(response, "eval_count", 0) or 0),
                eval_duration_ns=int(_get(response, "eval_duration", 0) or 0),
            ),
        )

    def installed_models(self) -> list[str]:
        try:
            response = self.client.list()
        except Exception as exc:
            raise ModelError(
                "ollama_unreachable",
                "无法连接本地 Ollama。",
                f"请先启动 Ollama。详情：{exc}",
            ) from exc
        names: list[str] = []
        for item in _get(response, "models", []) or []:
            name = _get(item, "model") or _get(item, "name")
            if name:
                names.append(str(name))
        return sorted(names)

    def pull(
        self, model: str, *, on_progress: Callable[[str, int, int], None] | None = None
    ) -> None:
        try:
            stream: Iterable[object] = self.client.pull(model=model, stream=True)
            for event in stream:
                status = str(_get(event, "status", "下载中"))
                completed = int(_get(event, "completed", 0) or 0)
                total = int(_get(event, "total", 0) or 0)
                if on_progress:
                    on_progress(status, completed, total)
        except Exception as exc:
            raise ModelError("ollama_pull_failed", f"模型 {model} 下载失败。", str(exc)) from exc
