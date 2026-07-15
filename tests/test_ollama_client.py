from __future__ import annotations

import pytest

from mailweek.errors import ModelError
from mailweek.ollama_client import OllamaAdapter


class TimeoutClient:
    def chat(self, **_kwargs):
        raise TimeoutError("request exceeded deadline")


class MetricsClient:
    def chat(self, **_kwargs):
        return {
            "message": {"content": "{}", "thinking": "", "tool_calls": []},
            "total_duration": 3_000_000_000,
            "load_duration": 100_000_000,
            "prompt_eval_count": 200,
            "prompt_eval_duration": 1_000_000_000,
            "eval_count": 50,
            "eval_duration": 2_000_000_000,
        }


def test_chat_timeout_is_reported_as_safe_model_error() -> None:
    adapter = OllamaAdapter("http://127.0.0.1:11434", client=TimeoutClient())  # type: ignore[arg-type]

    with pytest.raises(ModelError) as error:
        adapter.chat(model="fake", messages=[{"role": "user", "content": "hello"}])

    assert error.value.code == "ollama_chat_failed"


def test_chat_exposes_safe_inference_metrics() -> None:
    adapter = OllamaAdapter("http://127.0.0.1:11434", client=MetricsClient())  # type: ignore[arg-type]

    response = adapter.chat(model="fake", messages=[{"role": "user", "content": "hello"}])

    assert response.metrics is not None
    assert response.metrics.as_dict() == {
        "total_seconds": 3.0,
        "load_seconds": 0.1,
        "prompt_tokens": 200,
        "prompt_seconds": 1.0,
        "prompt_tokens_per_second": 200.0,
        "output_tokens": 50,
        "output_seconds": 2.0,
        "output_tokens_per_second": 25.0,
    }
