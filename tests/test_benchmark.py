from __future__ import annotations

import json

from mailweek.analysis import ClassificationSettings
from mailweek.benchmark import run_classifier_benchmark
from mailweek.ollama_client import NormalizedMessage


class FakeOllama:
    def __init__(
        self,
        *,
        score: int = 95,
        action_required: bool = True,
        theme: str = "工作项目",
    ) -> None:
        self.score = score
        self.action_required = action_required
        self.theme = theme

    def chat(self, **_kwargs):
        return NormalizedMessage(
            content=json.dumps(
                {
                    "priority_score": self.score,
                    "theme": self.theme,
                    "summary": "需要本周回复",
                    "importance_reason": "存在常规截止日期",
                    "action_required": self.action_required,
                    "suggested_action": "回复负责人" if self.action_required else None,
                    "confidence": 0.95,
                },
                ensure_ascii=False,
            ),
            thinking="",
            tool_calls=[],
        )


class PerfectSuiteOllama:
    def __init__(self) -> None:
        self.decisions = iter(
            (
                (95, "账户安全", True),
                (80, "工作项目", True),
                (20, "财务账单", False),
                (20, "新闻订阅", False),
                (5, "营销推广", False),
                (95, "财务账单", True),
                (80, "日程会议", True),
                (50, "工作项目", False),
                (20, "其他", False),
                (5, "营销推广", False),
                (50, "财务账单", False),
            )
        )

    def chat(self, **_kwargs):
        score, theme, action = next(self.decisions)
        return NormalizedMessage(
            content=json.dumps(
                {
                    "priority_score": score,
                    "theme": theme,
                    "summary": "质量套件摘要",
                    "importance_reason": "质量套件原因",
                    "action_required": action,
                    "suggested_action": "立即处理" if action else None,
                    "confidence": 0.95,
                },
                ensure_ascii=False,
            ),
            thinking="",
            tool_calls=[],
        )


def test_benchmark_fails_quality_gate_when_valid_output_has_wrong_priority() -> None:
    result = run_classifier_benchmark(
        FakeOllama(),  # type: ignore[arg-type]
        model="test-model",
        runs=1,
        settings=ClassificationSettings(),
    )

    assert result["runs"][0]["valid"] is True
    assert result["runs"][0]["expected_match"] is False
    assert result["ok"] is False


def test_benchmark_reports_explicit_quality_metrics() -> None:
    result = run_classifier_benchmark(
        FakeOllama(score=80),  # type: ignore[arg-type]
        model="test-model",
        runs=1,
        settings=ClassificationSettings(),
    )

    assert result["ok"] is True
    assert result["valid_outputs"] == 1
    assert result["expected_matches"] == 1
    assert result["match_rate"] == 1.0


def test_quality_suite_covers_at_least_ten_priority_boundary_cases() -> None:
    result = run_classifier_benchmark(
        PerfectSuiteOllama(),  # type: ignore[arg-type]
        model="test-model",
        runs=1,
        settings=ClassificationSettings(),
        suite=True,
    )

    case_names = {sample["case"] for sample in result["runs"]}
    assert result["samples"] >= 11
    assert {"autopay_receipt", "invoice_record"} <= case_names
    assert result["ok"] is True


def test_benchmark_can_run_one_named_quality_case() -> None:
    result = run_classifier_benchmark(
        FakeOllama(
            score=50,
            action_required=False,
            theme="财务账单",
        ),  # type: ignore[arg-type]
        model="test-model",
        runs=1,
        settings=ClassificationSettings(),
        case="invoice_record",
    )

    assert result["samples"] == 1
    assert result["runs"][0]["case"] == "invoice_record"
    assert result["ok"] is True
