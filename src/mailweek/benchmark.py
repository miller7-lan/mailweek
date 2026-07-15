from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

from .analysis import ClassificationSettings, MailAnalyzer
from .ollama_client import OllamaAdapter
from .schemas import EmailContent, Priority


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    subject: str
    paragraph: str
    expected_priority: Priority
    expected_action: bool


SYNTHETIC_CASES = (
    SyntheticCase(
        name="security_incident",
        subject="账号安全设置被未知设备修改",
        paragraph=(
            "系统检测到未知设备登录并修改安全设置，"
            "如果不是本人操作，需要立即冻结会话并重置凭据。"
        ),
        expected_priority=Priority.P0,
        expected_action=True,
    ),
    SyntheticCase(
        name="project_reply",
        subject="请在本周五前确认项目发布状态",
        paragraph=(
            "项目负责人请求在本周五之前确认测试结果，"
            "并回复是否可以发布；当前没有付款、安全或账号风险。"
        ),
        expected_priority=Priority.P1,
        expected_action=True,
    ),
    SyntheticCase(
        name="paid_invoice",
        subject="本月服务账单已支付",
        paragraph=(
            "本月服务账单已经成功自动支付，"
            "服务不会中断，仅需留档，不要求回复或采取操作。"
        ),
        expected_priority=Priority.P2,
        expected_action=False,
    ),
    SyntheticCase(
        name="newsletter",
        subject="每周技术资讯摘要",
        paragraph=(
            "这是一封新闻订阅，包含本周行业文章摘要。"
            "没有截止日期、风险、会议或回复要求，可有空时阅读。"
        ),
        expected_priority=Priority.P3,
        expected_action=False,
    ),
    SyntheticCase(
        name="marketing",
        subject="限时促销折扣",
        paragraph=(
            "这是一封营销邮件，宣传非必要商品折扣。"
            "没有账号风险、付款问题、工作事项或任何回复要求。"
        ),
        expected_priority=Priority.P4,
        expected_action=False,
    ),
)


def _synthetic_email(case: SyntheticCase, body_chars: int) -> EmailContent:
    padding = "补充说明：相关记录已整理，详细信息以邮件开头的事项为准。"
    body = (case.paragraph + padding * ((body_chars // len(padding)) + 1))[:body_chars]
    return EmailContent(
        uid="benchmark",
        subject=case.subject,
        sender="benchmark@example.invalid",
        body=body,
        content_type="text/plain",
        truncated=False,
    )


def run_classifier_benchmark(
    ollama: OllamaAdapter,
    *,
    model: str,
    runs: int,
    settings: ClassificationSettings,
    suite: bool = False,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    cases = SYNTHETIC_CASES if suite else (SYNTHETIC_CASES[1],)
    work = [(repeat, case) for repeat in range(1, runs + 1) for case in cases]
    for index, (repeat, case) in enumerate(work, start=1):
        analyzer = MailAnalyzer(ollama, model, settings=settings)
        started = time.monotonic()
        items, errors = analyzer.classify([_synthetic_email(case, settings.body_chars)])
        elapsed = round(time.monotonic() - started, 3)
        actual = items[0] if items else None
        samples.append(
            {
                "run": index,
                "repeat": repeat,
                "case": case.name,
                "elapsed_seconds": elapsed,
                "valid": len(items) == 1 and not errors,
                "expected_priority": case.expected_priority.value,
                "actual_priority": actual.priority.value if actual else None,
                "actual_score": actual.priority_score if actual else None,
                "expected_action": case.expected_action,
                "actual_action": actual.action_required if actual else None,
                "confidence": actual.confidence if actual else None,
                "expected_match": bool(
                    actual
                    and actual.priority is case.expected_priority
                    and actual.action_required is case.expected_action
                ),
                "attempts": len(analyzer.inference_metrics),
                "inference": [metric.as_dict() for metric in analyzer.inference_metrics],
                "errors": errors,
            }
        )
    elapsed_values = [float(sample["elapsed_seconds"]) for sample in samples]
    return {
        "ok": all(bool(sample["valid"]) for sample in samples),
        "kind": "synthetic_email_classification",
        "privacy": "不读取真实邮箱或邮件正文",
        "model": model,
        "suite": suite,
        "settings": {
            "body_chars": settings.body_chars,
            "retry_body_chars": settings.retry_body_chars,
            "num_ctx": settings.num_ctx,
            "num_predict": settings.num_predict,
            "schema_in_prompt": settings.include_schema_in_prompt,
        },
        "runs": samples,
        "average_seconds": round(statistics.mean(elapsed_values), 3),
        "median_seconds": round(statistics.median(elapsed_values), 3),
        "expected_matches": sum(bool(sample["expected_match"]) for sample in samples),
        "samples": len(samples),
    }
