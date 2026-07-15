from __future__ import annotations

from mailweek.schemas import EmailClassification, Priority, Theme, priority_for_score


def test_priority_boundaries() -> None:
    assert priority_for_score(100) is Priority.P0
    assert priority_for_score(90) is Priority.P0
    assert priority_for_score(89) is Priority.P1
    assert priority_for_score(70) is Priority.P1
    assert priority_for_score(69) is Priority.P2
    assert priority_for_score(40) is Priority.P2
    assert priority_for_score(15) is Priority.P3
    assert priority_for_score(14) is Priority.P4


def test_classification_overrides_model_priority_from_score() -> None:
    item = EmailClassification(
        uid="7",
        subject="付款失败",
        sender="billing@example.com",
        priority_score=95,
        priority="P4",
        theme=Theme.FINANCE,
        summary="付款失败",
        importance_reason="会中断服务",
        action_required=True,
        suggested_action="立即检查付款方式",
        confidence=0.9,
    )
    assert item.priority is Priority.P0
