from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


class AccountConfig(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[\w.-]+$")]
    email: Annotated[str, Field(min_length=3, max_length=320)]
    provider: Annotated[str, Field(min_length=1, max_length=32)] = "auto"
    host: Annotated[str, Field(min_length=1, max_length=255)]
    port: Annotated[int, Field(ge=1, le=65535)] = 993
    username: Annotated[str, Field(min_length=1, max_length=320)]
    folder: Annotated[str, Field(min_length=1, max_length=255)] = "INBOX"
    use_ssl: Literal[True] = True


class AppConfig(BaseModel):
    version: int = 1
    default_model: str = "qwen3.5:9b"
    fast_model: str = "qwen3.5:4b"
    ollama_host: str = "http://127.0.0.1:11434"
    active_account: str | None = None
    accounts: dict[str, AccountConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def active_account_exists(self) -> AppConfig:
        if self.active_account is not None and self.active_account not in self.accounts:
            self.active_account = None
        return self


class AttachmentInfo(BaseModel):
    filename: str | None = None
    content_type: str


class EmailHeader(BaseModel):
    uid: str
    message_id: str | None = None
    subject: str
    sender: str
    recipients: list[str] = Field(default_factory=list)
    received_at: datetime | None = None
    attachments: list[AttachmentInfo] = Field(default_factory=list)


class EmailContent(BaseModel):
    uid: str
    subject: str
    sender: str
    received_at: datetime | None = None
    body: str
    content_type: str
    truncated: bool
    attachments: list[AttachmentInfo] = Field(default_factory=list)
    trust: Literal["untrusted_email_content"] = "untrusted_email_content"


class SearchResult(BaseModel):
    account: str
    folder: str
    date_from: date
    date_to: date
    total_found: int
    returned: int
    truncated: bool
    emails: list[EmailHeader]


class Priority(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class Theme(StrEnum):
    WORK_PROJECT = "工作项目"
    CALENDAR = "日程会议"
    FINANCE = "财务账单"
    SECURITY = "账户安全"
    PERSONAL = "个人社交"
    NEWSLETTER = "新闻订阅"
    MARKETING = "营销推广"
    OTHER = "其他"


def priority_for_score(score: int) -> Priority:
    if score >= 90:
        return Priority.P0
    if score >= 70:
        return Priority.P1
    if score >= 40:
        return Priority.P2
    if score >= 15:
        return Priority.P3
    return Priority.P4


class EmailClassification(BaseModel):
    uid: str
    subject: str
    sender: str
    priority_score: Annotated[int, Field(ge=0, le=100)]
    priority: Priority = Priority.P4
    theme: Theme
    summary: Annotated[str, Field(min_length=1, max_length=500)]
    importance_reason: Annotated[str, Field(min_length=1, max_length=500)]
    action_required: bool
    suggested_action: Annotated[str | None, Field(max_length=500)] = None
    due_at: datetime | None = None
    confidence: Annotated[float, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def derive_priority(self) -> EmailClassification:
        self.priority = priority_for_score(self.priority_score)
        return self


class EmailDecision(BaseModel):
    """The model's decision for one email; identity is supplied by the program."""

    priority_score: Annotated[int, Field(ge=0, le=100)]
    theme: Theme
    summary: Annotated[str, Field(min_length=1, max_length=500)]
    importance_reason: Annotated[str, Field(min_length=1, max_length=500)]
    action_required: bool
    suggested_action: Annotated[str | None, Field(max_length=500)] = None
    due_at: datetime | None = None
    confidence: Annotated[float, Field(ge=0, le=1)]


class ClassificationBatch(BaseModel):
    items: Annotated[list[EmailClassification], Field(max_length=8)]


class ReviewSummary(BaseModel):
    overview: Annotated[str, Field(min_length=1, max_length=2000)]
    top_actions: Annotated[list[str], Field(max_length=12)] = Field(default_factory=list)
    risk_notes: Annotated[list[str], Field(max_length=12)] = Field(default_factory=list)
    key_topics: Annotated[list[str], Field(max_length=12)] = Field(default_factory=list)
    priorities: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def with_counts(cls, summary: ReviewSummary, items: list[EmailClassification]) -> ReviewSummary:
        counts = Counter(item.priority.value for item in items)
        summary.priorities = {
            priority.value: counts.get(priority.value, 0) for priority in Priority
        }
        return summary


class AgentRunResult(BaseModel):
    ok: Literal[True] = True
    answer: str
    account: str | None = None
    range: dict[str, str] | None = None
    tool_calls: int = 0
    emails_analyzed: int = 0
    priorities: dict[str, int] = Field(default_factory=dict)


class ToolExecutionResult(BaseModel):
    output: dict[str, Any] | list[Any] | str | int | bool | None
    sensitive: bool = False
    compacted: str = "[敏感工具结果已从会话上下文清除]"
