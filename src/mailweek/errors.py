from __future__ import annotations


class MailweekError(Exception):
    """A user-facing error with a stable machine-readable code."""

    def __init__(self, code: str, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": False,
            "error": {"code": self.code, "message": self.message, "hint": self.hint},
        }


class ConfigError(MailweekError):
    pass


class SecretError(MailweekError):
    pass


class MailError(MailweekError):
    pass


class ModelError(MailweekError):
    pass


class ToolError(MailweekError):
    pass


class AgentLimitError(MailweekError):
    pass
