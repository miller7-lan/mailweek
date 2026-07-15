from __future__ import annotations

import pytest

from mailweek.providers import (
    detect_provider,
    provider_by_key,
    provider_catalog,
    resolve_provider,
    suggested_imap_host,
)


def test_provider_is_detected_from_email_or_imap_host() -> None:
    assert detect_provider("user@qq.com").key == "qq"
    assert detect_provider("user@example.com", "imap.gmail.com").key == "gmail"
    assert detect_provider("user@example.com").key == "custom"


def test_provider_aliases_and_custom_host_are_stable() -> None:
    gmail = resolve_provider("google", email="user@example.com")

    assert gmail.key == "gmail"
    assert suggested_imap_host("user@example.com", gmail) == "imap.gmail.com"
    assert suggested_imap_host(
        "user@example.com",
        provider_by_key("custom"),
    ) == "imap.example.com"


def test_unknown_provider_is_rejected_without_secrets() -> None:
    with pytest.raises(ValueError, match="未知邮箱出处"):
        provider_by_key("unknown")

    assert all(set(item) == {"id", "name", "imap_host"} for item in provider_catalog())
