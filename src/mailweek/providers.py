from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MailProvider:
    key: str
    label: str
    imap_host: str | None
    domains: tuple[str, ...] = ()
    host_aliases: tuple[str, ...] = ()


PROVIDERS: tuple[MailProvider, ...] = (
    MailProvider(
        "qq",
        "QQ邮箱",
        "imap.qq.com",
        ("qq.com", "vip.qq.com", "foxmail.com"),
        ("imap.qq.com",),
    ),
    MailProvider(
        "gmail",
        "Gmail",
        "imap.gmail.com",
        ("gmail.com", "googlemail.com"),
        ("imap.gmail.com",),
    ),
    MailProvider(
        "outlook",
        "Outlook",
        "outlook.office365.com",
        ("outlook.com", "hotmail.com", "live.com"),
        ("outlook.office365.com", "imap-mail.outlook.com"),
    ),
    MailProvider(
        "icloud",
        "iCloud 邮箱",
        "imap.mail.me.com",
        ("icloud.com", "me.com", "mac.com"),
        ("imap.mail.me.com",),
    ),
    MailProvider(
        "163",
        "网易163邮箱",
        "imap.163.com",
        ("163.com", "yeah.net"),
        ("imap.163.com", "imap.yeah.net"),
    ),
    MailProvider(
        "126",
        "网易126邮箱",
        "imap.126.com",
        ("126.com",),
        ("imap.126.com",),
    ),
    MailProvider(
        "custom",
        "自定义 IMAP",
        None,
    ),
)

PROVIDER_BY_KEY = {provider.key: provider for provider in PROVIDERS}
PROVIDER_ALIASES = {
    "auto": "auto",
    "qqmail": "qq",
    "google": "gmail",
    "hotmail": "outlook",
    "office365": "outlook",
    "apple": "icloud",
    "netease": "163",
}


def provider_catalog() -> list[dict[str, str | None]]:
    return [
        {
            "id": provider.key,
            "name": provider.label,
            "imap_host": provider.imap_host,
        }
        for provider in PROVIDERS
    ]


def provider_by_key(value: str) -> MailProvider:
    key = PROVIDER_ALIASES.get(value.casefold(), value.casefold())
    provider = PROVIDER_BY_KEY.get(key)
    if provider is None:
        choices = ", ".join(PROVIDER_BY_KEY)
        raise ValueError(f"未知邮箱出处 {value!r}；可选：{choices}")
    return provider


def detect_provider(email: str, host: str | None = None) -> MailProvider:
    normalized_host = (host or "").casefold()
    domain = email.rsplit("@", 1)[-1].casefold()
    for provider in PROVIDERS:
        if normalized_host and normalized_host in provider.host_aliases:
            return provider
        if domain in provider.domains:
            return provider
    return PROVIDER_BY_KEY["custom"]


def resolve_provider(
    requested: str | None,
    *,
    email: str,
    host: str | None = None,
) -> MailProvider:
    if requested is None or requested.casefold() == "auto":
        return detect_provider(email, host)
    return provider_by_key(requested)


def suggested_imap_host(email: str, provider: MailProvider) -> str:
    if provider.imap_host:
        return provider.imap_host
    domain = email.rsplit("@", 1)[-1].casefold()
    return f"imap.{domain}"


def provider_label(
    configured: str | None,
    *,
    email: str,
    host: str,
) -> str:
    try:
        return resolve_provider(configured, email=email, host=host).label
    except ValueError:
        return detect_provider(email, host).label


def provider_id(
    configured: str | None,
    *,
    email: str,
    host: str,
) -> str:
    try:
        return resolve_provider(configured, email=email, host=host).key
    except ValueError:
        return detect_provider(email, host).key
