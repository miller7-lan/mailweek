from __future__ import annotations

import base64
import quopri
import re
import ssl
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from email.header import decode_header, make_header
from email.parser import BytesHeaderParser
from email.policy import default
from email.utils import getaddresses, parsedate_to_datetime
from typing import Any

from bs4 import BeautifulSoup
from imapclient import IMAPClient  # type: ignore[import-untyped]

from .config import SecretStore
from .errors import MailError, SecretError
from .schemas import AccountConfig, AttachmentInfo, EmailContent, EmailHeader, SearchResult

HEADER_FETCH = "BODY.PEEK[HEADER.FIELDS (MESSAGE-ID SUBJECT FROM TO CC DATE CONTENT-TYPE)]"


@dataclass(frozen=True)
class BodyPart:
    number: str | None
    content_type: str
    charset: str | None
    encoding: str
    disposition: str | None
    filename: str | None
    size: int | None


def decode_mime_header(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:
        return value.strip()


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _params(value: object) -> dict[str, str]:
    if not isinstance(value, (tuple, list)):
        return {}
    items = list(value)
    return {
        _as_text(items[index]).upper(): decode_mime_header(_as_text(items[index + 1]))
        for index in range(0, len(items) - 1, 2)
    }


def _disposition_from_tail(values: Iterable[object]) -> tuple[str | None, dict[str, str]]:
    for value in values:
        if not isinstance(value, (tuple, list)) or not value:
            continue
        name = _as_text(value[0]).upper()
        if name in {"ATTACHMENT", "INLINE"}:
            parameters = _params(value[1]) if len(value) > 1 else {}
            return name.lower(), parameters
    return None, {}


def walk_bodystructure(node: object, prefix: str = "") -> list[BodyPart]:
    """Turn IMAP BODYSTRUCTURE tuples into addressable leaf parts."""
    if not isinstance(node, (tuple, list)) or not node:
        return []
    values = list(node)
    if isinstance(values[0], (tuple, list)):
        children: list[BodyPart] = []
        for index, value in enumerate(values, start=1):
            if not isinstance(value, (tuple, list)):
                break
            number = f"{prefix}.{index}" if prefix else str(index)
            children.extend(walk_bodystructure(value, number))
        return children

    if len(values) < 7:
        return []
    media_type = _as_text(values[0]).lower()
    subtype = _as_text(values[1]).lower()
    parameters = _params(values[2])
    disposition, disposition_params = _disposition_from_tail(values[7:])
    filename = disposition_params.get("FILENAME") or parameters.get("NAME")
    size = values[6] if isinstance(values[6], int) else None
    return [
        BodyPart(
            number=prefix or None,
            content_type=f"{media_type}/{subtype}",
            charset=parameters.get("CHARSET"),
            encoding=_as_text(values[5]).lower() or "7bit",
            disposition=disposition,
            filename=filename,
            size=size,
        )
    ]


def _find_fetch_value(payload: dict[bytes, Any], starts: tuple[bytes, ...]) -> bytes:
    for key, value in payload.items():
        upper = key.upper() if isinstance(key, bytes) else str(key).encode().upper()
        if any(upper.startswith(prefix) for prefix in starts) and isinstance(value, bytes):
            return value
    return b""


def _decode_body(raw: bytes, part: BodyPart) -> str:
    try:
        if part.encoding == "base64":
            compact = re.sub(rb"\s+", b"", raw)
            compact += b"=" * (-len(compact) % 4)
            decoded = base64.b64decode(compact, validate=False)
        elif part.encoding in {"quoted-printable", "quopri"}:
            decoded = quopri.decodestring(raw)
        else:
            decoded = raw
    except Exception:
        decoded = raw
    charset = part.charset or "utf-8"
    try:
        text = decoded.decode(charset, errors="replace")
    except LookupError:
        text = decoded.decode("utf-8", errors="replace")
    if part.content_type == "text/html":
        text = BeautifulSoup(text, "html.parser").get_text("\n")
    lines = [line.strip() for line in text.replace("\r", "").split("\n")]
    return "\n".join(line for line in lines if line).strip()


class MailService:
    """Read-only IMAP access. Connections live only for the current process."""

    def __init__(self, secrets: SecretStore, *, timeout: int = 30) -> None:
        self.secrets = secrets
        self.timeout = timeout
        self._connections: dict[str, IMAPClient] = {}

    def _connect(self, account: AccountConfig) -> IMAPClient:
        cached = self._connections.get(account.name)
        if cached is not None:
            try:
                cached.noop()
                return cached
            except Exception:
                self._connections.pop(account.name, None)
        password, _source = self.secrets.get(account)
        if not password:
            raise SecretError(
                "imap_password_missing",
                f"账户 {account.name} 没有可用的 IMAP 应用专用密码。",
                "运行 mailweek accounts add 重新保存密码，或设置 MAILWEEK_IMAP_PASSWORD。",
            )
        try:
            client = IMAPClient(
                account.host,
                port=account.port,
                ssl=True,
                ssl_context=ssl.create_default_context(),
                timeout=self.timeout,
                use_uid=True,
            )
            client.login(account.username, password)
            self._connections[account.name] = client
            return client
        except Exception as exc:
            raise MailError(
                "imap_connection_failed",
                f"无法连接邮箱账户 {account.name}。",
                f"请检查 IMAP 地址、用户名和应用专用密码。详情：{exc}",
            ) from exc

    def test_connection(self, account: AccountConfig) -> dict[str, object]:
        client = self._connect(account)
        capabilities = sorted(_as_text(value) for value in client.capabilities())
        return {"ok": True, "account": account.name, "capabilities": capabilities}

    def list_folders(self, account: AccountConfig) -> list[str]:
        client = self._connect(account)
        try:
            folders = client.list_folders()
            return [_as_text(item[2]) for item in folders]
        except Exception as exc:
            raise MailError("imap_list_failed", "无法读取邮箱文件夹列表。", str(exc)) from exc

    @staticmethod
    def _select(client: IMAPClient, folder: str) -> None:
        try:
            client.select_folder(folder, readonly=True)
        except Exception as exc:
            raise MailError(
                "imap_folder_failed", f"无法以只读方式打开邮箱文件夹：{folder}", str(exc)
            ) from exc

    @staticmethod
    def _parse_header(uid: int | str, raw: bytes, structure: object = None) -> EmailHeader:
        message = BytesHeaderParser(policy=default).parsebytes(raw)
        recipients = [
            address
            for _name, address in getaddresses([message.get("To", ""), message.get("Cc", "")])
            if address
        ]
        received_at: datetime | None = None
        if raw_date := message.get("Date"):
            with suppress(TypeError, ValueError, OverflowError):
                received_at = parsedate_to_datetime(raw_date)
        parts = walk_bodystructure(structure)
        attachments = [
            AttachmentInfo(filename=part.filename, content_type=part.content_type)
            for part in parts
            if part.disposition == "attachment" or part.filename
        ]
        return EmailHeader(
            uid=str(uid),
            message_id=message.get("Message-ID"),
            subject=decode_mime_header(message.get("Subject")) or "（无主题）",
            sender=decode_mime_header(message.get("From")) or "（未知发件人）",
            recipients=recipients,
            received_at=received_at,
            attachments=attachments,
        )

    def search(
        self,
        account: AccountConfig,
        *,
        date_from: date,
        date_to: date,
        folder: str | None = None,
        sender: str | None = None,
        subject: str | None = None,
        limit: int = 100,
    ) -> SearchResult:
        if date_to < date_from:
            raise MailError("invalid_date_range", "结束日期不能早于开始日期。")
        client = self._connect(account)
        target_folder = folder or account.folder
        self._select(client, target_folder)
        criteria: list[object] = ["SINCE", date_from, "BEFORE", date_to + timedelta(days=1)]
        if sender:
            criteria.extend(["FROM", sender])
        if subject:
            criteria.extend(["SUBJECT", subject])
        try:
            all_uids = list(client.search(criteria))
            selected = all_uids[-limit:]
            fetched = (
                client.fetch(
                    selected,
                    [
                        HEADER_FETCH,
                        "BODYSTRUCTURE",
                    ],
                )
                if selected
                else {}
            )
        except Exception as exc:
            raise MailError("imap_search_failed", "邮箱搜索失败。", str(exc)) from exc
        emails: list[EmailHeader] = []
        for uid in reversed(selected):
            payload = fetched.get(uid, {})
            raw = _find_fetch_value(payload, (b"BODY[HEADER", b"BODY.PEEK[HEADER"))
            structure = payload.get(b"BODYSTRUCTURE")
            emails.append(self._parse_header(uid, raw, structure))
        return SearchResult(
            account=account.name,
            folder=target_folder,
            date_from=date_from,
            date_to=date_to,
            total_found=len(all_uids),
            returned=len(emails),
            truncated=len(all_uids) > len(emails),
            emails=emails,
        )

    def get_headers(
        self, account: AccountConfig, uids: list[str], *, folder: str | None = None
    ) -> list[EmailHeader]:
        client = self._connect(account)
        self._select(client, folder or account.folder)
        try:
            fetched = client.fetch(
                [int(uid) for uid in uids],
                [
                    HEADER_FETCH,
                    "BODYSTRUCTURE",
                ],
            )
        except Exception as exc:
            raise MailError("imap_fetch_failed", "无法读取邮件头。", str(exc)) from exc
        headers: list[EmailHeader] = []
        for uid in uids:
            payload = fetched.get(int(uid), {})
            if not payload:
                continue
            raw = _find_fetch_value(payload, (b"BODY[HEADER", b"BODY.PEEK[HEADER"))
            headers.append(self._parse_header(uid, raw, payload.get(b"BODYSTRUCTURE")))
        return headers

    def get_content(
        self,
        account: AccountConfig,
        uid: str,
        *,
        folder: str | None = None,
        max_chars: int = 6000,
    ) -> EmailContent:
        client = self._connect(account)
        self._select(client, folder or account.folder)
        try:
            fetched = client.fetch(
                [int(uid)],
                [
                    HEADER_FETCH,
                    "BODYSTRUCTURE",
                ],
            )
            payload = fetched.get(int(uid), {})
        except Exception as exc:
            raise MailError("imap_fetch_failed", f"无法读取邮件 UID {uid}。", str(exc)) from exc
        if not payload:
            raise MailError("email_not_found", f"未找到邮件 UID {uid}。")
        header_raw = _find_fetch_value(payload, (b"BODY[HEADER", b"BODY.PEEK[HEADER"))
        structure = payload.get(b"BODYSTRUCTURE")
        header = self._parse_header(uid, header_raw, structure)
        parts = walk_bodystructure(structure)
        text_parts = [
            part
            for part in parts
            if part.content_type in {"text/plain", "text/html"} and part.disposition != "attachment"
        ]
        text_parts.sort(key=lambda item: (item.content_type != "text/plain", item.number or ""))
        texts: list[str] = []
        content_types: list[str] = []
        fetch_budget = max(4096, max_chars * 5)
        for part in text_parts[:3]:
            selector = "TEXT" if part.number is None else part.number
            try:
                body_data = client.fetch([int(uid)], [f"BODY.PEEK[{selector}]<0.{fetch_budget}>"])
                body_payload = body_data.get(int(uid), {})
                raw = _find_fetch_value(
                    body_payload,
                    (f"BODY[{selector}".encode().upper(), f"BODY.PEEK[{selector}".encode().upper()),
                )
            except Exception:
                continue
            text = _decode_body(raw, part)
            if text and text not in texts:
                texts.append(text)
                content_types.append(part.content_type)
            if sum(len(value) for value in texts) >= max_chars:
                break
        body = "\n\n".join(texts).strip()
        if not body:
            body = "（未找到可读取的纯文本或 HTML 正文；附件内容未下载。）"
        truncated = len(body) > max_chars or any(
            part.size is not None and part.size > fetch_budget for part in text_parts[:3]
        )
        body = body[:max_chars]
        return EmailContent(
            uid=uid,
            subject=header.subject,
            sender=header.sender,
            received_at=header.received_at,
            body=body,
            content_type=",".join(content_types) or "unavailable",
            truncated=truncated,
            attachments=header.attachments,
        )

    def close_all(self) -> None:
        for client in self._connections.values():
            with suppress(Exception):
                client.logout()
        self._connections.clear()
