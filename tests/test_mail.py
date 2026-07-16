from __future__ import annotations

from datetime import date

from mailweek import mail as mail_module
from mailweek.mail import HEADER_FETCH, MailService, walk_bodystructure
from mailweek.schemas import AccountConfig, EmailHeader

PLAIN_PART = (
    b"TEXT",
    b"PLAIN",
    (b"CHARSET", b"UTF-8"),
    None,
    None,
    b"7BIT",
    128,
    4,
    None,
    (b"INLINE", None),
)
ATTACHMENT_PART = (
    b"APPLICATION",
    b"PDF",
    (b"NAME", b"invoice.pdf"),
    None,
    None,
    b"BASE64",
    5000,
    None,
    (b"ATTACHMENT", (b"FILENAME", b"invoice.pdf")),
)
STRUCTURE = (PLAIN_PART, ATTACHMENT_PART, b"MIXED")
HEADER = (
    b"Subject: Weekly update\r\n"
    b"From: sender@example.com\r\n"
    b"To: me@example.com\r\n"
    b"Date: Tue, 14 Jul 2026 09:00:00 +0800\r\n"
    b"Message-ID: <one@example.com>\r\n\r\n"
)


class FakeSecrets:
    def get(self, _account):
        return "secret", "test"

    def set(self, _account, _password):
        return None

    def delete(self, _account):
        return None


class FakeIMAPClient:
    def __init__(self) -> None:
        self.selected: list[tuple[str, bool]] = []
        self.fetch_requests: list[list[str]] = []

    def noop(self) -> None:
        return None

    def login(self, _username: str, _password: str) -> None:
        return None

    def select_folder(self, folder: str, readonly: bool = False):
        self.selected.append((folder, readonly))
        return {b"EXISTS": 1}

    def search(self, criteria):
        self.criteria = criteria
        return [1]

    def fetch(self, uids, fields):
        self.fetch_requests.append(list(fields))
        if HEADER_FETCH in fields:
            return {
                1: {
                    (
                        b"BODY[HEADER.FIELDS (MESSAGE-ID SUBJECT FROM TO CC DATE CONTENT-TYPE)]"
                    ): HEADER,
                    b"BODYSTRUCTURE": STRUCTURE,
                }
            }
        if any("BODY.PEEK[1]" in field for field in fields):
            return {1: {b"BODY[1]<0>": b"Please review the attached invoice."}}
        raise AssertionError(f"unexpected fetch: {fields}")

    def logout(self) -> None:
        return None


def _service() -> tuple[MailService, FakeIMAPClient, AccountConfig]:
    service = MailService(FakeSecrets())
    fake = FakeIMAPClient()
    service._connections["work"] = fake  # noqa: SLF001 - isolate IMAP behavior
    account = AccountConfig(
        name="work",
        email="me@example.com",
        host="imap.example.com",
        username="me@example.com",
    )
    return service, fake, account


def test_walk_bodystructure_finds_attachment_without_payload() -> None:
    parts = walk_bodystructure(STRUCTURE)
    assert [part.number for part in parts] == ["1", "2"]
    assert parts[1].filename == "invoice.pdf"
    assert parts[1].disposition == "attachment"


def test_search_is_read_only_and_returns_metadata() -> None:
    service, fake, account = _service()
    result = service.search(
        account,
        date_from=date(2026, 7, 13),
        date_to=date(2026, 7, 14),
    )
    assert fake.selected == [("INBOX", True)]
    assert result.returned == 1
    assert result.emails[0].attachments[0].filename == "invoice.pdf"
    assert "SINCE" in fake.criteria


def test_get_content_uses_body_peek_and_never_fetches_attachment_part() -> None:
    service, fake, account = _service()
    result = service.get_content(account, "1")
    assert result.body == "Please review the attached invoice."
    assert all(
        not any("BODY.PEEK[2]" in field for field in request) for request in fake.fetch_requests
    )
    assert fake.selected == [("INBOX", True)]


def test_get_content_reuses_metadata_fetched_by_search() -> None:
    service, fake, account = _service()
    service.search(
        account,
        date_from=date(2026, 7, 13),
        date_to=date(2026, 7, 14),
    )

    result = service.get_content(account, "1")

    metadata_requests = [
        request for request in fake.fetch_requests if HEADER_FETCH in request
    ]
    assert result.body == "Please review the attached invoice."
    assert len(metadata_requests) == 1


def test_reconnect_discards_cached_metadata_for_account(monkeypatch) -> None:
    service, _old_client, account = _service()

    class DeadClient:
        def noop(self) -> None:
            raise ConnectionError("connection lost")

    replacement = FakeIMAPClient()
    monkeypatch.setattr(
        mail_module,
        "IMAPClient",
        lambda *_args, **_kwargs: replacement,
    )
    service._connections["work"] = DeadClient()  # type: ignore[assignment]  # noqa: SLF001
    service._metadata[("work", "INBOX", "1")] = (  # noqa: SLF001
        EmailHeader(uid="1", subject="stale subject", sender="stale@example.com"),
        STRUCTURE,
    )

    result = service.get_content(account, "1")

    metadata_requests = [
        request for request in replacement.fetch_requests if HEADER_FETCH in request
    ]
    assert result.subject == "Weekly update"
    assert len(metadata_requests) == 1
