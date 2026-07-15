from __future__ import annotations

from dataclasses import dataclass

from .config import ConfigStore, KeyringSecretStore
from .mail import MailService
from .ollama_client import OllamaAdapter
from .tools import RuntimeServices, SessionContext


@dataclass
class Runtime:
    services: RuntimeServices

    @classmethod
    def create(cls, *, config_store: ConfigStore | None = None) -> Runtime:
        store = config_store or ConfigStore()
        config = store.load()
        secrets = KeyringSecretStore()
        ollama = OllamaAdapter(config.ollama_host)
        mail = MailService(secrets)
        session = SessionContext(
            active_account=config.active_account,
            model=config.default_model,
        )
        return cls(
            services=RuntimeServices(
                config_store=store,
                secrets=secrets,
                mail=mail,
                ollama=ollama,
                session=session,
            )
        )

    def close(self) -> None:
        self.services.mail.close_all()
