"""Transparent encryption for sensitive database text fields."""
from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator

from config import get_settings


class SensitiveDataConfigurationError(ValueError):
    """Raised when encrypted data is used without a valid configured key."""


def _fernet() -> Fernet | None:
    key = get_settings().data_encryption_key.strip()
    if not key:
        return None
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise SensitiveDataConfigurationError("DATA_ENCRYPTION_KEY is invalid") from exc


class EncryptedText(TypeDecorator[str]):
    """Encrypt new values while remaining able to read legacy plaintext rows."""

    impl = Text
    cache_ok = True
    _prefix = "enc:v1:"

    def process_bind_param(self, value: str | None, _dialect) -> str | None:
        if value is None or not value:
            return value
        cipher = _fernet()
        if cipher is None:
            if get_settings().environment.lower() in {"prod", "production"}:
                raise SensitiveDataConfigurationError("DATA_ENCRYPTION_KEY is required in production")
            return value
        return f"{self._prefix}{cipher.encrypt(value.encode('utf-8')).decode('ascii')}"

    def process_result_value(self, value: str | None, _dialect) -> str | None:
        if value is None or not value.startswith(self._prefix):
            return value
        cipher = _fernet()
        if cipher is None:
            raise SensitiveDataConfigurationError("DATA_ENCRYPTION_KEY is required to read encrypted data")
        try:
            return cipher.decrypt(value[len(self._prefix):].encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeDecodeError) as exc:
            raise SensitiveDataConfigurationError("encrypted sensitive data cannot be decrypted") from exc
