"""Input safety checks shared by chat and document-analysis flows."""
from __future__ import annotations

import re


class UnsafePromptError(ValueError):
    """Raised for high-confidence attempts to override application controls."""


_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"忽略.{0,12}(之前|以上).{0,12}(指令|规则)"),
    re.compile(r"输出.{0,12}(系统提示词|系统指令)"),
)


def contains_injection_pattern(text: str) -> bool:
    """Detect, without raising, an attempt to override application controls.

    Used for untrusted external content such as web snippets, where the right
    response is to quarantine and label the item instead of rejecting the
    user's whole question.
    """
    return any(pattern.search(str(text or "")) for pattern in _INJECTION_PATTERNS)


def validate_untrusted_text(text: str) -> None:
    """Block clear prompt override attempts while allowing ordinary questions."""
    if contains_injection_pattern(text):
        raise UnsafePromptError("输入包含不被允许的指令覆盖内容")
