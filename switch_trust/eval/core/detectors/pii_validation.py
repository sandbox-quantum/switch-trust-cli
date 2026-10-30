"""Rejects datafog regex matches that cannot be the PII their label claims."""

import ipaddress
import re

_DIGIT_BEFORE = re.compile(r"\d[.\-]?\Z")
_DIGIT_AFTER = re.compile(r"[.\-]?\d")
_GLUED_BEFORE = re.compile(r"[A-Za-z0-9]*\Z")
_GLUED_AFTER = re.compile(r"[A-Za-z0-9]*")
_HEX = re.compile(r"[0-9A-Fa-f]*")
_MIN_GLUED_BLOB = 8
_PHONE_SEPARATORS = re.compile(r"[\s().\-]")
_CARD_TAIL = re.compile(r"[ \-]\d{3}")


def _embedded_in_longer_token(text: str, start: int, end: int) -> bool:
    """True when digits continue the match, or it sits inside a hash or encoded blob.

    Letters glued to the match are allowed when they read as a label (SSN123-45-6789).
    """
    if _DIGIT_BEFORE.search(text, max(0, start - 2), start) or _DIGIT_AFTER.match(
        text, end
    ):
        return True
    glued = (
        _GLUED_BEFORE.search(text, max(0, start - 64), start).group()
        + _GLUED_AFTER.match(text, end, end + 64).group()
    )
    return len(glued) >= _MIN_GLUED_BLOB and (
        _HEX.fullmatch(glued) is not None or any(c.isdigit() for c in glued)
    )


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _known_issuer(digits: str) -> bool:
    n = len(digits)
    if digits[0] == "4":
        return n in (13, 16, 19)
    if 51 <= int(digits[:2]) <= 55 or 2221 <= int(digits[:4]) <= 2720:
        return n == 16
    if digits[:2] in ("34", "37"):
        return n == 15
    if digits[:4] == "6011" or digits[:2] == "65" or 644 <= int(digits[:3]) <= 649:
        return n in (16, 19)
    return False


def valid_credit_card(match: str) -> bool:
    digits = re.sub(r"\D", "", match)
    return 13 <= len(digits) <= 19 and _known_issuer(digits) and luhn_valid(digits)


def valid_ssn(match: str) -> bool:
    digits = re.sub(r"\D", "", match)
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    return (
        area not in ("000", "666")
        and area[0] != "9"
        and group != "00"
        and serial != "0000"
    )


def valid_phone(match: str) -> bool:
    if _PHONE_SEPARATORS.search(match):
        return True
    digits = match.lstrip("+")
    if match.startswith("+") and not digits.startswith("1"):
        return True
    if len(digits) == 11 and digits[0] == "1":
        digits = digits[1:]
    return len(digits) == 10 and digits[0] in "23456789" and digits[3] in "23456789"


def valid_ip(match: str) -> bool:
    try:
        ipaddress.ip_address(match.strip())
    except ValueError:
        return False
    return True


_NUMERIC_VALIDATORS = {
    "CREDIT_CARD": valid_credit_card,
    "SSN": valid_ssn,
    "PHONE": valid_phone,
}


def plausible_end(label: str, text: str, start: int, end: int) -> int | None:
    """End offset of the PII at text[start:end] tagged `label`, or None to drop it.

    A grouped 19-digit card is matched by datafog only up to its 16th digit.
    """
    match = text[start:end]
    if label == "EMAIL":
        return end
    if label == "IP_ADDRESS":
        return end if valid_ip(match) else None
    validator = _NUMERIC_VALIDATORS.get(label)
    if validator is None:
        return None
    if label == "CREDIT_CARD":
        tail = _CARD_TAIL.match(text, end)
        if (
            tail
            and match[4:5] == tail.group()[:1]
            and validator(text[start : tail.end()])
        ):
            end = tail.end()
            match = text[start:end]
    if validator(match) and not _embedded_in_longer_token(text, start, end):
        return end
    return None
