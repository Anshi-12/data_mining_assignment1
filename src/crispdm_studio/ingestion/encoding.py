"""Encoding detection for uploaded CSV bytes."""

from charset_normalizer import from_bytes

from crispdm_studio.exceptions import EncodingDetectionError

COMMON_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def detect_encoding(data: bytes, sample_bytes: int) -> str:
    sample = data[:sample_bytes]
    if not sample:
        raise EncodingDetectionError()

    for encoding in COMMON_ENCODINGS[:2]:
        try:
            sample.decode(encoding, errors="strict")
            return encoding
        except UnicodeDecodeError:
            pass

    result = from_bytes(sample).best()
    if result and result.encoding:
        try:
            sample.decode(result.encoding, errors="strict")
            return result.encoding
        except (UnicodeDecodeError, LookupError):
            pass

    for encoding in COMMON_ENCODINGS[2:]:
        try:
            sample.decode(encoding, errors="strict")
            return encoding
        except UnicodeDecodeError:
            pass

    raise EncodingDetectionError()
