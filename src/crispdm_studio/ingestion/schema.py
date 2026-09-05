"""Header normalization and lightweight schema utilities."""

import re
from collections import Counter

from crispdm_studio.exceptions import HeaderValidationError

_UNNAMED_RE = re.compile(r"^unnamed:\s*\d+$", re.IGNORECASE)


def normalize_header(value: object, position: int) -> str:
    header = str(value).strip().replace("\ufeff", "")
    if not header or _UNNAMED_RE.match(header):
        return f"column_{position + 1}"
    return re.sub(r"\s+", " ", header)


def make_headers_unique(headers: list[object]) -> tuple[list[str], dict[str, str], list[str]]:
    normalized = [normalize_header(value, index) for index, value in enumerate(headers)]
    if not normalized:
        raise HeaderValidationError("We couldn't analyze this file because no columns were detected.")

    counts: Counter[str] = Counter()
    unique: list[str] = []
    changes: dict[str, str] = {}
    warnings: list[str] = []

    for index, header in enumerate(normalized):
        counts[header] += 1
        final = header if counts[header] == 1 else f"{header}__{counts[header]}"
        original = str(headers[index])
        if final != original:
            changes[f"{index}:{original}"] = final
        unique.append(final)

    duplicates = sorted(name for name, count in Counter(normalized).items() if count > 1)
    if duplicates:
        warnings.append(
            "Duplicate column names were made unique: " + ", ".join(duplicates[:10])
            + (" …" if len(duplicates) > 10 else "")
        )

    blank_count = sum(1 for i, original in enumerate(headers) if normalized[i].startswith("column_") and (not str(original).strip() or _UNNAMED_RE.match(str(original).strip())))
    if blank_count:
        warnings.append(f"{blank_count} blank or unnamed column header(s) were assigned safe names.")

    return unique, changes, warnings
