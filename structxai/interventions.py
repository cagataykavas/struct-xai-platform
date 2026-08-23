from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Intervention:
    name: str
    kind: Literal["delete", "replace"]
    start: int
    end: int
    replacement: str = ""

    def apply(self, text: str) -> str:
        if not 0 <= self.start <= self.end <= len(text):
            raise ValueError(
                f"invalid intervention span [{self.start}:{self.end}] for text length {len(text)}"
            )
        replacement = "" if self.kind == "delete" else self.replacement
        return text[: self.start] + replacement + text[self.end :]


def find_literal_span(text: str, literal: str) -> tuple[int, int]:
    start = text.find(literal)
    if start < 0:
        raise ValueError(f"literal {literal!r} not found in prompt")
    return start, start + len(literal)


def delete_literal(text: str, literal: str, name: str | None = None) -> Intervention:
    start, end = find_literal_span(text, literal)
    return Intervention(name or f"delete:{literal}", "delete", start, end)


def replace_literal(
    text: str,
    literal: str,
    replacement: str,
    name: str | None = None,
) -> Intervention:
    start, end = find_literal_span(text, literal)
    return Intervention(name or f"replace:{literal}->{replacement}", "replace", start, end, replacement)
