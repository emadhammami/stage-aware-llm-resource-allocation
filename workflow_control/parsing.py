from __future__ import annotations

import re

_FINAL_ANSWER = re.compile(r"^FINAL_ANSWER:\s*(.+?)\s*$", re.IGNORECASE)
_EVIDENCE = re.compile(r"^EVIDENCE:\s*(.+?)\s*$", re.IGNORECASE)
_VERDICT = re.compile(r"^VERDICT:\s*(SUFFICIENT|INSUFFICIENT)\s*$", re.IGNORECASE)
_REASON = re.compile(r"^REASON:\s*(.+?)\s*$", re.IGNORECASE)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_final_answer(text: str) -> str | None:
    lines = _lines(text)
    if len(lines) != 2:
        return None
    answer_match = _FINAL_ANSWER.fullmatch(lines[0])
    evidence_match = _EVIDENCE.fullmatch(lines[1])
    if answer_match is None or evidence_match is None:
        return None
    answer = answer_match.group(1).strip()
    evidence = evidence_match.group(1).strip()
    return answer if answer and evidence else None


def parse_verdict(text: str) -> str | None:
    lines = _lines(text)
    if len(lines) != 2:
        return None
    verdict_match = _VERDICT.fullmatch(lines[0])
    reason_match = _REASON.fullmatch(lines[1])
    if verdict_match is None or reason_match is None or not reason_match.group(1).strip():
        return None
    return verdict_match.group(1).upper()
