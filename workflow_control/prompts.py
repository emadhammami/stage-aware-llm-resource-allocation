from __future__ import annotations

from benchmark.hotpotqa import EvidenceDocument, HotpotTask


def hotpot_plan_prompt(task: HotpotTask) -> str:
    return f"""Plan deterministic local retrieval for this multi-hop question.
Question: {task.question}
Return two concise search intents. Do not answer the question.
"""


def _evidence_text(documents: tuple[EvidenceDocument, ...]) -> str:
    return "\n\n".join(f"[{item.title}]\n{item.text}" for item in documents)


def hotpot_answer_prompt(task: HotpotTask, documents: tuple[EvidenceDocument, ...]) -> str:
    return f"""Answer the question using only the retrieved evidence.
Question: {task.question}
Retrieved evidence:
{_evidence_text(documents)}
Return ANSWER followed by a concise evidence citation.
"""


def hotpot_verify_prompt(
    task: HotpotTask,
    documents: tuple[EvidenceDocument, ...],
    candidate_answer: str,
) -> str:
    return f"""Verify whether the candidate is fully supported by the retrieved evidence.
Question: {task.question}
Candidate: {candidate_answer}
Retrieved evidence:
{_evidence_text(documents)}
Return SUFFICIENT or INSUFFICIENT, then one short reason.
"""


def hotpot_revision_prompt(
    task: HotpotTask,
    documents: tuple[EvidenceDocument, ...],
    prior_answer: str,
) -> str:
    return f"""Revise the answer using only the expanded retrieved evidence.
Question: {task.question}
Prior candidate: {prior_answer}
Expanded evidence:
{_evidence_text(documents)}
Return ANSWER followed by a concise evidence citation.
"""
