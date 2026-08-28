from __future__ import annotations

import re
import string
from collections import Counter

_SPECIAL_ANSWERS = {"yes", "no", "noanswer"}


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(char for char in lowered if char not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match_score(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1_score(prediction: str, gold: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    if (
        normalized_prediction in _SPECIAL_ANSWERS
        or normalized_gold in _SPECIAL_ANSWERS
    ) and normalized_prediction != normalized_gold:
        return 0.0
    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    common = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_answer(prediction: str | None, gold: str) -> dict[str, float | bool]:
    if prediction is None:
        return {
            "exact_match": 0.0,
            "token_f1": 0.0,
            "answer_available": False,
        }
    return {
        "exact_match": exact_match_score(prediction, gold),
        "token_f1": token_f1_score(prediction, gold),
        "answer_available": True,
    }
