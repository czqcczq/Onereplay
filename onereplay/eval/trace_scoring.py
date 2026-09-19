"""Scorers for the eight TRACE tasks, reimplemented from the official metrics.py.

Kept separate from the metric classes so every formula here is testable without
a model, a GPU, or the benchmark files -- see test_code/check_trace_line.py.

Three of TRACE's four scoring dependencies are unavailable on this cluster:
``rouge`` (pltrdy), ``fuzzywuzzy``, and ``datasets.load_metric("sari")``, which
downloads a script at call time and so cannot work under HF_HUB_OFFLINE=1. Only
``nltk`` is installed. Each is replaced below by a direct implementation rather
than a near-equivalent package, so what is being computed is visible in this
file instead of pinned to a transitive dependency.

Where the replacement is exact, it says so. Two notes where it is not:

**ROUGE-L.** TRACE calls ``Rouge(metrics=["rouge-l"]).get_scores(str1, str2)``
with the target as ``str1`` and the prediction as ``str2``. pltrdy's signature is
``get_scores(hyps, refs)``, so TRACE is scoring with the two swapped, and that
library's F-measure uses beta=1.2, which is asymmetric -- the swap therefore
changes the number. Implemented here as the standard LCS F1 (beta=1), which is
symmetric and makes the argument order moot. Absolute values sit slightly off
the paper's; every comparison in this experiment is within-pipeline.

**Averaging.** TRACE divides the sum of per-row scores by ``len(results)``, not
by the number of rows it actually scored, while skipping rows whose prediction
or target is empty (metrics.py:37-47, 60-70, 126-135). An empty generation
therefore counts as a zero. That is reproduced exactly, because it is the part
of the protocol that makes "the model stopped answering" show up as a score
drop, which is precisely the signal this experiment is looking for.
"""

from __future__ import annotations

import difflib
import re
import warnings
from collections import Counter
from typing import Iterable, Sequence

# TRACE's own tokenizer: whitespace or a literal period, nothing else. Not a
# reasonable tokenizer, but it is the one its BLEU and nothing else was measured
# with, so BLEU numbers only mean what they mean under it.
_TOKEN_SPLIT = re.compile(r"\s|\.")

_PY150_LITERAL = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)

_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def trace_tokenize(text: str) -> list[str]:
    """Split on whitespace or '.', dropping empties. Verbatim from metrics.py:15."""

    return [token for token in _TOKEN_SPLIT.split(text) if len(token) > 0]


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------


def bleu_score(reference: str, hypothesis: str, gram: int) -> float:
    """Uniform-weight BLEU-n over TRACE's tokenization, no smoothing.

    nltk is the same library TRACE uses, so this is exact. The weights tuple has
    length ``gram`` with uniform mass, which is what metrics.py:21-34 spells out
    case by case.

    nltk warns on every row with no n-gram overlap, which for BLEU-4 on short
    answers is most of them; the warning is the expected outcome here, not a
    problem, so it is silenced rather than printed thousands of times.
    """

    from nltk.translate.bleu_score import sentence_bleu

    reference_tokens = trace_tokenize(reference)
    hypothesis_tokens = trace_tokenize(hypothesis)
    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    weights = tuple(1.0 / gram for _ in range(gram))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(sentence_bleu([reference_tokens], hypothesis_tokens, weights))
        except (ZeroDivisionError, ValueError):
            return 0.0


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Longest common subsequence length, two rolling rows.

    Two rows rather than a full table because MeetingBank references run to
    hundreds of tokens and this is called once per row per stage.
    """

    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0] * (len(right) + 1)
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def rouge_l(reference: str, hypothesis: str) -> float:
    """LCS-based ROUGE-L F1 in [0, 1]. Symmetric, so argument order is moot."""

    reference_tokens = trace_tokenize(reference)
    hypothesis_tokens = trace_tokenize(hypothesis)
    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    lcs = _lcs_length(reference_tokens, hypothesis_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hypothesis_tokens)
    recall = lcs / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# fuzzywuzzy's ratio
# ---------------------------------------------------------------------------


def fuzz_ratio(left: str, right: str) -> float:
    """``fuzzywuzzy.fuzz.ratio``, exactly.

    Not an approximation: with python-Levenshtein absent, fuzzywuzzy's ratio is
    ``int(round(100 * SequenceMatcher(None, s1, s2).ratio()))`` over the raw
    strings -- ``ratio`` does no preprocessing, unlike ``WRatio``. So this is the
    same number the official Py150 scorer produces.
    """

    if not left or not right:
        return 0.0
    return float(round(100 * difflib.SequenceMatcher(None, left, right).ratio()))


# ---------------------------------------------------------------------------
# Py150 literal restoration
# ---------------------------------------------------------------------------


def py150_postprocess(code: str) -> str:
    """Restore CodeXGLUE literal placeholders. From evaluations/eval_Py150.py:6-12."""

    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    for kind, literal in _PY150_LITERAL.findall(code):
        code = code.replace(f"<{kind}_LIT:{literal}>", literal)
    return code


# ---------------------------------------------------------------------------
# ScienceQA answer / reasoning split
# ---------------------------------------------------------------------------


def split_science_qa(text: str) -> tuple[str, str]:
    """First character is the choice, everything past the newline is the reasoning.

    From evaluations/eval_ScienceQA.py:6-13, which takes ``datium[0]`` and
    ``datium[2:]`` -- position 1 being the newline between them.
    """

    stripped = text.lstrip()
    if not stripped:
        return "", ""
    return stripped[0], stripped[2:] if len(stripped) > 2 else ""


# ---------------------------------------------------------------------------
# Lenient extraction, used alongside TRACE's strict equality
# ---------------------------------------------------------------------------


def extract_choice(text: str, labels: Iterable[str]) -> str:
    """First standalone occurrence of one of ``labels``, or "".

    "Standalone" means not embedded in a longer Latin word, which is the only
    thing that goes wrong in practice: a chat model answering C-STANCE writes
    "Absolutely not", and a naive substring search would read that as label A.
    Neighbouring CJK characters are fine, so "答案是C" resolves to C.
    """

    label_set = {label.upper() for label in labels}
    if not text:
        return ""
    stripped = text.strip()
    if stripped.upper() in label_set:
        return stripped.upper()

    for index, character in enumerate(stripped):
        upper = character.upper()
        if upper not in label_set:
            continue
        before = stripped[index - 1] if index > 0 else ""
        after = stripped[index + 1] if index + 1 < len(stripped) else ""
        if (before and before.isascii() and before.isalpha()) or (
            after and after.isascii() and after.isalpha()
        ):
            continue
        return upper
    return ""


def extract_number(text: str) -> str:
    """First number in the text with thousands separators removed, or "".

    NumGLUE golds are bare numbers, but a chat model volunteers a sentence
    around them, so strict equality alone cannot tell "got it wrong" from
    "stopped answering in the trained format".
    """

    match = _NUMBER.search(text or "")
    if not match:
        return ""
    return match.group(0).replace(",", "").lstrip("+")


def numbers_equal(left: str, right: str) -> bool:
    """Compare two numeric strings, falling back to string equality.

    ``12`` and ``12.0`` are the same answer; ``float`` says so and ``==`` on the
    strings does not.
    """

    if not left or not right:
        return False
    if left == right:
        return True
    try:
        return abs(float(left) - float(right)) < 1e-6
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# SARI
# ---------------------------------------------------------------------------


def _sari_ngram(
    source_grams: list[str],
    candidate_grams: list[str],
    reference_grams_list: list[list[str]],
) -> tuple[float, float, float]:
    """One n-gram order of SARI: (keep F1, delete precision, add F1).

    Follows Xu et al. (2016) and the reference implementation HuggingFace's
    ``sari`` metric wraps, including its asymmetries: deletion is scored by
    precision only, and addition is scored on the *set* of added n-grams rather
    than their counts.
    """

    number_of_references = len(reference_grams_list)
    reference_counter: Counter[str] = Counter()
    for reference_grams in reference_grams_list:
        reference_counter.update(reference_grams)

    source_counter = Counter(source_grams)
    candidate_counter = Counter(candidate_grams)
    # Counts are scaled by the reference count so that a source n-gram kept once
    # can be credited against every reference independently.
    source_repeated = Counter(
        {gram: count * number_of_references for gram, count in source_counter.items()}
    )
    candidate_repeated = Counter(
        {gram: count * number_of_references for gram, count in candidate_counter.items()}
    )

    keep_counter = source_repeated & candidate_repeated
    keep_good = keep_counter & reference_counter
    keep_all = source_repeated & reference_counter

    keep_precision_sum = 0.0
    keep_recall_sum = 0.0
    for gram in keep_good:
        keep_precision_sum += keep_good[gram] / keep_counter[gram]
        keep_recall_sum += keep_good[gram]
    keep_precision = keep_precision_sum / len(keep_counter) if keep_counter else 0.0
    keep_total = sum(keep_all.values())
    keep_recall = keep_recall_sum / keep_total if keep_total else 0.0
    keep_score = (
        2.0 * keep_precision * keep_recall / (keep_precision + keep_recall)
        if (keep_precision + keep_recall) > 0
        else 0.0
    )

    delete_counter = source_repeated - candidate_repeated
    delete_good = delete_counter - reference_counter
    delete_all = source_repeated - reference_counter
    delete_precision_sum = 0.0
    for gram in delete_good:
        if delete_counter[gram]:
            delete_precision_sum += delete_good[gram] / delete_counter[gram]
    delete_precision = delete_precision_sum / len(delete_counter) if delete_counter else 0.0

    add_candidates = set(candidate_counter) - set(source_counter)
    add_good = add_candidates & set(reference_counter)
    add_all = set(reference_counter) - set(source_counter)
    add_precision = len(add_good) / len(add_candidates) if add_candidates else 0.0
    add_recall = len(add_good) / len(add_all) if add_all else 0.0
    add_score = (
        2.0 * add_precision * add_recall / (add_precision + add_recall)
        if (add_precision + add_recall) > 0
        else 0.0
    )

    return keep_score, delete_precision, add_score


def _ngrams(tokens: Sequence[str], order: int) -> list[str]:
    return [" ".join(tokens[index : index + order]) for index in range(len(tokens) - order + 1)]


def _sari_normalize(text: str) -> list[str]:
    """Lowercase and split off punctuation, then tokenize on whitespace.

    SARI is defined over word n-grams and is sensitive to punctuation attached to
    words, so ``Haus.`` and ``Haus`` have to become the same token.
    """

    lowered = (text or "").lower()
    spaced = re.sub(r"([^\w\s])", r" \1 ", lowered, flags=re.UNICODE)
    return spaced.split()


def sari_sentence(source: str, candidate: str, references: Sequence[str]) -> float:
    """SARI for one row, averaged over n-gram orders 1..4, scaled to 0-100."""

    source_tokens = _sari_normalize(source)
    candidate_tokens = _sari_normalize(candidate)
    reference_tokens = [_sari_normalize(reference) for reference in references]
    if not reference_tokens:
        return 0.0

    keep_scores: list[float] = []
    delete_scores: list[float] = []
    add_scores: list[float] = []
    for order in (1, 2, 3, 4):
        keep, delete, add = _sari_ngram(
            _ngrams(source_tokens, order),
            _ngrams(candidate_tokens, order),
            [_ngrams(tokens, order) for tokens in reference_tokens],
        )
        keep_scores.append(keep)
        delete_scores.append(delete)
        add_scores.append(add)

    average = (
        sum(keep_scores) / len(keep_scores)
        + sum(delete_scores) / len(delete_scores)
        + sum(add_scores) / len(add_scores)
    ) / 3.0
    return 100.0 * average


def sari_score(
    sources: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[str],
) -> float:
    """Corpus SARI: the mean of per-row SARI, single reference per row.

    Denominator is ``len(predictions)``, matching how TRACE averages every other
    metric, so an empty generation costs a zero rather than being dropped.
    """

    if not predictions:
        return 0.0
    total = 0.0
    for source, prediction, reference in zip(sources, predictions, references):
        if not prediction.strip() or not reference.strip():
            continue
        total += sari_sentence(source, prediction, [reference])
    return total / len(predictions)


# ---------------------------------------------------------------------------
# Corpus-level helpers, with TRACE's len(results) denominator
# ---------------------------------------------------------------------------


def corpus_mean(values: Sequence[float], denominator: int) -> float:
    """Sum of scored rows over the total row count. See the module docstring."""

    if denominator <= 0:
        return 0.0
    return float(sum(values)) / denominator


def strict_accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """TRACE's ``caculate_accuracy``: full-string equality, no normalization."""

    if not predictions:
        return 0.0
    hits = 0
    for prediction, target in zip(predictions, targets):
        if prediction == "" or target == "":
            continue
        if prediction == target:
            hits += 1
    return hits / len(predictions)


__all__ = [
    "bleu_score",
    "corpus_mean",
    "extract_choice",
    "extract_number",
    "fuzz_ratio",
    "numbers_equal",
    "py150_postprocess",
    "rouge_l",
    "sari_score",
    "sari_sentence",
    "split_science_qa",
    "strict_accuracy",
    "trace_tokenize",
]
