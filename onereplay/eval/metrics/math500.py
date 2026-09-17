"""MATH-500 metric with the Hendrycks MATH answer-equivalence check.

Unlike aime.py (integer equality), MATH answers are LaTeX -- fractions, roots,
units, matrices -- so scoring needs the official Hendrycks `is_equiv` string
normalization (fractions, sqrt, \\left/\\right, units, etc.). Using integer
equality here would systematically under-count correct answers and make the
retention numbers meaningless.

The predicted answer is the last \\boxed{...} in the model response; the gold
answer is the dataset's `answer` field (already extracted) or the boxed span in
its `solution`. Both are normalized and compared with `is_equiv`.

Three benchmarks share this file because they share the boxed-answer contract:
MATH-500, AMC, and Minerva Math. They differ in the data file, the output
directory, and the equivalence check. Only MATH-500 uses ``is_equiv`` alone:
AMC's golds are integers written as floats ("142.0") and Minerva's are measured
quantities ("4.5e33"), so both compare numbers instead -- see the block above
``to_number``, which records what string equality was costing them.

``math_num_samples`` > 1 decodes k independent responses per question and
reports their mean accuracy (average@k). That only measures something when
decoding is stochastic: with the default greedy policy the k samples are
identical and average@k is a slower way to get accuracy@1.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from onereplay.eval.generation import batched_generate, resolve_batch_size

QUESTION_KEYS = ("problem", "question", "prompt", "input")
ANSWER_KEYS = ("answer", "final_answer", "solution", "target", "output")


# --- Hendrycks MATH answer normalization (verbatim logic from the MATH repo) ---
def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post
                else:
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_int, b_int = int(a), int(b)
        if string != f"{a_int}/{b_int}":
            return string
        return "\\frac{" + str(a_int) + "}{" + str(b_int) + "}"
    except ValueError:
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(str1: str | None, str2: str | None) -> bool:
    """Hendrycks MATH answer equivalence via string normalization."""

    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:  # noqa: BLE001
        return str1 == str2


def last_boxed_only_string(string: str) -> str | None:
    """Return the last \\boxed{...} / \\fbox{...} span with balanced braces."""

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    depth = 0
    right = None
    for i in range(idx, len(string)):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
    return string[idx : right + 1] if right is not None else None


def remove_boxed(span: str | None) -> str | None:
    """Strip the \\boxed{...} wrapper, returning the inner answer text."""

    if span is None:
        return None
    for prefix in ("\\boxed{", "\\fbox{"):
        if span.startswith(prefix) and span.endswith("}"):
            return span[len(prefix) : -1]
    if span.startswith("\\boxed "):
        return span[len("\\boxed ") :]
    return span


def extract_answer(text: str) -> str | None:
    """Pull the final boxed answer from a solution or a model response."""

    return remove_boxed(last_boxed_only_string(text))


def build_prompt(question: str) -> str:
    """Render the eval prompt. Shared so budget probes measure the real thing."""

    return (
        "Solve the following math problem. Reason step by step, then give "
        "the final answer as \\boxed{...} at the end.\n\n"
        f"Problem: {question}"
    )


def load_json_records(path: str) -> list[dict[str, Any]]:
    """Load MATH-500 examples from jsonl or json."""

    data_path = Path(path)
    if data_path.suffix.lower() == ".jsonl":
        records = []
        with data_path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
        return records
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("test", "validation", "eval", "train", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def first_existing_key(example: dict, preferred: str, candidates: tuple[str, ...]) -> str:
    """Return the configured key or the first available candidate key."""

    if preferred:
        if preferred not in example:
            raise KeyError(f"Field {preferred!r} not found. Keys: {sorted(example)}")
        return preferred
    for key in candidates:
        if key in example:
            return key
    raise KeyError(f"None of {candidates} found. Keys: {sorted(example)}")


# --- numeric answers --------------------------------------------------------
# Two of the three sets here need a number, not a string, on both sides:
#
#   AMC     the golds arrive as floats from the source parquet, so they land in
#           the jsonl as "142.0". No model ever writes \boxed{142.0} for an
#           integer answer, and _strip_string does not touch a trailing ".0" --
#           so string equality scored **every** AMC answer wrong and the metric
#           reported a structural 0.000 no matter how the model did. Comparing
#           numerically is what makes the set scorable at all.
#   Minerva the golds are measured quantities ("4.5e33", "0.006", "41.8") from
#           the 272 MIT OpenCourseWare problems, given to a stated precision.
#           The same number has many correct spellings ("4.5 \times 10^{33}"),
#           and the last digit is a rounding choice, so it also needs a
#           tolerance -- see MinervaMathMetric.
#
# Either side that is not a number (symbolic answers like
# "\arcsin{1.3 \sin{\theta_w}}", or an expression the model boxed such as
# "342+103=445") falls through to is_equiv, which is the conservative direction:
# it can only refuse credit, never invent it.
_UNIT_MACROS = re.compile(r"\\(?:text|mathrm|mathbf|operatorname|hbox|mbox)\s*\{[^{}]*\}")
_SCI_NOTATION = re.compile(r"^([+-]?[\d.]+)\s*(?:\\times|\\cdot|\*|x)\s*10\s*\^\s*\{?([+-]?\d+)\}?$")
_BARE_POWER = re.compile(r"^([+-]?)10\s*\^\s*\{?([+-]?\d+)\}?$")
_FRACTION = re.compile(r"^\\d?frac\{([^{}]+)\}\{([^{}]+)\}$")


def _clean_numeric(text: str) -> str:
    """Strip the LaTeX decoration that never carries numeric meaning."""

    cleaned = text.strip()
    for token in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "$", "~"):
        cleaned = cleaned.replace(token, "")
    cleaned = _UNIT_MACROS.sub("", cleaned)
    cleaned = cleaned.replace("^{\\circ}", "").replace("^\\circ", "")
    cleaned = cleaned.replace("\\%", "").replace("%", "")
    cleaned = cleaned.replace(" ", "")
    # Thousands separators only; "1,2" style tuples are left alone so they fail
    # to parse and fall through to the string comparison.
    cleaned = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", cleaned)
    return cleaned.rstrip(".")


def to_number(text: str | None) -> float | None:
    """Best-effort float for a boxed answer; None when it is not a number."""

    if text is None:
        return None
    cleaned = _clean_numeric(text)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        pass
    for pattern, build in (
        (_SCI_NOTATION, lambda m: float(m.group(1)) * 10.0 ** int(m.group(2))),
        (_BARE_POWER, lambda m: (-1.0 if m.group(1) == "-" else 1.0) * 10.0 ** int(m.group(2))),
    ):
        match = pattern.match(cleaned)
        if match:
            try:
                return build(match)
            except (ValueError, OverflowError):
                return None
    match = _FRACTION.match(cleaned)
    if not match and cleaned.count("/") == 1:
        match = re.match(r"^([^/]+)/([^/]+)$", cleaned)
    if match:
        top, bottom = to_number(match.group(1)), to_number(match.group(2))
        if top is not None and bottom not in (None, 0.0):
            return top / bottom
    return None


def numbers_close(prediction: float, gold: float, rel_tol: float) -> bool:
    """Relative comparison. A gold of exactly zero has no relative scale, and
    "within 1% of zero" would accept any small number, so it demands equality."""

    if gold == 0.0:
        return prediction == 0.0
    return abs(prediction - gold) <= rel_tol * abs(gold)


class MATH500Metric:
    name = "math500"
    # Config keys checked in order for this metric's eval file. Subclasses (AMC,
    # Minerva) point at their own path so several boxed-answer sets can run in
    # one job without clobbering each other's output_dir / summary csv.
    data_path_keys = ("math500_data_path", "data_path")
    # Minerva turns this on: its scoring is numeric-tolerant, so the strict
    # string verdict is worth carrying alongside as a floor.
    report_strict_string = False
    # Opt-in, and off here on purpose. MATH-500 is reported as greedy@1 across
    # every existing table, so it must ignore math_num_samples even when the two
    # hard sets in the same job are decoding four samples each.
    allow_multi_sample = False

    def is_correct(self, prediction: str | None, gold: str, cfg: dict[str, Any]) -> bool:
        """Answer equivalence. Overridden where string equality is too strict."""

        return is_equiv(prediction, gold)

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = ""
        for key in self.data_path_keys:
            if cfg.get(key):
                data_path = cfg[key]
                break
        limit = int(cfg.get("limit", 0))
        max_new_tokens = int(cfg.get("math_max_new_tokens", cfg.get("max_new_tokens", 1024)))
        requested_samples = max(1, int(cfg.get("math_num_samples", 1) or 1))
        num_samples = requested_samples if self.allow_multi_sample else 1
        if requested_samples != num_samples:
            print(
                f"[{self.name}] math_num_samples={requested_samples} ignored: this "
                "benchmark is reported as a single greedy pass",
                flush=True,
            )
        run_name = cfg.get("run_name", "base")
        question_field = cfg.get("question_field", "")
        answer_field = cfg.get("answer_field", "")

        records = load_json_records(data_path)
        examples: list[dict[str, str]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            question_key = first_existing_key(record, question_field, QUESTION_KEYS)
            answer_key = first_existing_key(record, answer_field, ANSWER_KEYS)
            question = str(record[question_key]).strip()
            raw_answer = str(record[answer_key]).strip()
            # If the gold field is a full solution, pull its boxed answer.
            gold = extract_answer(raw_answer) or raw_answer
            if question and gold:
                examples.append({"question": question, "answer": gold})
            if limit > 0 and len(examples) >= limit:
                break

        # Sample-major order: response index s * n + i is sample s of question i.
        # Repeating the prompt list is enough because generate() draws for each
        # row independently, and batched_generate returns in input order.
        prompts = [build_prompt(example["question"]) for example in examples]
        responses = batched_generate(
            model,
            tokenizer,
            prompts * num_samples,
            device,
            max_new_tokens,
            resolve_batch_size(cfg, "math_batch_size"),
            log_label=self.name,
        )

        correct = 0
        strict_correct = 0
        per_sample_correct = [0] * num_samples
        solved_at_least_once = [False] * len(examples)
        response_path = output_dir / "responses.jsonl"
        with response_path.open("w", encoding="utf-8") as file:
            for flat_index, response in enumerate(responses):
                sample_index, index = divmod(flat_index, len(examples))
                example = examples[index]
                gold = example["answer"]
                pred = extract_answer(response)
                is_hit = self.is_correct(pred, gold, cfg)
                correct += int(is_hit)
                per_sample_correct[sample_index] += int(is_hit)
                solved_at_least_once[index] = solved_at_least_once[index] or is_hit
                row = {
                    "question": example["question"],
                    "gold": gold,
                    "prediction": pred,
                    "correct": is_hit,
                    "response": response,
                }
                if num_samples > 1:
                    row["sample_index"] = sample_index
                if self.report_strict_string:
                    strict_hit = is_equiv(pred, gold)
                    strict_correct += int(strict_hit)
                    row["correct_strict_string"] = strict_hit
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        scored = len(responses)
        summary = {
            "run_name": run_name,
            "adapter_path": cfg.get("adapter_path", ""),
            "data_path": data_path,
            "num_examples": len(examples),
            "num_scored": scored,
            "correct": correct,
            # With k samples this is average@k: correct answers over all k*n
            # decoded responses, i.e. the mean of the k per-sample accuracies.
            "accuracy": correct / max(scored, 1),
            "output_dir": str(output_dir),
        }
        # Keys are added only where they mean something. <metric>_summary.csv is
        # appended to with a header taken from these keys, so a run that emits
        # extra columns into a file written by an earlier run would misalign it.
        if self.report_strict_string:
            summary["accuracy_strict_string"] = strict_correct / max(scored, 1)
        if num_samples > 1:
            summary["num_samples"] = num_samples
            summary["accuracy_per_sample"] = "|".join(
                f"{hits / max(len(examples), 1):.4f}" for hits in per_sample_correct
            )
            # Not the reported number, but it separates "the model cannot do this
            # problem" from "it can, unreliably" -- which is the whole reason a
            # hard set is decoded k times instead of once.
            summary["pass_at_k"] = sum(solved_at_least_once) / max(len(examples), 1)
            summary["temperature"] = cfg.get("decode_temperature", 0.0)
            summary["max_new_tokens"] = max_new_tokens
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary_csv = Path(cfg.get("output_root", output_dir.parent)) / f"{self.name}_summary.csv"
        exists = summary_csv.exists()
        with summary_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(summary)
        return summary


class AMCMetric(MATH500Metric):
    """AMC competition set: integer answers, compared as numbers."""

    name = "amc"
    data_path_keys = ("amc_data_path", "data_path")
    # Hard enough that a single greedy pass mostly measures which side of one
    # problem the coin landed on, hence average@k.
    allow_multi_sample = True

    def is_correct(self, prediction: str | None, gold: str, cfg: dict[str, Any]) -> bool:
        """Exact numeric equality, **not** string equality.

        The golds come out of the source parquet as floats and are written to the
        jsonl as "142.0"; models write \\boxed{142}. _strip_string leaves both
        alone, so is_equiv said no to every single correct answer and this metric
        reported 0.000 for every run -- see the block above to_number(). No
        tolerance: these are exact integers, unlike Minerva's measurements.
        """

        if prediction is None:
            return False
        predicted_number = to_number(prediction)
        gold_number = to_number(gold)
        if predicted_number is not None and gold_number is not None:
            return predicted_number == gold_number
        return is_equiv(prediction, gold)


class MinervaMathMetric(MATH500Metric):
    """Minerva Math (math-ai/minervamath): 272 OCW problems, numeric answers.

    Same prompt and boxed-answer extraction as MATH-500 -- so the model sees the
    same contract across all three sets -- with numeric-tolerant scoring on top.
    """

    name = "minervamath"
    data_path_keys = ("minervamath_data_path", "data_path")
    allow_multi_sample = True
    report_strict_string = True

    def is_correct(self, prediction: str | None, gold: str, cfg: dict[str, Any]) -> bool:
        if prediction is None:
            return False
        rel_tol = float(cfg.get("minerva_rel_tol", 0.01) or 0.01)
        predicted_number = to_number(prediction)
        gold_number = to_number(gold)
        if predicted_number is not None and gold_number is not None:
            return numbers_close(predicted_number, gold_number, rel_tol)
        return is_equiv(prediction, gold)
