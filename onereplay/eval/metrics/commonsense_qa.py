"""Commonsense reasoning accuracy over the eight LLM-Adapters test sets.

This is the accuracy counterpart to metrics/commonsense.py, which only reports
held-out SFT loss. Scored here are BoolQ, PIQA, SIQA, HellaSwag, WinoGrande,
ARC-e, ARC-c and OBQA, in the exact template Commonsense170k was built with, so
the training pool and these test sets are the train/test halves of the same
eight datasets. Read every number as in-domain task accuracy, not as a general
capability.

Input is LLM-Adapters' own dataset/<task>/test.json, whose rows already carry
the rendered instruction:

    {"instruction": "Please answer the following question with true or false,
                     question: ...\\n\\nAnswer format: true/false",
     "input": "", "output": "the correct answer is false", "answer": "false"}

--------------------------------------------------------------------------
Two scorers, and why the headline is the logprob one
--------------------------------------------------------------------------
LLM-Adapters scores by generating and regex-matching the label out of the
response. That measurement is not safe for a base model. Qwen3-1.7B-Base never
saw this template, so it does not answer in it -- it continues the prompt --
and `re.findall(r'true|false', response)[0]` then reads whichever word the
continuation happened to contain. The base score becomes a property of English
word frequency rather than of the model's commonsense, and the train-vs-base
delta inflates by however much of that floor was coincidence. Part 1's IFBench
result was the same failure mode: a base score that was really collisions in
long prose (see results_log/2026-09-12, section 8c).

So the default scorer is `logprob`: for each row, score every candidate
"the correct answer is <label>" as a continuation of the prompt and take the
argmax. It asks the model to rank a closed set rather than to obey a format, it
never returns "unparseable", and it is the same measurement for base and
trained. `gen` is still computed because it answers a different and also real
question -- did the model learn to *answer in the trained format* -- and its
`parse_rate` is what tells you whether its accuracy column means anything.

Both scorers run off one model load; the logprob pass costs one forward per
(row, candidate) pair over ~130-token sequences, which is minutes for all eight
sets.

--------------------------------------------------------------------------
Prompt rendering is the training prompt, verified
--------------------------------------------------------------------------
render_user_prompt applies the chat template with enable_thinking=False, which
makes Qwen3 emit an empty "<think>\\n\\n</think>\\n\\n" block before the
assistant's turn. apply_train_template's *full* text carries that same block,
because the template inserts it for the final assistant turn too. The training
prompt and the eval prompt are therefore byte-identical, and the supervised
span is the whole answer (7-8 tokens), not a suffix of it. This was measured on
Qwen3-1.7B-Base's own template rather than assumed -- prepare_openr1_math.py
documents the case where the two *do* diverge, and there the first 4 answer
tokens fall out of the loss. It does not happen here only because these answers
contain no "</think>".

--------------------------------------------------------------------------
Candidate sets come from the row, not from a table
--------------------------------------------------------------------------
The label set is parsed out of the instruction's own "Answer format: a/b/c"
tail. ARC rows carry four or five options depending on the question, so a
per-task table would silently score some rows against the wrong candidate set.
Parsing also means a ninth dataset in this template needs no code change.

Length normalization is deliberately not implemented: within one task every
candidate tokenizes to the same number of tokens (measured: true/false 5 each,
solution1/2 and answer1..5 and ending1..4 and option1/2 6 each), so dividing by
length cannot reorder them. acc_norm would be acc.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import torch

from onereplay.eval.generation import batched_generate, resolve_batch_size

# The eight sets LLM-Adapters ships, in the order the papers table them.
DEFAULT_TASKS = (
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Easy",
    "ARC-Challenge",
    "openbookqa",
)

ANSWER_FORMAT_RE = re.compile(r"answer format:\s*([A-Za-z0-9/ ]+)", re.IGNORECASE)

# Only used when a row's instruction has no parsable "Answer format:" tail.
FALLBACK_LABELS = {
    "boolq": ["true", "false"],
    "piqa": ["solution1", "solution2"],
    "social_i_qa": ["answer1", "answer2", "answer3"],
    "hellaswag": ["ending1", "ending2", "ending3", "ending4"],
    "winogrande": ["option1", "option2"],
    "ARC-Easy": ["answer1", "answer2", "answer3", "answer4", "answer5"],
    "ARC-Challenge": ["answer1", "answer2", "answer3", "answer4", "answer5"],
    "openbookqa": ["answer1", "answer2", "answer3", "answer4", "answer5"],
}

ANSWER_PREFIX = "the correct answer is "


def parse_labels(instruction: str, task: str) -> list[str]:
    """Read the candidate labels off the instruction's 'Answer format:' tail."""

    match = ANSWER_FORMAT_RE.search(instruction)
    if match:
        labels = [part.strip().lower() for part in match.group(1).split("/")]
        labels = [label for label in labels if label]
        if len(labels) >= 2:
            return labels
    return list(FALLBACK_LABELS.get(task, []))


def find_task_file(root: Path, task: str) -> Path | None:
    """Locate one task's test split under the eval root.

    LLM-Adapters lays the sets out as dataset/<task>/test.json; a flat
    <task>.json dump of the same rows is accepted too.
    """

    for candidate in (root / task / "test.json", root / f"{task}.json"):
        if candidate.is_file():
            return candidate
    return None


def load_task(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    """Read one test.json (a JSON array) or its JSONL equivalent."""

    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        rows = json.loads(stripped)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if limit > 0:
        rows = rows[:limit]
    return rows


def score_continuations(
    model,
    tokenizer,
    device,
    pairs: list[tuple[str, str]],
    batch_size: int,
    log_label: str = "",
    log_every: int = 20,
) -> tuple[list[float], int]:
    """Sum the log-probabilities of each continuation given its prompt.

    Returns the scores plus a count of rows where tokenizing prompt+continuation
    did not reproduce the prompt's own token ids. That is a BPE merge across the
    boundary; the span is recomputed from the real common prefix so the score
    stays correct, but a nonzero count means the two halves are not cleanly
    separable and is worth seeing in the summary.

    Padding is applied here explicitly and on the right. The generation path
    requires tokenizer.padding_side == "left" and shares this process, so this
    function must not touch that setting; scoring also needs real tokens to
    start at position 0, which right padding gives.
    """

    if not pairs:
        return [], 0

    encoded: list[tuple[list[int], int]] = []
    boundary_mismatches = 0
    for prompt_text, continuation in pairs:
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt_text + continuation, add_special_tokens=False)["input_ids"]
        boundary = len(prompt_ids)
        if full_ids[:boundary] != prompt_ids:
            boundary_mismatches += 1
            boundary = 0
            limit = min(len(full_ids), len(prompt_ids))
            while boundary < limit and full_ids[boundary] == prompt_ids[boundary]:
                boundary += 1
        # A continuation that tokenized away entirely cannot be scored; leaving
        # boundary at len(full_ids) - 1 keeps at least one scored token.
        boundary = min(max(boundary, 1), max(len(full_ids) - 1, 1))
        encoded.append((full_ids, boundary))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    # Length-bucketed like batched_generate_from_texts, so one long row does not
    # set the padded width for a whole batch.
    order = sorted(range(len(encoded)), key=lambda index: len(encoded[index][0]))
    scores = [0.0] * len(encoded)
    num_batches = (len(order) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(order), batch_size), start=1):
        chunk = order[start : start + batch_size]
        width = max(len(encoded[index][0]) for index in chunk)
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, index in enumerate(chunk):
            ids = encoded[index][0]
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        for row, index in enumerate(chunk):
            ids, boundary = encoded[index]
            # logits[t] predicts token t+1, so the answer's first token is
            # predicted at boundary - 1. Slicing before the float32 cast keeps
            # this to a few positions instead of the full [B, L, vocab] tensor.
            window = logits[row, boundary - 1 : len(ids) - 1].float()
            targets = torch.tensor(ids[boundary : len(ids)], device=device)
            token_logprobs = torch.log_softmax(window, dim=-1).gather(
                -1, targets.unsqueeze(-1)
            )
            scores[index] = float(token_logprobs.sum())

        if log_label and log_every > 0 and batch_number % log_every == 0:
            print(f"{log_label} scored batch {batch_number}/{num_batches}", flush=True)

    return scores, boundary_mismatches


def extract_label(response: str, labels: list[str]) -> str:
    """First candidate label mentioned in a generated response, else "".

    Longest-first so "solution1" is not shadowed by a shorter label that is a
    prefix of it, and so a label never matches inside another one.
    """

    lowered = response.lower()
    best_label = ""
    best_position = len(lowered) + 1
    for label in sorted(labels, key=len, reverse=True):
        position = lowered.find(label)
        if position != -1 and position < best_position:
            best_position = position
            best_label = label
    return best_label


class CommonsenseQAMetric:
    name = "commonsense_qa"

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        eval_dir = cfg.get("cs_eval_dir", "")
        if not eval_dir:
            raise ValueError(
                "commonsense_qa needs --cs_eval_dir pointing at LLM-Adapters' "
                "dataset/ directory (one subdirectory per task, each holding "
                "test.json)."
            )
        root = Path(eval_dir)
        if not root.is_dir():
            raise ValueError(f"--cs_eval_dir is not a directory: {root}")

        requested = cfg.get("cs_eval_tasks", "")
        tasks = (
            [name.strip() for name in str(requested).split(",") if name.strip()]
            if requested
            else list(DEFAULT_TASKS)
        )
        scoring = str(cfg.get("cs_scoring", "both")).lower()
        if scoring not in ("logprob", "gen", "both"):
            raise ValueError(f"--cs_scoring must be logprob, gen or both; got {scoring!r}")
        do_logprob = scoring in ("logprob", "both")
        do_gen = scoring in ("gen", "both")

        limit = int(cfg.get("cs_limit", 0) or cfg.get("limit", 0))
        max_new_tokens = int(cfg.get("cs_max_new_tokens", 32))
        gen_batch_size = resolve_batch_size(cfg, "cs_batch_size")
        # The logprob pass has a different constraint from decoding: the model
        # materializes a [batch, seq, vocab] logit tensor with no KV-cache reuse,
        # and Qwen3's vocab is 151936. Hence its own, smaller, default.
        score_batch_size = int(cfg.get("cs_score_batch_size", 8) or 8)
        run_name = cfg.get("run_name", "base")

        missing: list[str] = []
        loaded: list[tuple[str, list[dict[str, Any]]]] = []
        for task in tasks:
            path = find_task_file(root, task)
            if path is None:
                missing.append(task)
                continue
            loaded.append((task, load_task(path, limit)))
        if not loaded:
            raise ValueError(
                f"No test.json found under {root} for any of {tasks}. Expected "
                f"{root}/<task>/test.json."
            )
        if missing:
            print(f"commonsense_qa: skipping tasks with no test file: {missing}")

        per_task: dict[str, dict[str, Any]] = {}
        response_path = output_dir / "responses.jsonl"
        total_mismatches = 0
        with response_path.open("w", encoding="utf-8") as sink:
            for task, rows in loaded:
                per_task[task] = self._run_task(
                    model,
                    tokenizer,
                    device,
                    task,
                    rows,
                    sink,
                    do_logprob=do_logprob,
                    do_gen=do_gen,
                    max_new_tokens=max_new_tokens,
                    gen_batch_size=gen_batch_size,
                    score_batch_size=score_batch_size,
                )
                total_mismatches += per_task[task].pop("_boundary_mismatches", 0)

        summary = self._summarize(
            per_task,
            run_name=run_name,
            cfg=cfg,
            output_dir=output_dir,
            root=root,
            scoring=scoring,
            limit=limit,
            max_new_tokens=max_new_tokens,
            gen_batch_size=gen_batch_size,
            score_batch_size=score_batch_size,
            missing=missing,
            boundary_mismatches=total_mismatches,
        )
        self._write_csv(summary, cfg, output_dir)
        return summary

    def _run_task(
        self,
        model,
        tokenizer,
        device,
        task: str,
        rows: list[dict[str, Any]],
        sink,
        *,
        do_logprob: bool,
        do_gen: bool,
        max_new_tokens: int,
        gen_batch_size: int,
        score_batch_size: int,
    ) -> dict[str, Any]:
        """Score one task and stream its per-row records into responses.jsonl."""

        from onereplay.eval.generation import render_user_prompt

        instructions = [str(row.get("instruction", "")) for row in rows]
        golds = [str(row.get("answer", "")).strip().lower() for row in rows]
        candidates = [
            parse_labels(instruction, task) for instruction in instructions
        ]

        unlabeled = sum(1 for labels in candidates if len(labels) < 2)
        if unlabeled:
            print(
                f"commonsense_qa/{task}: {unlabeled} rows have no parsable "
                "'Answer format:' tail and no fallback label set; they are "
                "counted as wrong."
            )
        # A gold label outside the candidate set makes the row unscorable by
        # ranking: no candidate can ever match it.
        gold_off_menu = sum(
            1 for gold, labels in zip(golds, candidates) if labels and gold not in labels
        )
        if gold_off_menu:
            print(
                f"commonsense_qa/{task}: {gold_off_menu} rows have a gold answer "
                "outside their own candidate set; check the test file's template."
            )

        logprob_preds: list[str] = [""] * len(rows)
        boundary_mismatches = 0
        if do_logprob:
            pairs: list[tuple[str, str]] = []
            owners: list[tuple[int, str]] = []
            for index, (instruction, labels) in enumerate(zip(instructions, candidates)):
                prompt_text = render_user_prompt(tokenizer, instruction)
                for label in labels:
                    pairs.append((prompt_text, f"{ANSWER_PREFIX}{label}"))
                    owners.append((index, label))
            scores, boundary_mismatches = score_continuations(
                model,
                tokenizer,
                device,
                pairs,
                score_batch_size,
                log_label=f"{self.name}/{task}",
            )
            best: dict[int, tuple[float, str]] = {}
            for (index, label), score in zip(owners, scores):
                current = best.get(index)
                if current is None or score > current[0]:
                    best[index] = (score, label)
            for index, (_, label) in best.items():
                logprob_preds[index] = label

        gen_preds: list[str] = [""] * len(rows)
        responses: list[str] = [""] * len(rows)
        if do_gen:
            responses = batched_generate(
                model,
                tokenizer,
                instructions,
                device,
                max_new_tokens,
                gen_batch_size,
                log_label=f"{self.name}/{task}",
            )
            gen_preds = [
                extract_label(response, labels)
                for response, labels in zip(responses, candidates)
            ]

        logprob_correct = sum(
            1 for pred, gold in zip(logprob_preds, golds) if pred and pred == gold
        )
        gen_correct = sum(1 for pred, gold in zip(gen_preds, golds) if pred and pred == gold)
        parsed = sum(1 for pred in gen_preds if pred)

        for index, row in enumerate(rows):
            record = {
                "task": task,
                "instruction": instructions[index],
                "gold": golds[index],
                "candidates": candidates[index],
            }
            if do_logprob:
                record["logprob_pred"] = logprob_preds[index]
                record["logprob_correct"] = logprob_preds[index] == golds[index]
            if do_gen:
                record["gen_pred"] = gen_preds[index]
                record["gen_correct"] = bool(gen_preds[index]) and gen_preds[index] == golds[index]
                record["response"] = responses[index]
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            _ = row

        total = max(len(rows), 1)
        entry: dict[str, Any] = {
            "num_examples": len(rows),
            "rows_without_candidates": unlabeled,
            "rows_gold_off_menu": gold_off_menu,
            "_boundary_mismatches": boundary_mismatches,
        }
        if do_logprob:
            entry["acc_logprob"] = logprob_correct / total
            entry["correct_logprob"] = logprob_correct
            entry["random_baseline"] = (
                sum(1 / len(labels) for labels in candidates if labels) / total
            )
        if do_gen:
            # accuracy counts an unparseable response as wrong, which is
            # LLM-Adapters' convention; accuracy_on_parsed and parse_rate are
            # what separate "got it wrong" from "did not answer in the format".
            entry["acc_gen"] = gen_correct / total
            entry["correct_gen"] = gen_correct
            entry["parse_rate"] = parsed / total
            entry["acc_gen_on_parsed"] = gen_correct / max(parsed, 1)
        print(
            f"commonsense_qa/{task}: n={len(rows)}"
            + (f"  acc_logprob={entry.get('acc_logprob', 0):.4f}" if do_logprob else "")
            + (
                f"  acc_gen={entry.get('acc_gen', 0):.4f}"
                f"  parse_rate={entry.get('parse_rate', 0):.3f}"
                if do_gen
                else ""
            ),
            flush=True,
        )
        return entry

    def _summarize(
        self,
        per_task: dict[str, dict[str, Any]],
        *,
        run_name: str,
        cfg: dict[str, Any],
        output_dir: Path,
        root: Path,
        scoring: str,
        limit: int,
        max_new_tokens: int,
        gen_batch_size: int,
        score_batch_size: int,
        missing: list[str],
        boundary_mismatches: int,
    ) -> dict[str, Any]:
        """Macro-average across tasks, which is the 'Average' column in the papers."""

        def macro(key: str) -> float | None:
            values = [entry[key] for entry in per_task.values() if key in entry]
            return sum(values) / len(values) if values else None

        num_examples = sum(entry["num_examples"] for entry in per_task.values())
        summary: dict[str, Any] = {
            "run_name": run_name,
            "adapter_path": cfg.get("adapter_path", ""),
            "eval_dir": str(root),
            "tasks": sorted(per_task),
            "tasks_missing": missing,
            "scoring": scoring,
            "num_examples": num_examples,
            "limit": limit,
            "max_new_tokens": max_new_tokens,
            "gen_batch_size": gen_batch_size,
            "score_batch_size": score_batch_size,
            "boundary_mismatches": boundary_mismatches,
            "per_task": per_task,
            "output_dir": str(output_dir),
            "note": "acc_logprob ranks 'the correct answer is <label>' over the "
            "row's own candidate set and is the headline; acc_gen regex-matches a "
            "generated answer and is only readable next to parse_rate. Averages "
            "are macro (unweighted mean over tasks), matching the papers. These "
            "test sets are the held-out halves of the datasets Commonsense170k "
            "was built from, so they measure in-domain task accuracy.",
        }
        for key, out_key in (
            ("acc_logprob", "average_acc_logprob"),
            ("acc_gen", "average_acc_gen"),
            ("parse_rate", "average_parse_rate"),
            ("random_baseline", "average_random_baseline"),
        ):
            value = macro(key)
            if value is not None:
                summary[out_key] = value

        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    def _write_csv(
        self, summary: dict[str, Any], cfg: dict[str, Any], output_dir: Path
    ) -> None:
        """Append one row per task plus an AVERAGE row.

        Other metrics write a single row whose columns are the summary keys, but
        this metric has eight subsets and that shape would either stringify a
        dict into one cell or grow a column per task. One row per (run, task)
        keeps the columns fixed and stays sortable across runs.
        """

        fieldnames = [
            "run_name",
            "adapter_path",
            "task",
            "num_examples",
            "acc_logprob",
            "acc_gen",
            "parse_rate",
            "acc_gen_on_parsed",
            "random_baseline",
            "scoring",
        ]
        rows = []
        for task in summary["tasks"]:
            entry = summary["per_task"][task]
            rows.append(
                {
                    "run_name": summary["run_name"],
                    "adapter_path": summary["adapter_path"],
                    "task": task,
                    "num_examples": entry["num_examples"],
                    "acc_logprob": entry.get("acc_logprob", ""),
                    "acc_gen": entry.get("acc_gen", ""),
                    "parse_rate": entry.get("parse_rate", ""),
                    "acc_gen_on_parsed": entry.get("acc_gen_on_parsed", ""),
                    "random_baseline": entry.get("random_baseline", ""),
                    "scoring": summary["scoring"],
                }
            )
        rows.append(
            {
                "run_name": summary["run_name"],
                "adapter_path": summary["adapter_path"],
                "task": "AVERAGE",
                "num_examples": summary["num_examples"],
                "acc_logprob": summary.get("average_acc_logprob", ""),
                "acc_gen": summary.get("average_acc_gen", ""),
                "parse_rate": summary.get("average_parse_rate", ""),
                "acc_gen_on_parsed": "",
                "random_baseline": summary.get("average_random_baseline", ""),
                "scoring": summary["scoring"],
            }
        )

        summary_csv = Path(cfg.get("output_root", output_dir.parent)) / "commonsense_qa_summary.csv"
        exists = summary_csv.exists()
        with summary_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
