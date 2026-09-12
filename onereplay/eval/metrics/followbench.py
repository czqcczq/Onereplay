"""FollowBench metric: 820 multi-level constrained instructions (Jiang et al., ACL 2024).

FollowBench builds each instruction as a *ladder*: an initial instruction
(level 0, never generated) plus one added constraint per level up to level 5.
An `example_id` group is therefore 5 scored rows, and the reported numbers are
per level -- a model that satisfies level 1 but not level 4 has a shape, not
just a score.

This file is the generation half plus the rule-scored half. It does NOT call an
LLM judge, following the split direct_safety.py already established: 470 of the
820 rows can only be scored by a judge, which needs network access the compute
nodes do not have. Those rows are decoded here and written to judge_inputs.jsonl
with the prompt upstream would have sent; scripts/judge_followbench.py runs them
off-cluster and scripts/score_followbench.py merges the two halves.

Which half a row lands in is upstream's rule_based_source whitelist, not our
choice (see RULE_SOURCES):

    rule    350 rows / 70 groups   example 40, content 12, situation 8,
                                   mixed 8, format 2
    judge   470 rows / 94 groups   style 30, format 28, content 13,
                                   situation 14, mixed 9

No group mixes the two -- verified across all 164 -- which is what makes this
file useful on its own: HSR, SSR and CSL are all computable from the rule half
alone, because CSL needs a group's full level-1..5 ladder and every rule group
has one. The 70 rule groups are also the deterministic, reproducible part of
FollowBench, so they are the part worth trusting for a before/after comparison.

Denominators: upstream divides each level's satisfied count by the number of
*groups*, merging rule and judge groups per category. We only have the rule
half here, so every rate in summary.json is explicitly out of rule_groups. The
judge counts are recorded alongside so score_followbench.py can rebuild
upstream's combined denominator without re-reading the data files.

The scorers themselves come from onereplay/third_party/followbench, vendored at
a pinned commit -- see its README.md for what was changed (import lines only)
and for the upstream behaviours we deliberately do not reuse.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from onereplay.eval.generation import batched_generate, resolve_batch_size

THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

from followbench import gpt4_based_evaluation, rule_based_evaluation  # noqa: E402

# Copied from upstream, where the same list is repeated verbatim inside
# rule_evaluation, csl_five_constraint and acquire_discriminative_eval_input.
# It is a local variable in all three, so it cannot be imported; assert_scorable
# guards the copy by checking every source actually resolves to a checker.
RULE_SOURCES = frozenset(
    {
        "E2E",
        "WIKIEVENTS",
        "CONLL2003",
        "text_editing",
        "cnn_dailymail",
        "xsum",
        "samsum",
        "gigaword",
        "arxiv",
        "BBH_logical",
        "BBH_time",
        "self_made_space",
        "gsm_8k",
    }
)

# Two format-category ladders are hand-coded in rule_evaluation_format instead
# of going through a source-named checker. Upstream special-cases them by id.
FORMAT_RULE_EXAMPLE_IDS = frozenset({22, 30})

# example is last because it is the only category with no level 0 and the only
# one scored by exact template match; keeping it at the end makes the per-family
# tables read in increasing strictness.
CATEGORIES = ("content", "situation", "style", "format", "mixed", "example")

LEVELS = (1, 2, 3, 4, 5)

# Two example-category ladders are degenerate under upstream's check_match, in
# opposite directions. Reported in summary.json rather than excluded, so our
# numbers stay comparable to anything else scored with upstream's checker -- but
# they are 2 of that category's 40 groups, so read example HSR knowing that 2.5%
# is unreachable and another 2.5% is free.
#
#   example_id 4  (billsum)  template "{{'Response': '{answer}'}}" becomes the
#       regex \{'Response':\ '.*'\}, and check_match uses re.fullmatch, where
#       "." does not span newlines. The gold answer is a multi-line bill, so no
#       correct response can match. This ladder scores 0 for every model.
#   example_id 21 (limit)    template is bare "{answer}", so the regex is ".*"
#       and any single-line output passes. This ladder scores 5 for almost any
#       model, including one that ignores the constraint entirely.
DEGENERATE_EXAMPLE_LADDERS = {4: "unsatisfiable (multiline gold vs re.fullmatch '.')", 21: "vacuous (pattern is '.*')"}


@dataclasses.dataclass
class FollowBenchRow:
    category: str
    example_id: int
    level: int
    instruction: str
    source: str
    target: str
    mode: str  # "rule" or "judge"

    @property
    def key(self) -> str:
        return f"{self.category}/{self.example_id}/L{self.level}"


def default_data_dir() -> str:
    """The 820 English instructions, as vendored."""

    return str(THIRD_PARTY / "followbench" / "data")


def scoring_mode(category: str, example_id: int, source: str) -> str:
    """Reproduce upstream's rule-vs-judge split for one row."""

    if category == "example":
        return "rule"
    if source in RULE_SOURCES:
        return "rule"
    if category == "format" and example_id in FORMAT_RULE_EXAMPLE_IDS:
        return "rule"
    return "judge"


def read_rows(data_dir: str, categories: tuple[str, ...]) -> tuple[
    list[FollowBenchRow], dict[tuple[str, int], dict[int, str]]
]:
    """Return the scored rows plus each ladder's level -> instruction map.

    Level 0 is excluded from the returned rows because it is not generated, but
    its text is kept in the ladder map: the judge prompt replays the whole
    evolution path from level 0 up, which is the mechanism the paper credits for
    letting a judge pinpoint the single newly added constraint.
    """

    rows: list[FollowBenchRow] = []
    ladders: dict[tuple[str, int], dict[int, str]] = defaultdict(dict)
    for category in categories:
        path = Path(data_dir) / f"{category}_constraints.json"
        if not path.exists():
            raise FileNotFoundError(
                f"FollowBench data file not found: {path}. Expected the vendored "
                f"copy under {default_data_dir()}, or point --followbench_data at "
                "another checkout's data/ directory."
            )
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records:
            example_id = int(record["example_id"])
            level = int(record["level"])
            ladders[(category, example_id)][level] = record["instruction"]
            if level == 0:
                continue
            rows.append(
                FollowBenchRow(
                    # The file name, deliberately not record["category"]. In
                    # mixed_constraints.json that field holds the accumulated
                    # constraint types for the level ("format, content, style"),
                    # not "mixed", so it names no prompt builder and collides
                    # with the content file's rows if used as an identity.
                    # Upstream also drives off the file name (its
                    # constraint_type argument) everywhere it matters.
                    category=category,
                    example_id=example_id,
                    level=level,
                    instruction=record["instruction"],
                    source=str(record.get("source", "")),
                    # target carries the reference the rule checkers compare
                    # against; it is "" for judge rows and for the checkers that
                    # only inspect the response's shape.
                    target=str(record.get("target", "")),
                    mode=scoring_mode(category, example_id, str(record.get("source", ""))),
                )
            )
    return rows, dict(ladders)


def assert_scorable(rows: list[FollowBenchRow], ladders: dict[tuple[str, int], dict[int, str]]) -> None:
    """Fail before decoding rather than after.

    Every failure mode here costs a full generation pass to discover otherwise,
    and two of them would not raise at all -- they would quietly score a row as
    unsatisfied and depress the result.
    """

    missing_checkers = sorted(
        {
            row.source
            for row in rows
            if row.mode == "rule"
            and row.category != "example"
            and not (row.category == "format" and row.example_id in FORMAT_RULE_EXAMPLE_IDS)
            and not hasattr(rule_based_evaluation, f"rule_evaluation_{row.source}")
        }
    )
    if missing_checkers:
        raise RuntimeError(
            f"No FollowBench rule checker for sources {missing_checkers}. RULE_SOURCES "
            "in this file is a copy of upstream's rule_based_source list and has "
            "drifted from the vendored rule_based_evaluation.py."
        )

    missing_prompts = sorted(
        {
            row.category
            for row in rows
            if row.mode == "judge"
            and not hasattr(gpt4_based_evaluation, f"{row.category}_evaluation_prompt")
        }
    )
    if missing_prompts:
        raise RuntimeError(
            f"No FollowBench judge prompt builder for categories {missing_prompts}."
        )

    # A judge row at level n replays instructions 0..n. A hole in that range
    # would silently shorten the evolution path and shift every constraint the
    # judge is asked to count.
    incomplete = sorted(
        {
            f"{row.category}/{row.example_id}"
            for row in rows
            if row.mode == "judge"
            and any(level not in ladders[(row.category, row.example_id)] for level in range(row.level + 1))
        }
    )
    if incomplete:
        raise RuntimeError(
            f"Judge rows with a gap in their evolution path: {incomplete[:10]}. "
            "The data file is not the one this metric was written against."
        )

    # CSL counts consecutive satisfied levels from 1, so a partial ladder would
    # cap it below the model's real reach. Upstream asserts len == 5 too.
    short = sorted(
        {
            f"{category}/{example_id}"
            for (category, example_id), levels in ladders.items()
            if any(level not in levels for level in LEVELS)
        }
    )
    if short:
        raise RuntimeError(
            f"Ladders missing one of levels 1-5: {short[:10]}. CSL is undefined for "
            "these; if you subsampled, subsample whole example_id groups."
        )


def score_rule_row(row: FollowBenchRow, response: str) -> bool:
    """Dispatch one rule-scored row to the vendored checker upstream would use."""

    if row.category == "example":
        # target embeds the few-shot template; upstream strips the placeholder
        # for the instruction itself, leaving a pattern that check_match turns
        # into a full-string regex. Passing means the response is *only* the
        # requested structure -- no lead-in, no sign-off.
        template = row.target.replace("{instruction}\n", "")
        return bool(rule_based_evaluation.check_match(template, response))
    if row.category == "format" and row.example_id in FORMAT_RULE_EXAMPLE_IDS:
        return bool(
            rule_based_evaluation.rule_evaluation_format(response, row.example_id, row.level)
        )
    checker = getattr(rule_based_evaluation, f"rule_evaluation_{row.source}")
    return bool(checker(response, row.target, row.level))


def build_judge_prompt(
    row: FollowBenchRow, ladder: dict[int, str], response: str
) -> str:
    """Render the judge prompt for one row, through upstream's own builder.

    Built here rather than in the judging script so the prompt is pinned to the
    same data load that produced the response: the builder takes the evolution
    path by value, and a mismatch between the two would be invisible in the
    judge's reply.
    """

    evolve_instructions = [ladder[level] for level in range(row.level + 1)]
    builder = getattr(gpt4_based_evaluation, f"{row.category}_evaluation_prompt")
    return builder(evolve_instructions, response)


def aggregate_rule(
    results: list[tuple[FollowBenchRow, bool]]
) -> dict[str, dict[str, Any]]:
    """Per-category HSR, SSR and CSL over the rule-scored ladders.

    SSR equals HSR on this half by construction: a rule checker for level n
    re-verifies all n constraints and returns a single bool, so there is no
    partial credit to average. Upstream does the same -- save_discriminative_
    evaluation feeds the identical rule_result into both the HSR and the SSR
    column. The distinction only becomes real on the judge half, where the
    reply is a per-constraint YES/NO list.
    """

    satisfied: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    ladder_hits: dict[str, dict[int, dict[int, bool]]] = defaultdict(lambda: defaultdict(dict))
    for row, ok in results:
        satisfied[row.category][row.level] += int(ok)
        ladder_hits[row.category][row.example_id][row.level] = ok

    summary: dict[str, dict[str, Any]] = {}
    for category, per_example in ladder_hits.items():
        groups = len(per_example)
        hsr = {
            f"level_{level}": satisfied[category][level] / groups for level in LEVELS
        }
        consistent = []
        for levels in per_example.values():
            count = 0
            for level in LEVELS:
                if not levels[level]:
                    break
                count += 1
            consistent.append(count)
        summary[category] = {
            "groups": groups,
            "rows": groups * len(LEVELS),
            "hsr": hsr,
            "ssr": dict(hsr),
            "csl": sum(consistent) / groups,
        }
    return summary


def response_shape(responses: list[str]) -> dict[str, Any]:
    """Response length, which FollowBench scores are sensitive to.

    Part 1's IFBench diagnosis (results_log/2026-09-12) ended with a standing
    rule: never report an IF score without the response length next to it,
    because SFT shifts output length and several constraint families are really
    measuring length. Format and style constraints here are no different.
    """

    if not responses:
        return {"words_p50": 0, "words_p90": 0, "words_max": 0, "empty": 0}
    words = sorted(len(response.split()) for response in responses)
    return {
        "words_p50": words[len(words) // 2],
        "words_p90": words[min(int(len(words) * 0.9), len(words) - 1)],
        "words_max": words[-1],
        "empty": sum(1 for response in responses if not response.strip()),
    }


class FollowBenchMetric:
    name = "followbench"

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = cfg.get("run_name", "base")
        data_dir = cfg.get("followbench_data") or default_data_dir()
        categories = tuple(
            name.strip()
            for name in str(cfg.get("followbench_categories", ",".join(CATEGORIES))).split(",")
            if name.strip()
        )
        unknown = [name for name in categories if name not in CATEGORIES]
        if unknown:
            raise ValueError(f"Unknown FollowBench categories {unknown}; pick from {CATEGORIES}.")
        # Upstream decodes 2048. Format ladders ask for tables and multi-section
        # answers, and a response cut off at the cap fails every constraint it
        # had not reached yet, so a small budget reads as a low score.
        max_new_tokens = int(cfg.get("followbench_max_new_tokens", 2048))
        # Counted in example_id groups, never rows: the rule-side denominators
        # and CSL both assume a full level-1..5 ladder, and a row-level cut
        # would break that silently rather than raise.
        group_limit = int(cfg.get("followbench_group_limit", 0))

        rows, ladders = read_rows(data_dir, categories)
        if group_limit > 0:
            keep = set()
            per_category: dict[str, int] = defaultdict(int)
            for (category, example_id) in sorted(ladders):
                if per_category[category] < group_limit:
                    per_category[category] += 1
                    keep.add((category, example_id))
            rows = [row for row in rows if (row.category, row.example_id) in keep]
            ladders = {key: value for key, value in ladders.items() if key in keep}
        assert_scorable(rows, ladders)

        responses = batched_generate(
            model,
            tokenizer,
            [row.instruction for row in rows],
            device,
            max_new_tokens,
            # Not falling back to ifbench_batch_size: the example category
            # carries few-shot exemplars inline, so prompts here are far longer
            # than IFBench's and the KV cache a given batch needs is not the
            # same number.
            resolve_batch_size(cfg, "followbench_batch_size"),
            log_label="followbench",
        )

        # Paired by index. Upstream matches generations back to rows by exact
        # instruction string (utils.data_match_api_output) and only prints the
        # row number on a miss, which turns any normalisation of the prompt into
        # a silent scoring error.
        with (output_dir / "responses.jsonl").open("w", encoding="utf-8") as file:
            for row, response in zip(rows, responses):
                file.write(
                    json.dumps(
                        {
                            "key": row.key,
                            "category": row.category,
                            "example_id": row.example_id,
                            "level": row.level,
                            "source": row.source,
                            "mode": row.mode,
                            "instruction": row.instruction,
                            "response": response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        rule_results: list[tuple[FollowBenchRow, bool]] = []
        judge_rows: list[tuple[FollowBenchRow, str]] = []
        for row, response in zip(rows, responses):
            if row.mode == "rule":
                rule_results.append((row, score_rule_row(row, response)))
            else:
                judge_rows.append((row, response))

        with (output_dir / "eval_results_rule.jsonl").open("w", encoding="utf-8") as file:
            for row, ok in rule_results:
                file.write(
                    json.dumps(
                        {
                            "key": row.key,
                            "category": row.category,
                            "example_id": row.example_id,
                            "level": row.level,
                            "source": row.source,
                            "satisfied": ok,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        judge_prompt_words: list[int] = []
        with (output_dir / "judge_inputs.jsonl").open("w", encoding="utf-8") as file:
            for row, response in judge_rows:
                prompt = build_judge_prompt(row, ladders[(row.category, row.example_id)], response)
                judge_prompt_words.append(len(prompt.split()))
                file.write(
                    json.dumps(
                        {
                            "key": row.key,
                            "category": row.category,
                            "example_id": row.example_id,
                            "level": row.level,
                            "source": row.source,
                            # Named prompt_new to match upstream's field, so a
                            # reply file from their llm_eval.py can be scored by
                            # ours and vice versa.
                            "prompt_new": prompt,
                            "response": response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        per_category = aggregate_rule(rule_results)
        judge_groups: dict[str, set[int]] = defaultdict(set)
        for row, _ in judge_rows:
            judge_groups[row.category].add(row.example_id)

        rule_groups = sum(entry["groups"] for entry in per_category.values())
        overall_hsr = {}
        for level in LEVELS:
            hits = sum(int(ok) for row, ok in rule_results if row.level == level)
            overall_hsr[f"level_{level}"] = hits / rule_groups if rule_groups else 0.0
        overall_csl = (
            sum(entry["csl"] * entry["groups"] for entry in per_category.values()) / rule_groups
            if rule_groups
            else 0.0
        )

        example_summary = per_category.get("example", {})
        summary = {
            "run_name": run_name,
            "adapter_path": cfg.get("adapter_path", ""),
            "num_generated": len(rows),
            "rule_rows": len(rule_results),
            "rule_groups": rule_groups,
            "judge_rows": len(judge_rows),
            "judge_groups": sum(len(ids) for ids in judge_groups.values()),
            **{f"rule_hsr_{key}": value for key, value in overall_hsr.items()},
            "rule_csl": overall_csl,
            **{
                f"example_hsr_{key}": value
                for key, value in example_summary.get("hsr", {}).items()
            },
            "example_csl": example_summary.get("csl", 0.0),
            **{f"response_{key}": value for key, value in response_shape(responses).items()},
            "max_new_tokens": max_new_tokens,
            "categories": ",".join(categories),
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
        }

        detail = {
            **summary,
            "per_category_rule": per_category,
            "judge_pending": {
                category: {"groups": len(ids), "rows": len(ids) * len(LEVELS)}
                for category, ids in sorted(judge_groups.items())
            },
            # So the combined denominators can be checked without reloading the
            # data files: upstream's per-category rate is out of
            # rule_groups + judge_groups.
            "combined_groups_per_category": {
                category: per_category.get(category, {}).get("groups", 0) + len(judge_groups.get(category, ()))
                for category in categories
            },
            "judge_prompt_words": {
                "p50": int(statistics.median(judge_prompt_words)) if judge_prompt_words else 0,
                "max": max(judge_prompt_words) if judge_prompt_words else 0,
                "total": sum(judge_prompt_words),
            },
            # Expect 0/5 for ladder 4 and 5/5 for ladder 21 (see
            # DEGENERATE_EXAMPLE_LADDERS). Anything else means upstream changed
            # check_match or the data, and example HSR moved for that reason
            # rather than because the model did.
            "example_degenerate_ladders": {
                str(example_id): {
                    "why": why,
                    "satisfied_levels": sorted(
                        row.level
                        for row, ok in rule_results
                        if ok and row.category == "example" and row.example_id == example_id
                    ),
                }
                for example_id, why in DEGENERATE_EXAMPLE_LADDERS.items()
                if any(
                    row.category == "example" and row.example_id == example_id
                    for row, _ in rule_results
                )
            },
            "per_category_response_words_p50": {
                category: response_shape(
                    [response for row, response in zip(rows, responses) if row.category == category]
                )["words_p50"]
                for category in categories
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        summary_csv = Path(cfg.get("output_root", output_dir.parent)) / "followbench_summary.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        exists = summary_csv.exists()
        with summary_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(summary)

        print(
            f"followbench: scored {len(rule_results)} rule rows over {rule_groups} groups; "
            f"{len(judge_rows)} rows await scripts/judge_followbench.py",
            flush=True,
        )
        return summary
