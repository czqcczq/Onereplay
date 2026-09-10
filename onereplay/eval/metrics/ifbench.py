"""IFBench metric: 300 out-of-domain verifiable constraints (Pyatkin et al., 2025).

Kept as its own metric instead of folded into ifeval.py. IFBench ships a
patched copy of the 25 classic IFEval checkers next to its 58 new ones, so
scoring the old 541 prompts through the new registry would quietly move numbers
we have already reported. Each dataset stays with the registry it was scored
under; ifeval.py keeps using third_party/instruction_following_eval.

The checkers come from the upstream package (see _INSTALL_HINT). It bundles
IFBench_test.jsonl but not the repo's evaluation_lib.py, so the strict/loose
scoring logic is reimplemented here; it was verified to agree with upstream's
scorer on all 300 prompts, both passes.

Reported metric in the paper is prompt-level loose accuracy.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from onereplay.eval.generation import batched_generate, resolve_batch_size

# Upstream's README claims the package is on PyPI; it is not, so we install
# from git. Pinned because the verification functions are the metric: an
# upstream fix to a checker would move our scores without anything in this
# repo changing.
IFBENCH_COMMIT = "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d"

_INSTALL_HINT = (
    "ifbench is not installed (it is not on PyPI, despite its README). On a\n"
    "login node with network access:\n"
    "    source .venv/bin/activate\n"
    '    pip install "ifbench @ git+https://github.com/allenai/IFBench.git@'
    f'{IFBENCH_COMMIT}"\n'
    "It pulls in emoji and syllapy, which the IFEval checkers never needed."
)

_NLTK_HINT = (
    "IFBench needs NLTK corpora that the IFEval checkers never touch, and it\n"
    "tries to nltk.download() them at import time -- which returns quietly\n"
    "without downloading anything on an offline compute node. Fetch them once\n"
    "on a login node:\n"
    "    export NLTK_DATA=<cache_root>/nltk_data\n"
    "    python -m nltk.downloader punkt punkt_tab stopwords"
    " averaged_perceptron_tagger_eng\n"
    "then forward the same NLTK_DATA into the job (SINGULARITYENV_NLTK_DATA)."
)


@dataclasses.dataclass
class IFBenchExample:
    key: str
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]


@dataclasses.dataclass
class IFBenchResult:
    key: str
    prompt: str
    response: str
    instruction_id_list: list[str]
    follow_instruction_list: list[bool]

    @property
    def follow_all_instructions(self) -> bool:
        return all(self.follow_instruction_list)


def load_registry() -> dict[str, Any]:
    """Return IFBench's instruction_id -> checker class mapping."""

    try:
        from ifbench import instructions_registry
    except ImportError as error:  # pragma: no cover - environment specific
        raise RuntimeError(_INSTALL_HINT) from error
    return instructions_registry.INSTRUCTION_DICT


def default_input_path() -> str:
    """Path to IFBench_test.jsonl as bundled in the installed wheel."""

    try:
        import ifbench
    except ImportError as error:  # pragma: no cover - environment specific
        raise RuntimeError(_INSTALL_HINT) from error
    return str(ifbench.data_path())


def check_nltk_data() -> None:
    """Fail before decoding rather than after.

    A missing corpus only surfaces when some checker reaches pos_tag or
    stopwords, which is after all 300 responses have been generated -- an hour
    of GPU time thrown away for a LookupError.
    """

    import nltk

    probes = (
        ("punkt", lambda: nltk.sent_tokenize("A sentence. And another one.")),
        ("punkt_tab", lambda: nltk.word_tokenize("hello world")),
        ("stopwords", lambda: nltk.corpus.stopwords.words("english")),
        ("averaged_perceptron_tagger_eng", lambda: nltk.pos_tag(["run", "fast"])),
    )
    for name, probe in probes:
        try:
            probe()
        except LookupError as error:
            raise RuntimeError(f"NLTK resource {name!r} is unavailable.\n{_NLTK_HINT}") from error


def read_examples(path: str) -> list[IFBenchExample]:
    examples: list[IFBenchExample] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            examples.append(
                IFBenchExample(
                    key=str(record["key"]),
                    prompt=record["prompt"],
                    instruction_id_list=list(record["instruction_id_list"]),
                    # Every row carries the union of all checker parameters with
                    # the irrelevant ones set to null; build_description would
                    # choke on those, so drop them here. Upstream does the same
                    # thing, but only inside the strict pass, which leaves the
                    # loose pass depending on strict having mutated the input
                    # first. Filtering once at load time removes that ordering
                    # trap without changing any verdict.
                    kwargs=[
                        {key: value for key, value in kwarg.items() if value is not None}
                        for kwarg in record["kwargs"]
                    ],
                )
            )
    return examples


def assert_registry_covers(examples: list[IFBenchExample], registry: dict[str, Any]) -> None:
    """Refuse to score constraints we have no checker for.

    Counting an unknown constraint as a failure would silently depress the
    score, and every id in the shipped test file is covered, so a miss here
    means the data file and the installed package disagree.
    """

    unknown = sorted(
        {
            instruction_id
            for example in examples
            for instruction_id in example.instruction_id_list
            if instruction_id not in registry
        }
    )
    if unknown:
        raise RuntimeError(
            f"No IFBench checker for {unknown}. The test file and the installed "
            "ifbench package disagree -- pip install -U ifbench, or point "
            "--ifbench_input at the file that matches the installed version."
        )


def build_checker(instruction_id: str, kwargs: dict[str, Any], prompt: str, registry):
    checker = registry[instruction_id](instruction_id)
    checker.build_description(**kwargs)
    args = checker.get_instruction_args()
    # A checker that takes the prompt as an argument (repeat_prompt and
    # friends) gets it injected here. The second build_description call drops
    # the other kwargs, which is what upstream does too.
    if args and "prompt" in args:
        checker.build_description(prompt=prompt)
    return checker


def loose_variants(response: str) -> list[str]:
    """The eight rewrites upstream accepts as "close enough".

    Models like to wrap answers in a lead-in line, a sign-off, or markdown
    bold; stripping those lets a response that satisfies the constraint in
    substance still count.
    """

    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


def score_example(
    example: IFBenchExample,
    response: str,
    registry: dict[str, Any],
    loose: bool,
) -> IFBenchResult:
    candidates = loose_variants(response) if loose else [response]
    follow_list: list[bool] = []
    for index, instruction_id in enumerate(example.instruction_id_list):
        checker = build_checker(instruction_id, example.kwargs[index], example.prompt, registry)
        follow_list.append(
            any(
                bool(candidate.strip()) and checker.check_following(candidate)
                for candidate in candidates
            )
        )
    return IFBenchResult(
        key=example.key,
        prompt=example.prompt,
        response=response,
        instruction_id_list=example.instruction_id_list,
        follow_instruction_list=follow_list,
    )


def write_results(path: Path, results: list[IFBenchResult]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(
                json.dumps(
                    {
                        "key": result.key,
                        "prompt": result.prompt,
                        "response": result.response,
                        "instruction_id_list": result.instruction_id_list,
                        "follow_instruction_list": result.follow_instruction_list,
                        "follow_all_instructions": result.follow_all_instructions,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def accuracy(results: list[IFBenchResult]) -> dict[str, float]:
    prompt_total = len(results)
    prompt_correct = sum(result.follow_all_instructions for result in results)
    instruction_total = sum(len(result.follow_instruction_list) for result in results)
    instruction_correct = sum(sum(result.follow_instruction_list) for result in results)
    return {
        "prompt_accuracy": prompt_correct / max(prompt_total, 1),
        "instruction_accuracy": instruction_correct / max(instruction_total, 1),
    }


def per_family_accuracy(results: list[IFBenchResult]) -> dict[str, dict[str, Any]]:
    """Instruction accuracy grouped by the id prefix (count:, format:, ...).

    300 prompts is too few to read a single constraint's number, but the
    families are large enough to show which kind of constraint a run lost.
    """

    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    for result in results:
        for instruction_id, followed in zip(
            result.instruction_id_list, result.follow_instruction_list
        ):
            family = instruction_id.split(":")[0]
            totals[family] += 1
            correct[family] += int(followed)
    return {
        family: {"total": totals[family], "accuracy": correct[family] / totals[family]}
        for family in sorted(totals)
    }


class IFBenchMetric:
    name = "ifbench"

    def run(self, model, tokenizer, device, cfg: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = cfg.get("run_name", "base")
        input_data = cfg.get("ifbench_input") or default_input_path()
        # Upstream decodes 4096 tokens. Constraints like "repeat the prompt" or
        # "one paragraph per letter of the alphabet" need real room, and a
        # response cut off at the cap fails every constraint it had not
        # satisfied yet, so a small budget shows up as a low score.
        max_new_tokens = int(cfg.get("ifbench_max_new_tokens", 2048))
        limit = int(cfg.get("ifbench_limit", cfg.get("limit", 0)))

        registry = load_registry()
        examples = read_examples(input_data)
        if limit > 0:
            examples = examples[:limit]
        assert_registry_covers(examples, registry)
        check_nltk_data()

        responses = batched_generate(
            model,
            tokenizer,
            [example.prompt for example in examples],
            device,
            max_new_tokens,
            # Deliberately not falling back to ifeval_batch_size: the token
            # budget here is ~3x IFEval's, so inheriting a batch tuned for
            # IFEval would size the KV cache off the wrong number.
            resolve_batch_size(cfg, "ifbench_batch_size"),
            log_label="ifbench",
        )

        # Paired with responses by index, not through a prompt -> response dict
        # the way upstream does it: IFBench reuses the same WildChat prompt
        # under different constraints, and a dict would collapse those rows
        # onto one response.
        with (output_dir / "responses.jsonl").open("w", encoding="utf-8") as file:
            for example, response in zip(examples, responses):
                file.write(
                    json.dumps(
                        {"key": example.key, "prompt": example.prompt, "response": response},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        strict_results = [
            score_example(example, response, registry, loose=False)
            for example, response in zip(examples, responses)
        ]
        loose_results = [
            score_example(example, response, registry, loose=True)
            for example, response in zip(examples, responses)
        ]
        write_results(output_dir / "eval_results_strict.jsonl", strict_results)
        write_results(output_dir / "eval_results_loose.jsonl", loose_results)

        strict = accuracy(strict_results)
        loose = accuracy(loose_results)
        summary = {
            "run_name": run_name,
            "adapter_path": cfg.get("adapter_path", ""),
            "num_prompts": len(examples),
            "strict_prompt_accuracy": strict["prompt_accuracy"],
            "strict_instruction_accuracy": strict["instruction_accuracy"],
            "loose_prompt_accuracy": loose["prompt_accuracy"],
            "loose_instruction_accuracy": loose["instruction_accuracy"],
            "max_new_tokens": max_new_tokens,
            "input_data": str(input_data),
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(
                {**summary, "per_family_loose": per_family_accuracy(loose_results)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        summary_csv = Path(cfg.get("output_root", output_dir.parent)) / "ifbench_summary.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        exists = summary_csv.exists()
        with summary_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(summary)
        return summary
