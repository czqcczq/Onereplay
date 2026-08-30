"""Build the OpenCoder educational_instruct prompt pool for C_code.

Replaces prepare_magicoder_ccode.py. Magicoder rows are free-form (problem,
solution) pairs; educational_instruct additionally carries `entry_point` and a
`testcase` list of asserts, which makes it structurally closer to the two eval
sets -- MBPP is (text, test_list) and HumanEval is (signature + docstring).

Two prompt styles are emitted, in the spirit of prepare_metamath_cmath.py's
canonical/rephrased split:

  bare   the instruction verbatim, same as the Magicoder pool. Matches MBPP's
         "here is a task, write a function" shape.
  heval  an import/helper preamble, the real signature parsed out of `code`,
         and a docstring synthesized from the instruction. Matches HumanEval's
         "here is a stub, complete it" shape.

Neither style puts the asserts in the prompt. The eval harnesses are left
untouched, so build_mbpp_prompt still shows tests at eval time and this pool
still does not -- a deliberate gap, see --heval_ratio discussion below.

`testcase` is used only offline, as the correctness guarantee on the heval
rewrite: a row is converted only if the rewritten stub, with the original body
reinserted, still passes the asserts the row shipped with. Anything that fails
that check falls back to `bare` rather than entering the pool malformed.

Everything else is inherited from the Magicoder builder: dedup, word-level
8-gram decontamination against HumanEval/MBPP, prompt-length filtering, and a
fixed-seed uniform sample.

    python -m onereplay.scripts.prepare_opc_ccode \\
        --opc_path /path/datasets/code_replay/opc_educational_instruct.parquet \\
        --humaneval_data_file /path/datasets/code/humaneval_test.parquet \\
        --mbpp_dataset_path /path/datasets/code/mbpp_full --mbpp_split test \\
        --out_dir /path/datasets/OPC_views --prefix opc_edu_ccode_20k \\
        --num_samples 20000 --heval_ratio 0.5 --seed 1 \\
        --tokenizer_path /path/models/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
from pathlib import Path
from typing import Any

from onereplay.scripts.prepare_magicoder_ccode import (
    count_tokens,
    humaneval_docstrings,
    ngram_hashes,
    normalize,
    percentile,
)
from onereplay.scripts.prepare_metamath_cmath import DATA_SUFFIXES, find_data_files, load_local

INSTRUCTION_KEYS = ("instruction", "problem")
OUTPUT_KEYS = ("output", "response", "solution")
CODE_KEYS = ("code",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the OpenCoder C_code prompt pool.")
    parser.add_argument(
        "--opc_path",
        type=str,
        default="",
        help="Local parquet/jsonl/save_to_disk path. Empty falls back to the hub.",
    )
    parser.add_argument("--opc_repo", type=str, default="OpenCoder-LLM/opc-sft-stage2")
    parser.add_argument("--opc_config", type=str, default="educational_instruct")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="opc_edu_ccode_20k")
    parser.add_argument("--num_samples", type=int, default=20000, help="0 keeps the whole pool.")
    parser.add_argument(
        "--heval_ratio",
        type=float,
        default=0.5,
        help="Fraction of sampled rows to rewrite into HumanEval-style stubs. The "
        "assignment is random over the sampled rows, so both styles see the same "
        "problem distribution; rows whose rewrite fails verification fall back to "
        "bare, so the realized fraction is slightly lower. 0 reproduces the "
        "Magicoder pool's all-bare layout.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=1024,
        help="Drop problems longer than this before sampling. C is a token-mean so "
        "long prompts dominate it, and generate_replay_targets discards rows that "
        "leave no room for an answer inside --max_len.",
    )
    parser.add_argument(
        "--require_python_fence",
        type=int,
        default=1,
        help="1 requires ```python in the output field. educational_instruct is "
        "Python-only by construction, so this is a sanity check rather than the "
        "language filter it was for Magicoder.",
    )
    parser.add_argument(
        "--humaneval_data_file",
        type=str,
        default="",
        help="Contamination check against HumanEval; skipped when empty.",
    )
    parser.add_argument("--mbpp_dataset_path", type=str, default="")
    parser.add_argument("--mbpp_split", type=str, default="test")
    parser.add_argument(
        "--drop_contaminated",
        type=int,
        default=1,
        help="1 removes rows sharing an 8-gram with a HumanEval/MBPP item. Leave on: "
        "educational_instruct is 118k LLM-written 'Write a function to...' tasks "
        "against MBPP's 500 human-written ones, so near-duplicates are likelier "
        "here than they were with Magicoder.",
    )
    parser.add_argument(
        "--verify_exec",
        type=int,
        default=300,
        help="Execute the asserts on this many rewritten rows to measure how often "
        "the rewrite changes behaviour. Every rewrite is AST-verified regardless; "
        "this is the stronger, slower check on a random subset. 0 disables it, "
        "-1 checks every rewritten row.",
    )
    parser.add_argument("--verify_timeout", type=float, default=10.0)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=(
            str(Path(os.environ["MODEL_DIR"]) / os.environ.get("MODEL_NAME", "Qwen3-1.7B"))
            if os.environ.get("MODEL_DIR")
            else ""
        ),
        help="Model dir for token stats; defaults to $MODEL_DIR/$MODEL_NAME when set.",
    )
    parser.add_argument("--cache_dir", type=str, default="")
    return parser.parse_args()


def load_opc(args: argparse.Namespace):
    """Load educational_instruct from a local path or the hub."""

    from datasets import load_dataset, load_from_disk

    if not args.opc_path:
        return load_dataset(
            args.opc_repo,
            args.opc_config,
            split=args.split,
            cache_dir=args.cache_dir or None,
        )

    source = Path(args.opc_path)
    if source.is_dir():
        if (source / "dataset_info.json").exists() or (source / "dataset_dict.json").exists():
            dataset = load_from_disk(str(source))
            if hasattr(dataset, "keys"):
                dataset = dataset[args.split if args.split in dataset else next(iter(dataset))]
            return dataset
        fmt, files = find_data_files(source)
        print(f"loading {len(files)} {fmt} file(s) from {source}")
        return load_local(fmt, files)

    fmt = dict(DATA_SUFFIXES).get(source.suffix.lower())
    if fmt is None:
        raise ValueError(f"Unsupported OpenCoder file type: {source}")
    return load_local(fmt, [str(source)])


def pick_column(columns: list[str], candidates: tuple[str, ...], role: str) -> str:
    for name in candidates:
        if name in columns:
            return name
    raise SystemExit(f"OpenCoder {role} column not found; tried {candidates}, got {columns}")


def indent_block(text: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in text.split("\n"))


def make_docstring(instruction: str) -> str:
    """Render the instruction as a function docstring.

    HumanEval prompts end at an unclosed function whose only body is its
    docstring, so the wording the model conditions on has to live there.
    """

    body = str(instruction).strip().replace('"""', '\\"\\"\\"')
    return indent_block(f'"""\n{body}\n"""')


def find_target_function(tree: ast.Module, entry_point: str):
    """The top-level def matching entry_point, or None.

    Only module-level functions qualify: a method on a class cannot be turned
    into a HumanEval-style stub without also emitting the class, and the eval
    harness calls the entry point as a bare name.
    """

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            return node
    return None


def signature_line(node) -> str:
    """`def name(args) -> ret:` with decorators, reconstructed from the AST.

    entry_point alone is not enough: parameter names, defaults, annotations and
    async-ness all have to survive, or the completion will not match the tests.
    """

    parts = [f"@{ast.unparse(decorator)}" for decorator in node.decorator_list]
    keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    parts.append(f"{keyword} {node.name}({ast.unparse(node.args)}){returns}:")
    return "\n".join(parts)


def strip_docstring(node):
    """The function body with a leading docstring removed, if present."""

    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return body


_PREAMBLE_TYPES = (
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.AnnAssign,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


def keep_in_preamble(node) -> bool:
    """Whether a module-level statement belongs in the stub preamble.

    Whitelist, not blacklist. A HumanEval prompt shows only imports, module
    constants and helper definitions before the function it asks you to
    complete. Everything else in an OpenCoder `code` field is demo scaffolding
    that sits after the solution -- `print(...)` calls, and crucially top-level
    `assert` self-tests that reference the target by name. A blacklist that only
    dropped bare expressions let those asserts through into the preamble, where
    they run before the function is defined and raise NameError. Keeping a
    strict whitelist is the robust fix: anything that references the entry point
    lives after the stub, in the body, never before it.
    """

    return isinstance(node, _PREAMBLE_TYPES)


def build_heval_view(code: str, entry_point: str, instruction: str) -> tuple[str, str] | None:
    """Rewrite one row into (stub, reference_body); None when not convertible.

    The preamble keeps the module-level statements a HumanEval prompt would
    legitimately show -- imports, constants, helper defs -- because a completion
    that calls a helper we dropped would fail for a reason that has nothing to
    do with the model.     It keeps only imports, constants and
    helper defs via keep_in_preamble, dropping demo scaffolding (top-level
    prints, `assert` self-tests, the `__main__` guard) so the stub never
    references the entry point before defining it. Helpers originally defined
    after the target still work: Python resolves those names at call time.
    """

    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None

    node = find_target_function(tree, entry_point)
    if node is None:
        return None

    body = strip_docstring(node)
    if not body:
        return None

    preamble_nodes = [
        other for other in tree.body if other is not node and keep_in_preamble(other)
    ]
    preamble = "\n".join(ast.unparse(other) for other in preamble_nodes)

    stub_parts = []
    if preamble.strip():
        stub_parts.append(preamble.rstrip())
    stub_parts.append(signature_line(node))
    stub = "\n\n".join(stub_parts) + "\n" + make_docstring(instruction) + "\n"
    reference_body = indent_block("\n".join(ast.unparse(statement) for statement in body))

    # Cheap guarantee, applied to every row: the stub must be parseable once a
    # body is attached, and the reinserted body must reproduce a valid module.
    # This is also what catches an instruction whose text breaks out of the
    # docstring -- a trailing backslash, say, or an embedded triple quote.
    try:
        ast.parse(stub + "    pass\n")
        ast.parse(stub + reference_body + "\n")
    except (SyntaxError, ValueError):
        return None

    return stub, reference_body


def verify_by_execution(
    program: str, tests: list[str], timeout: float
) -> tuple[bool, str]:
    from onereplay.eval.code_exec import evaluate_assert_program

    return evaluate_assert_program(program, tests, timeout)


def build_eval_ngrams_by_source(args) -> tuple[dict[str, set[int]], dict[str, int]]:
    """Eval n-gram sets kept separate by signal, for attributing contamination.

    The union over the returned sets equals what prepare_magicoder_ccode's
    build_eval_ngrams produces, so the set of removed rows is unchanged -- this
    only lets the report say how much of a hit came from a specific signature
    (a HumanEval docstring or an MBPP assert, i.e. real overlap) versus the
    shared "Write a function to ..." task-description phrasing that MBPP text and
    educational_instruct have in common (boilerplate, a false-positive source).
    """

    sources: dict[str, set[int]] = {
        "humaneval_docstring": set(),
        "mbpp_assert": set(),
        "mbpp_text": set(),
    }
    counts = {"humaneval_signatures": 0, "mbpp_signatures": 0}

    if args.humaneval_data_file and Path(args.humaneval_data_file).is_file():
        docstrings = humaneval_docstrings(args.humaneval_data_file, args.cache_dir)
        counts["humaneval_signatures"] = len(docstrings)
        for text in docstrings:
            sources["humaneval_docstring"] |= ngram_hashes(text)
    else:
        print("跳过 HumanEval 污染自检：未给 --humaneval_data_file 或文件不存在")

    if args.mbpp_dataset_path and Path(args.mbpp_dataset_path).exists():
        from datasets import load_from_disk

        dataset_dict = load_from_disk(args.mbpp_dataset_path)
        dataset = (
            dataset_dict[args.mbpp_split]
            if args.mbpp_split in dataset_dict
            else dataset_dict
        )
        signatures = 0
        for i in range(len(dataset)):
            row = dict(dataset[i])
            text = str(row.get("text", ""))
            if normalize(text):
                sources["mbpp_text"] |= ngram_hashes(text)
                signatures += 1
            tests = row.get("test_list") or []
            if isinstance(tests, str):
                tests = [tests]
            for test in tests:
                if normalize(str(test)):
                    sources["mbpp_assert"] |= ngram_hashes(str(test))
                    signatures += 1
        counts["mbpp_signatures"] = signatures
    else:
        print("跳过 MBPP 污染自检：未给 --mbpp_dataset_path 或路径不存在")

    return sources, counts


def main() -> None:
    args = parse_args()
    dataset = load_opc(args)
    columns = list(dataset.column_names)
    print(f"loaded {len(dataset)} rows, columns={columns}")

    instruction_key = pick_column(columns, INSTRUCTION_KEYS, "instruction")
    output_key = pick_column(columns, OUTPUT_KEYS, "output")
    code_key = pick_column(columns, CODE_KEYS, "code")
    print(f"using columns: {instruction_key!r} / {output_key!r} / {code_key!r}")

    instructions = dataset[instruction_key]
    outputs = dataset[output_key]
    codes = dataset[code_key]
    entry_points = dataset["entry_point"] if "entry_point" in columns else [""] * len(dataset)
    testcases = dataset["testcase"] if "testcase" in columns else [[]] * len(dataset)

    stats: dict[str, Any] = {"rows_in": len(dataset)}

    # -- step 1: basic validity + Python sanity check -----------------------
    fence = args.require_python_fence == 1
    kept: list[int] = []
    dropped_empty = 0
    dropped_fence = 0
    for i in range(len(dataset)):
        if not str(instructions[i]).strip() or not str(codes[i]).strip():
            dropped_empty += 1
            continue
        if fence and "```python" not in str(outputs[i]).lower():
            dropped_fence += 1
            continue
        kept.append(i)
    stats["dropped_empty"] = dropped_empty
    stats["dropped_no_python_fence"] = dropped_fence
    stats["after_validity"] = len(kept)
    print(f"有效性过滤: 空字段去掉 {dropped_empty}，缺 ```python 去掉 {dropped_fence} -> 剩 {len(kept)}")

    # -- step 2: exact duplicates -------------------------------------------
    seen: set[str] = set()
    deduped: list[int] = []
    for i in kept:
        key = normalize(instructions[i])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(i)
    stats["dropped_duplicate"] = len(kept) - len(deduped)
    stats["after_dedup"] = len(deduped)
    print(f"去重: 去掉 {stats['dropped_duplicate']} 条重复题面 -> 剩 {len(deduped)}")

    # -- step 3: contamination against the eval sets ------------------------
    # Removal is unchanged: a row goes if it shares an 8-gram with the union of
    # all eval signatures. The per-source split is report-only, so the sampled
    # pool stays byte-identical -- it just tells apart real overlap (a matched
    # docstring or assert) from shared task-description boilerplate.
    sources, sig_counts = build_eval_ngrams_by_source(args)
    stats.update(sig_counts)
    eval_grams: set[int] = set().union(*sources.values()) if sources else set()
    specific = sources["humaneval_docstring"] | sources["mbpp_assert"]
    if eval_grams:
        per_source = {name: 0 for name in sources}
        contaminated: list[int] = []
        specific_hits = 0
        text_only = 0
        for i in deduped:
            grams = ngram_hashes(f"{instructions[i]}\n{codes[i]}")
            if not (grams & eval_grams):
                continue
            contaminated.append(i)
            for name, gramset in sources.items():
                if grams & gramset:
                    per_source[name] += 1
            if grams & specific:
                specific_hits += 1
            else:
                text_only += 1
        stats["contaminated_hits"] = len(contaminated)
        stats["contaminated_by_humaneval_docstring"] = per_source["humaneval_docstring"]
        stats["contaminated_by_mbpp_assert"] = per_source["mbpp_assert"]
        stats["contaminated_by_mbpp_text"] = per_source["mbpp_text"]
        stats["contaminated_specific"] = specific_hits
        stats["contaminated_text_only"] = text_only
        print(f"污染自检 (HumanEval+MBPP, 8-gram): !! {len(contaminated)} 条命中")
        print(
            f"  来源拆分（可重叠）: HumanEval docstring={per_source['humaneval_docstring']} "
            f"MBPP assert={per_source['mbpp_assert']} MBPP 题面={per_source['mbpp_text']}"
        )
        print(f"  特异信号命中（docstring 或 assert，真重叠估计）: {specific_hits}")
        print(f"  仅题面模板命中（boilerplate 假阳性估计）: {text_only}")
        if contaminated and args.drop_contaminated == 1:
            flagged = set(contaminated)
            deduped = [i for i in deduped if i not in flagged]
            print(f"  为保险全部剔除，剩 {len(deduped)} 条")
    stats["after_decontamination"] = len(deduped)

    # -- step 4: prompt length ----------------------------------------------
    prompt_tokens: dict[int, int] = {}
    if args.tokenizer_path:
        lengths = count_tokens([str(instructions[i]) for i in deduped], args.tokenizer_path)
        prompt_tokens = dict(zip(deduped, lengths))
        stats["prompt_tokens_p50"] = percentile(lengths, 50)
        stats["prompt_tokens_p90"] = percentile(lengths, 90)
        stats["prompt_tokens_p99"] = percentile(lengths, 99)
        stats["prompt_tokens_max"] = max(lengths) if lengths else 0
        print(
            f"题面 token: P50={stats['prompt_tokens_p50']} P90={stats['prompt_tokens_p90']} "
            f"P99={stats['prompt_tokens_p99']} max={stats['prompt_tokens_max']}"
        )
        if args.max_prompt_tokens > 0:
            before = len(deduped)
            deduped = [i for i in deduped if prompt_tokens[i] <= args.max_prompt_tokens]
            stats["dropped_too_long"] = before - len(deduped)
            print(
                f"长度过滤: > {args.max_prompt_tokens} token 去掉 "
                f"{stats['dropped_too_long']} -> 剩 {len(deduped)}"
            )
    else:
        print("未给 --tokenizer_path，跳过 token 统计与长度过滤")
    stats["after_length"] = len(deduped)

    # -- step 5: uniform random sample --------------------------------------
    rng = random.Random(args.seed)
    if args.num_samples > 0 and args.num_samples < len(deduped):
        selected = rng.sample(deduped, args.num_samples)
    else:
        if args.num_samples > len(deduped):
            print(
                f"!! 只剩 {len(deduped)} 条可用，少于请求的 {args.num_samples}。"
                "放宽 --max_prompt_tokens 或改小 --num_samples；"
                "别默默用一个更小的池，λ 会对不上。"
            )
        selected = list(deduped)
    selected.sort()
    stats["sampled"] = len(selected)

    # -- step 6: style assignment + heval rewrite ---------------------------
    # Assigning styles after sampling keeps both halves on the same problem
    # distribution. Picking the heval half from convertible rows only would
    # have loaded every class-based and multi-function problem into the bare
    # half, confounding style with problem shape.
    want_heval = int(round(args.heval_ratio * len(selected)))
    heval_targets = set(rng.sample(selected, want_heval)) if want_heval else set()

    views: dict[int, dict[str, str]] = {}
    rewrite_failures = 0
    for i in selected:
        if i not in heval_targets:
            views[i] = {"style": "bare", "inputs": str(instructions[i]), "targets": str(outputs[i])}
            continue
        built = build_heval_view(str(codes[i]), str(entry_points[i]), str(instructions[i]))
        if built is None:
            rewrite_failures += 1
            views[i] = {"style": "bare", "inputs": str(instructions[i]), "targets": str(outputs[i])}
            continue
        stub, reference_body = built
        views[i] = {"style": "heval", "inputs": stub, "targets": reference_body}
    stats["heval_requested"] = want_heval
    stats["heval_rewrite_failed"] = rewrite_failures
    print(f"heval 改造: 目标 {want_heval}，AST 校验失败 {rewrite_failures} 条退回 bare")

    # The length filter in step 4 measured the bare instruction. A heval stub is
    # that instruction plus a preamble and a signature, so it can overrun the
    # budget even when the instruction fits -- and an overlong prompt leaves no
    # room for the self-distilled answer, which generate_replay_targets then
    # writes as an empty truncated row that collect_cov drops. Falling back to
    # bare is always safe because bare already passed the filter.
    if args.tokenizer_path and args.max_prompt_tokens > 0:
        heval_indices = [i for i in selected if views[i]["style"] == "heval"]
        if heval_indices:
            stub_lengths = count_tokens(
                [views[i]["inputs"] for i in heval_indices], args.tokenizer_path
            )
            too_long = [i for i, n in zip(heval_indices, stub_lengths) if n > args.max_prompt_tokens]
            stats["heval_stub_tokens_p50"] = percentile(stub_lengths, 50)
            stats["heval_stub_tokens_p99"] = percentile(stub_lengths, 99)
            stats["heval_over_budget"] = len(too_long)
            print(
                f"heval stub token: P50={stats['heval_stub_tokens_p50']} "
                f"P99={stats['heval_stub_tokens_p99']}，超 {args.max_prompt_tokens} 的 "
                f"{len(too_long)} 条退回 bare"
            )
            for i in too_long:
                views[i] = {
                    "style": "bare",
                    "inputs": str(instructions[i]),
                    "targets": str(outputs[i]),
                }

    stats["heval_written"] = sum(1 for view in views.values() if view["style"] == "heval")
    stats["bare_written"] = sum(1 for view in views.values() if view["style"] == "bare")
    print(
        f"风格分布: heval {stats['heval_written']} / bare {stats['bare_written']}"
        f"（heval 实际占比 {stats['heval_written'] / max(len(selected), 1):.1%}）"
    )

    # -- step 7: execution check on the rewrite ------------------------------
    # The AST round-trip above proves the stub parses. This proves the rewrite
    # did not change behaviour: reinsert the original body and run the asserts
    # the row shipped with. Rows whose ORIGINAL code already fails its own
    # asserts are reported separately -- that is a dataset defect, not ours.
    heval_indices = [i for i in selected if views[i]["style"] == "heval"]
    if args.verify_exec != 0 and heval_indices:
        if args.verify_exec < 0 or args.verify_exec >= len(heval_indices):
            sample = heval_indices
        else:
            sample = random.Random(args.seed + 1).sample(heval_indices, args.verify_exec)
        broken: list[int] = []
        bad_source = 0
        for i in sample:
            tests = [str(test) for test in (testcases[i] or [])]
            if not tests:
                continue
            original_ok, _ = verify_by_execution(str(codes[i]), tests, args.verify_timeout)
            if not original_ok:
                bad_source += 1
                continue
            rebuilt = views[i]["inputs"] + views[i]["targets"] + "\n"
            rebuilt_ok, _ = verify_by_execution(rebuilt, tests, args.verify_timeout)
            if not rebuilt_ok:
                broken.append(i)
        stats["verify_exec_checked"] = len(sample)
        stats["verify_exec_source_already_failing"] = bad_source
        stats["verify_exec_broken_by_rewrite"] = len(broken)
        print(
            f"执行校验: 抽查 {len(sample)} 条，原始代码本身不过 assert 的 {bad_source} 条，"
            f"改造后行为变化 {len(broken)} 条"
        )
        if broken:
            flagged = set(broken)
            for i in flagged:
                views[i] = {
                    "style": "bare",
                    "inputs": str(instructions[i]),
                    "targets": str(outputs[i]),
                }
            stats["heval_written"] -= len(flagged)
            stats["bare_written"] += len(flagged)
            print(f"  这些行已退回 bare -> heval {stats['heval_written']} / bare {stats['bare_written']}")

    # -- step 8: write ------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.prefix}_python.jsonl"
    with out_path.open("w", encoding="utf-8") as file:
        for i in selected:
            view = views[i]
            file.write(
                json.dumps(
                    {
                        # No asserts in either style: the eval harnesses are
                        # left as they are, and build_mbpp_prompt does show
                        # tests at eval time, so the pool is deliberately the
                        # looser of the two.
                        "inputs": view["inputs"],
                        # Overwritten by generate_replay_targets with the base
                        # model's own answer; the dataset's solution never
                        # enters C.
                        "targets": view["targets"],
                        "style": view["style"],
                        "entry_point": str(entry_points[i]),
                        "source_index": i,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    stats["rows_out"] = len(selected)
    print(f"\n{stats['rows_out']} 行 -> {out_path}")

    if prompt_tokens:
        chosen = [prompt_tokens[i] for i in selected]
        stats["selected_prompt_tokens_p50"] = percentile(chosen, 50)
        stats["selected_prompt_tokens_p99"] = percentile(chosen, 99)
        stats["selected_prompt_tokens_total"] = sum(chosen)

    manifest = {
        "prefix": args.prefix,
        "source": args.opc_path or f"{args.opc_repo}:{args.opc_config}",
        "seed": args.seed,
        "num_samples_requested": args.num_samples,
        "heval_ratio": args.heval_ratio,
        "max_prompt_tokens": args.max_prompt_tokens,
        "require_python_fence": args.require_python_fence,
        "drop_contaminated": args.drop_contaminated,
        "verify_exec": args.verify_exec,
        "path": str(out_path),
        "stats": stats,
        "note": (
            "targets 是重建后的参考解答，仅作参照；generate_replay_targets 会用 base "
            "自蒸馏覆盖它，真正进入 C 的是 prompt + base 自己生成的解答。"
            "testcase 不进 prompt，只用于校验 heval 改造是否保持语义。"
        ),
    }
    manifest_path = out_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
