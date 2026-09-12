# Vendored FollowBench scorers

Source: <https://github.com/YJiangcm/FollowBench> at commit
`6278f4c1377b4eafab737267b8b21acd52ea0e52` (2025-06-12).

Paper: Jiang et al., *FollowBench: A Multi-level Fine-grained Constraints
Following Benchmark for Large Language Models*, ACL 2024.

Vendored rather than pip-installed for the same reason `ifbench` is pinned to a
commit in `requirements.txt`: **the checkers and the judge prompts are the
metric.** An upstream edit to `rule_evaluation_xsum` or to one word of
`format_evaluation_prompt` would move our numbers without anything in this repo
changing. `onereplay/eval/metrics/followbench.py` imports from here.

## What is here

| File | Used for |
| --- | --- |
| `rule_based_evaluation.py` | the 350 rule-scored rows: `rule_evaluation_*`, `check_match`, `check_format_30` |
| `gpt4_based_evaluation.py` | the 470 judge-scored rows: the five `*_evaluation_prompt` builders and `paring_discriminative_generation` |
| `utils.py` | imported by the two above; its `data_match_api_output` is **not** used (see below) |
| `data/*.json` | the 820 English instructions, unmodified |

Not vendored: `model_inference{,_vllm}.py` (needs vLLM and hardcodes ChatML;
we decode through `onereplay.eval.generation.batched_generate` so that all
three IF benchmarks share one rendering path), `llm_eval.py` and `eval.py`
(driver scripts replaced by `scripts/judge_followbench.py` and
`scripts/score_followbench.py`), and the Chinese `code_zh/` + `data_zh/`.

## Changes made to the upstream files

**No scoring logic was touched.** Every change is an import-line edit:

1. Added `__init__.py`, making this a package.
2. `from utils import ...` -> `from .utils import ...`
   (`rule_based_evaluation.py`, `gpt4_based_evaluation.py`). Upstream expects
   its `code/` directory on `sys.path`, which would put a module named `utils`
   in the top-level namespace -- too generic a name to squat on.
3. `from rule_based_evaluation import ...` -> `from .rule_based_evaluation ...`
   (`gpt4_based_evaluation.py`).
4. `import matplotlib.pyplot as plt` wrapped in `try/except ImportError`
   (both files). matplotlib is not a onereplay dependency and is only touched
   by the `save_*` helpers, which we do not call -- we write our own
   `summary.json` / `*_summary.csv` to match the other metrics. Calling a
   `save_*` function without matplotlib installed will fail on `plt is None`.

Verify with:

```bash
diff <(git -C /path/to/FollowBench show 6278f4c:code/rule_based_evaluation.py) \
     onereplay/third_party/followbench/rule_based_evaluation.py
```

## Upstream behaviour we deliberately do not reuse

Recorded here because each one is a trap that does not raise:

- **`data_match_api_output` pairs prompts to responses by exact string
  equality**, O(n^2), and on a miss only `print(i)`s before leaving the row
  without a `generation` key. Any strip or normalisation on the generation side
  silently unpairs everything. Our metric pairs by index. (The 820 rows contain
  no duplicate instruction strings, so upstream's matching is sound on the data
  itself -- the fragility is in the plumbing.)
- **`paring_discriminative_generation` returns `-1` when the judge's reply
  cannot be parsed**, and `discriminative_evaluation` adds that straight into
  the numerator -- one unparseable reply *subtracts* a point while leaving the
  denominator alone. Upstream logs `You must manually fix the evaluation.`
  Our scorer counts parse failures separately and refuses to report past a
  threshold.
- **Rule-side denominators come from `n_group = n_group // 5`.** All 164
  groups do carry levels 1-5 (verified), so this holds at full size, but it
  breaks silently under row-level truncation. Our metric only subsamples whole
  `example_id` groups, never rows.
- **The judge defaults to `gpt-4o-mini`** in upstream `llm_eval.py`, not the
  GPT-4 of the paper, so absolute scores are not comparable to the published
  leaderboard even before our other changes. Only before/after comparisons
  within our own runs mean anything.

## Data quirks worth knowing before reading a score

Found while wiring this up, all verified against the vendored data; the metric
reproduces upstream's behaviour rather than fixing it, so numbers stay
comparable, but they change how a result should be read.

- **`mixed_constraints.json` does not carry `"mixed"` in its `category` field.**
  It carries the accumulated constraint types for that level, e.g.
  `"format, content, content, style"`. So `category` there names no prompt
  builder, and using it as a row identity collides with `content`'s rows. Both
  upstream and our metric key off the *file name* instead. Only the mixed file
  is affected; the other five agree with their file names.
- **Two of the 40 `example` ladders are degenerate under `check_match`**, in
  opposite directions, together biasing that category's HSR by about +-2.5%:
  - `example_id 4` (billsum) is **unsatisfiable**. Its template
    `{{'Response': '{answer}'}}` becomes `\{'Response':\ '.*'\}`, matched with
    `re.fullmatch`, where `.` does not cross newlines -- and the gold answer is
    a multi-line bill. Even echoing the gold answer verbatim scores 0/5.
  - `example_id 21` (limit) is **vacuous**. Its template is a bare `{answer}`,
    so the pattern is `.*` and any single-line output passes all five levels.

  `followbench.py` records both in `summary.json` under
  `example_degenerate_ladders`, so if upstream ever fixes `check_match` the
  score shift has a visible cause.
- **A rule checker returns one bool for level n**, re-verifying all n
  constraints, so there is no partial credit and SSR equals HSR on the rule
  half. Upstream feeds the same `rule_result` into both columns. The two only
  diverge on the judge half, whose reply is a per-constraint YES/NO list.
