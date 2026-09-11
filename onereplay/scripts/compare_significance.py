"""Can this benchmark tell these two adapters apart?

Every retention number in the project so far is one seed on 541 IFEval prompts,
and the headline claim rests on OneReplay beating EWC by 0.18pp -- a single
prompt. This script puts a resolution limit on such comparisons before they get
read as findings.

Three questions, kept apart because they have different answers:

  1. Evaluation resolution, paired. Two adapters are scored on the *same*
     prompts, so the unpaired binomial SE (2.04pp at n=541, p=0.66) overstates
     the uncertainty of their difference. What carries information is the
     discordant pairs: prompts one adapter got right and the other wrong.
     McNemar's exact test uses exactly those, and the paired CI half-width is
     the honest resolution of a single-seed comparison. IFEval decodes greedily,
     so re-scoring one adapter is deterministic and this term is entirely about
     which prompts differ -- not about generation sampling.

  2. Training noise, across seeds. Re-running an arm under a different seed
     produces a different adapter, and no amount of evaluation data shrinks the
     spread between them. If one arm's between-seed spread exceeds the gap
     between two arms, the ranking is not a property of the methods. This has
     never been measured here, which is why every seed-1 ranking in the logs is
     currently unfalsifiable.

  3. Instruction-level accuracy is not prompt-level. Instructions are nested
     inside prompts, so treating them as independent Bernoulli draws understates
     the variance. Those columns get a cluster bootstrap that resamples prompts.

The verdict block at the end is the point: a gap must clear both the paired CI
and the between-seed spread before it means anything. When it cannot, the script
reports how many seeds the observed gap *would* need, which is usually the
number that settles whether a comparison is worth pursuing at all.

Usage
  # one seed, two arms: just the evaluation resolution
  python -m onereplay.scripts.compare_significance \
      --results_root /scratch/.../results \
      --arm "OneReplay=cs_onereplay_lam3e-2_seed{seed}" \
      --arm "EWC=cs_ewc_lam3e2_seed{seed}" \
      --seeds 1

  # the full picture once 66_seed_variance.pbs has produced seeds 2 and 3
  python -m onereplay.scripts.compare_significance \
      --results_root /scratch/.../results \
      --arm "OneReplay=cs_onereplay_uniform_lam3e-2_seed{seed}_regonce" \
      --arm "EWC=cs_ewc_lam3e2_seed{seed}" \
      --arm "Replay=cs_replaymix_4n4r_seed{seed}" \
      --arm "Vanilla=cs_vanilla_seed{seed}" \
      --seeds 1,2,3

  # IFBench, whose headline number is prompt-level loose accuracy
  python -m onereplay.scripts.compare_significance --bench ifbench --mode loose ...

  # Multi-IF instead, pooling the three turns (turn 3 is where OneReplay loses)
  python -m onereplay.scripts.compare_significance --bench multiif ... --by_turn 1

  # extra result trees are searched in order, for runs that landed elsewhere
  python -m onereplay.scripts.compare_significance \
      --results_root .../results --also_root .../results_safety ...
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from itertools import combinations
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import random  # noqa: E402

# Two-sided 95% t quantiles. Three seeds give df=2 and a critical value of 4.30,
# which is worth seeing spelled out: with K=3 the between-seed term can only
# certify gaps several times the observed spread.
T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}
Z_95 = 1.959964
# 1.96 + 0.8416: the usual two-sided-95% / 80%-power constant.
POWER_CONST = 2.8016


def t_crit_95(df: int) -> float:
    if df < 1:
        return float("inf")
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    known = sorted(key for key in T_CRIT_95 if key <= df)
    return T_CRIT_95[known[-1]] if known else Z_95


# ---------------------------------------------------------------------------
# Loading per-item outcomes
#
# Every bench already writes per-item verdicts; nothing needs re-generating.
#   IFEval  : eval_results_{strict,loose}.jsonl, one row per prompt, from the
#             vendored Google checker.
#   IFBench : the same two files in the same shape, from its own registry, so
#             load_ifeval reads both. The verdicts are not interchangeable
#             across the two benches -- different prompts, different checkers --
#             but within one bench the pairing logic is identical.
#   Multi-IF: responses.jsonl, one row per conversation with a turns list. Only
#             rows the metric marked scored=True carry follow lists; the rest hit
#             an instruction id the registry does not implement.
# Items are keyed so that two runs align on the same units. IFEval keys on the
# prompt text (the checker's own key) and Multi-IF on (conversation key, turn).
# ---------------------------------------------------------------------------


class Item:
    """One scored unit: did it follow every instruction, and which ones."""

    __slots__ = ("key", "followed_all", "follow_list")

    def __init__(self, key: str, followed_all: bool, follow_list: list[bool]) -> None:
        self.key = key
        self.followed_all = followed_all
        self.follow_list = follow_list


def load_ifeval(run_dir: Path, mode: str) -> dict[str, Item]:
    path = run_dir / f"eval_results_{mode}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. It is written by onereplay.eval.metrics.ifeval; a run "
            "evaluated before that file existed needs re-scoring (greedy decoding, so "
            "the numbers will not move)."
        )
    items: dict[str, Item] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            follow_list = [bool(value) for value in record["follow_instruction_list"]]
            items[record["prompt"]] = Item(
                record["prompt"], bool(record["follow_all_instructions"]), follow_list
            )
    return items


def load_multiif(run_dir: Path, mode: str, turns: set[int] | None) -> dict[str, Item]:
    path = run_dir / "responses.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (Multi-IF writes per-turn verdicts there)")
    field = f"{mode}_follow_list"
    items: dict[str, Item] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            conversation = record.get("key")
            for turn_record in record.get("turns", []):
                if not turn_record.get("scored"):
                    continue
                turn = int(turn_record["turn"])
                if turns is not None and turn not in turns:
                    continue
                follow_list = [bool(value) for value in turn_record[field]]
                key = f"{conversation}#t{turn}"
                items[key] = Item(key, all(follow_list), follow_list)
    return items


def find_run_dir(roots: list[Path], bench: str, run: str) -> Path | None:
    for root in roots:
        candidate = root / bench / run
        if candidate.is_dir():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Paired statistics on a common item set
# ---------------------------------------------------------------------------


class PairedPrompt:
    """McNemar on prompt-level pass/fail over the items both runs scored."""

    def __init__(self, left: dict[str, Item], right: dict[str, Item]) -> None:
        shared = sorted(set(left) & set(right))
        self.n = len(shared)
        self.keys = shared
        # b: left right, right wrong.  c: the reverse.  Concordant pairs carry no
        # information about the difference and drop out of the test entirely.
        self.b = sum(1 for k in shared if left[k].followed_all and not right[k].followed_all)
        self.c = sum(1 for k in shared if not left[k].followed_all and right[k].followed_all)
        self.left_correct = sum(1 for k in shared if left[k].followed_all)
        self.right_correct = sum(1 for k in shared if right[k].followed_all)

    @property
    def diff(self) -> float:
        """left - right, in accuracy points (not percent)."""

        return (self.b - self.c) / self.n if self.n else 0.0

    @property
    def discordant(self) -> int:
        return self.b + self.c

    @property
    def se(self) -> float:
        """SE of the paired difference of proportions."""

        if not self.n:
            return 0.0
        variance = (self.b + self.c - (self.b - self.c) ** 2 / self.n) / self.n**2
        return math.sqrt(max(variance, 0.0))

    @property
    def ci_half(self) -> float:
        return Z_95 * self.se

    @property
    def p_value(self) -> float:
        return mcnemar_exact(self.b, self.c)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs.

    The chi-square form of McNemar needs b+c to be reasonably large. Here b+c is
    routinely under 40 -- the two adapters agree on most prompts -- so the exact
    test is the right one and costs nothing at this size.
    """

    total = b + c
    if total == 0:
        return 1.0
    smaller = min(b, c)
    tail = sum(math.comb(total, k) for k in range(smaller + 1)) / 2**total
    return min(1.0, 2.0 * tail)


def instruction_accuracy(items: dict[str, Item], keys: list[str]) -> float:
    correct = sum(sum(items[k].follow_list) for k in keys)
    total = sum(len(items[k].follow_list) for k in keys)
    return correct / total if total else 0.0


def cluster_bootstrap_instruction(
    left: dict[str, Item],
    right: dict[str, Item],
    keys: list[str],
    reps: int,
    seed: int,
) -> tuple[float, float, float]:
    """Difference in instruction accuracy, resampling prompts rather than instructions.

    Returns (point estimate, 2.5th pct, 97.5th pct). Resampling at the prompt
    level keeps the instructions belonging to one prompt together, which is what
    makes the interval honest: prompts carry 1 to 3 instructions and a prompt
    that fails tends to fail all of them.
    """

    point = instruction_accuracy(left, keys) - instruction_accuracy(right, keys)
    if not keys or reps <= 0:
        return point, point, point

    rng = random.Random(seed)
    size = len(keys)
    # Precompute per-prompt (correct, total) for both sides so a replicate is
    # just two running sums instead of a re-walk of the follow lists.
    left_pairs = [(sum(left[k].follow_list), len(left[k].follow_list)) for k in keys]
    right_pairs = [(sum(right[k].follow_list), len(right[k].follow_list)) for k in keys]

    diffs = []
    for _ in range(reps):
        left_correct = left_total = right_correct = right_total = 0
        for _ in range(size):
            index = rng.randrange(size)
            correct, total = left_pairs[index]
            left_correct += correct
            left_total += total
            correct, total = right_pairs[index]
            right_correct += correct
            right_total += total
        left_accuracy = left_correct / left_total if left_total else 0.0
        right_accuracy = right_correct / right_total if right_total else 0.0
        diffs.append(left_accuracy - right_accuracy)
    diffs.sort()
    low = diffs[int(0.025 * (reps - 1))]
    high = diffs[int(0.975 * (reps - 1))]
    return point, low, high


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def percent(value: float) -> str:
    return f"{value * 100:.2f}"


def signed_percent(value: float) -> str:
    return f"{value * 100:+.2f}"


def report_readings(
    arms: list[tuple[str, str]],
    seeds: list[int],
    loaded: dict[tuple[str, int], dict[str, Item]],
    missing: list[str],
    bench: str,
    mode: str,
) -> None:
    print("=" * 96)
    print(f"1. 逐 run 读数（{bench} / {mode}）")
    print("=" * 96)
    header = f"{'arm':<14} {'seed':>5} {'n':>6} {'prompt acc':>11} {'inst acc':>10}  run"
    print(header)
    print("-" * len(header))
    for name, run in arms:
        for seed in seeds:
            items = loaded.get((name, seed))
            run_name = run.format(seed=seed)
            if items is None:
                print(f"{name:<14} {seed:>5} {'--':>6} {'--':>11} {'--':>10}  {run_name}（缺）")
                continue
            keys = sorted(items)
            prompt_accuracy = sum(items[k].followed_all for k in keys) / len(keys)
            print(
                f"{name:<14} {seed:>5} {len(keys):>6} "
                f"{percent(prompt_accuracy):>11} "
                f"{percent(instruction_accuracy(items, keys)):>10}  {run_name}"
            )
    if missing:
        print()
        print("缺的 run（下面的检验会自动跳过它们）:")
        for run_name in missing:
            print(f"  {run_name}")


def report_seed_noise(
    arms: list[tuple[str, str]],
    seeds: list[int],
    loaded: dict[tuple[str, int], dict[str, Item]],
) -> dict[str, dict[str, float]]:
    """Per-arm spread across seeds. This is the term no amount of eval data shrinks."""

    print()
    print("=" * 96)
    print("2. 训练噪声：同一配置换 seed")
    print("=" * 96)

    stats: dict[str, dict[str, float]] = {}
    header = (
        f"{'arm':<14} {'seeds':>6} {'mean':>8} {'sd':>7} {'spread':>8} "
        f"{'mean 95% CI':>16}   逐 seed"
    )
    print(header)
    print("-" * len(header))
    for name, _ in arms:
        values = []
        per_seed = []
        for seed in seeds:
            items = loaded.get((name, seed))
            if items is None:
                continue
            keys = sorted(items)
            value = sum(items[k].followed_all for k in keys) / len(keys)
            values.append(value)
            per_seed.append(f"s{seed}={percent(value)}")
        if not values:
            continue
        count = len(values)
        mean = statistics.fmean(values)
        if count >= 2:
            sd = statistics.stdev(values)
            spread = max(values) - min(values)
            half = t_crit_95(count - 1) * sd / math.sqrt(count)
            ci = f"±{percent(half)}"
        else:
            sd = float("nan")
            spread = float("nan")
            half = float("nan")
            ci = "n/a（单 seed）"
        stats[name] = {"mean": mean, "sd": sd, "spread": spread, "count": count}
        sd_text = "  --  " if math.isnan(sd) else f"{percent(sd):>7}"
        spread_text = "   --   " if math.isnan(spread) else f"{percent(spread):>8}"
        print(
            f"{name:<14} {count:>6} {percent(mean):>8} {sd_text} {spread_text} "
            f"{ci:>16}   {' '.join(per_seed)}"
        )

    usable = [name for name, value in stats.items() if value["count"] >= 2]
    print()
    if not usable:
        print("只有一个 seed，训练噪声无法估计。这一栏空着的时候，第 3 栏的配对区间")
        print("是**下界**而不是总不确定度：它只覆盖评测，不覆盖重训。")
    else:
        worst = max(usable, key=lambda name: stats[name]["spread"])
        print(
            f"跨 seed 波动最大的是 {worst}，极差 {percent(stats[worst]['spread'])} 个点。"
            "任何比它小的方法间差值都不能当作方法的性质。"
        )
    return stats


def report_pairs(
    arms: list[tuple[str, str]],
    seeds: list[int],
    loaded: dict[tuple[str, int], dict[str, Item]],
    bootstrap_reps: int,
) -> dict[tuple[str, str], list[PairedPrompt]]:
    print()
    print("=" * 96)
    print("3. 评测分辨率：逐题配对（McNemar 精确检验）")
    print("=" * 96)
    print("b = 左对右错的题数，c = 反向。只有这 b+c 道不一致的题带信息，其余全部抵消。")
    print()

    paired: dict[tuple[str, str], list[PairedPrompt]] = {}
    header = (
        f"{'A vs B':<30} {'seed':>5} {'n':>5} {'b':>4} {'c':>4} "
        f"{'ΔA-B':>8} {'95% CI':>10} {'p':>8} {'Δinst':>8} {'inst 95% CI':>18}"
    )
    print(header)
    print("-" * len(header))
    for (left_name, left_run), (right_name, right_run) in combinations(arms, 2):
        label = f"{left_name} vs {right_name}"
        rows: list[PairedPrompt] = []
        for seed in seeds:
            left = loaded.get((left_name, seed))
            right = loaded.get((right_name, seed))
            if left is None or right is None:
                continue
            test = PairedPrompt(left, right)
            if not test.n:
                continue
            rows.append(test)
            point, low, high = cluster_bootstrap_instruction(
                left, right, test.keys, bootstrap_reps, seed
            )
            print(
                f"{label:<30} {seed:>5} {test.n:>5} {test.b:>4} {test.c:>4} "
                f"{signed_percent(test.diff):>8} "
                f"{'±' + percent(test.ci_half):>10} "
                f"{test.p_value:>8.3f} "
                f"{signed_percent(point):>8} "
                f"{'[' + signed_percent(low) + ', ' + signed_percent(high) + ']':>18}"
            )
        if rows:
            paired[(left_name, right_name)] = rows
    print()
    print("Δ 与 CI 都是百分点。inst 列用按 prompt 重抽的 cluster bootstrap，因为")
    print("instruction 嵌在 prompt 里，不是独立样本。")
    return paired


def report_verdict(
    paired: dict[tuple[str, str], list[PairedPrompt]],
    seed_stats: dict[str, dict[str, float]],
) -> None:
    print()
    print("=" * 96)
    print("4. 判决：这个差值站得住吗")
    print("=" * 96)

    for (left_name, right_name), rows in paired.items():
        label = f"{left_name} - {right_name}"
        diffs = [row.diff for row in rows]
        gap = statistics.fmean(diffs)
        eval_limit = statistics.median(row.ci_half for row in rows)
        prompts = statistics.fmean(row.n for row in rows)

        print()
        print(f"### {label}")
        print(
            f"平均差值 {signed_percent(gap)} 个点"
            f"（{prompts:.0f} 题上约 {abs(gap) * prompts:.1f} 题）"
        )
        print(f"评测分辨率（配对 95% CI 半宽，中位数）  ±{percent(eval_limit)} 个点")

        verdicts: list[str] = []
        if abs(gap) < eval_limit:
            verdicts.append(
                "单 seed 的评测分辨率就盖不住这个差值：同一批题上换个模型抖一抖就能翻转"
            )

        left_stats = seed_stats.get(left_name, {})
        right_stats = seed_stats.get(right_name, {})
        counts = [
            value.get("count", 0)
            for value in (left_stats, right_stats)
            if value.get("count", 0) >= 2
        ]
        if counts:
            spreads = [
                value["spread"]
                for value in (left_stats, right_stats)
                if value.get("count", 0) >= 2
            ]
            train_spread = max(spreads)
            print(f"训练噪声（两臂中较大的跨 seed 极差）      {percent(train_spread)} 个点")
            if abs(gap) < train_spread:
                verdicts.append(
                    "差值小于单臂换 seed 的自然波动，排名不是方法的性质"
                )

        if len(rows) >= 2:
            # Pair the arms within each seed: the same seed means the same
            # new-task data order, so the difference is the right unit and the
            # test only needs to beat the seed-to-seed variation of that
            # difference, not of each arm separately.
            sd_diff = statistics.stdev(diffs)
            count = len(diffs)
            standard_error = sd_diff / math.sqrt(count)
            critical = t_crit_95(count - 1)
            half = critical * standard_error
            print(
                f"按 seed 配对的 t 检验（K={count}, df={count - 1}, t*={critical:.3f}）"
                f"  {signed_percent(gap)} ± {percent(half)}"
            )
            if standard_error > 0:
                statistic = abs(gap) / standard_error
                print(f"  |t| = {statistic:.2f}  →  {'显著' if statistic > critical else '不显著'}")
            if abs(gap) > 0 and sd_diff > 0:
                needed = (POWER_CONST * sd_diff / abs(gap)) ** 2
                print(
                    f"  要在 80% 功效下分辨出这个幅度，需要约 {math.ceil(needed)} 个 seed"
                    f"（当前 {count}）"
                )
        else:
            print("按 seed 配对的 t 检验                    需要 ≥2 个 seed，跳过")

        if verdicts:
            for text in verdicts:
                print(f"  [NO] {text}")
        else:
            print("  [OK] 差值同时超过评测分辨率与跨 seed 波动")

    print()
    print("-" * 96)
    print("读法：一个差值要先大于第 3 栏的配对区间（评测能不能分辨），再大于第 2 栏的")
    print("跨 seed 极差（重训能不能复现），才配写进结论。只满足前者的差值，换个 seed")
    print("就会翻面 -- 这正是 lambda=3e-2 那条线在换集群 / 换 C 之后发生的事。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired McNemar + across-seed variance for IFEval / Multi-IF runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results_root",
        type=str,
        required=True,
        help="Result tree holding <bench>/<run>/ subdirectories",
    )
    parser.add_argument(
        "--also_root",
        type=str,
        action="append",
        default=[],
        help=(
            "Extra result trees searched after --results_root, for runs that landed "
            "elsewhere (results_safety holds cs_onereplay_lam3e-2_seed1_regonce). "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--bench", type=str, choices=["ifeval", "ifbench", "multiif"], default="ifeval"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["strict", "loose"],
        default="strict",
        help=(
            "Which checker verdict to test. strict is what every IFEval table reports; "
            "use loose for ifbench, which is the number its paper reports."
        ),
    )
    parser.add_argument(
        "--arm",
        type=str,
        action="append",
        required=True,
        help=(
            "Name=run_template, where the template may contain {seed}. Repeatable; "
            'every pair of arms gets tested. Example: --arm "EWC=cs_ewc_lam3e2_seed{seed}"'
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1",
        help="Comma-separated seeds substituted into each --arm template",
    )
    parser.add_argument(
        "--turns",
        type=str,
        default="",
        help="Multi-IF only: comma-separated turns to keep, e.g. 3 for the turn-3 gap",
    )
    parser.add_argument(
        "--by_turn",
        type=int,
        default=0,
        help="Multi-IF only: 1 runs the whole report once per turn as well as pooled",
    )
    parser.add_argument(
        "--bootstrap_reps",
        type=int,
        default=2000,
        help="Cluster bootstrap replicates for the instruction-level interval; 0 skips",
    )
    args = parser.parse_args()

    arms: list[tuple[str, str]] = []
    for spec in args.arm:
        if "=" not in spec:
            parser.error(f"--arm needs Name=run_template, got {spec!r}")
        name, run = spec.split("=", 1)
        arms.append((name.strip(), run.strip()))
    if len(arms) < 2:
        parser.error("need at least two --arm entries to compare")

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    roots = [Path(args.results_root)] + [Path(value) for value in args.also_root]

    turn_groups: list[tuple[str, set[int] | None]] = []
    explicit = {int(value) for value in args.turns.split(",") if value.strip()}
    if args.bench == "multiif":
        base = explicit or None
        turn_groups.append(("pooled" if base is None else f"turns {sorted(base)}", base))
        if args.by_turn:
            for turn in sorted(explicit or {1, 2, 3}):
                turn_groups.append((f"turn {turn}", {turn}))
    else:
        turn_groups.append(("", None))

    for group_label, turns in turn_groups:
        loaded: dict[tuple[str, int], dict[str, Item]] = {}
        missing: list[str] = []
        for name, run in arms:
            for seed in seeds:
                run_name = run.format(seed=seed)
                run_dir = find_run_dir(roots, args.bench, run_name)
                if run_dir is None:
                    missing.append(f"{run_name}（在 {len(roots)} 棵 results 树里都没找到）")
                    continue
                try:
                    if args.bench in ("ifeval", "ifbench"):
                        items = load_ifeval(run_dir, args.mode)
                    else:
                        items = load_multiif(run_dir, args.mode, turns)
                except FileNotFoundError as error:
                    missing.append(f"{run_name}: {error}")
                    continue
                if not items:
                    missing.append(f"{run_name}: 没有可用的逐题结果")
                    continue
                loaded[(name, seed)] = items

        if not loaded:
            print("没有读到任何逐题结果。检查 --results_root 与 --arm 的 run 名。")
            for text in missing:
                print(f"  {text}")
            raise SystemExit(1)

        if group_label:
            print()
            print("#" * 96)
            print(f"# Multi-IF：{group_label}")
            print("#" * 96)

        report_readings(arms, seeds, loaded, missing, args.bench, args.mode)
        seed_stats = report_seed_noise(arms, seeds, loaded)
        paired = report_pairs(arms, seeds, loaded, args.bootstrap_reps)
        report_verdict(paired, seed_stats)


if __name__ == "__main__":
    main()
