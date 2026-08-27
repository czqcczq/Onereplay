"""Score a finished checkpoint's cross-entropy on fixed probe corpora.

Chapter 3 ran this measurement *during* training (`--probe_every`), which is
what produced the held-out/in-pool curves. There was no way to ask the same
question of an adapter that already exists, so comparing finished runs meant
falling back on benchmark accuracy -- a metric whose paired resolution on
MATH-500 is 3.5pp while the effect under study is 0.8pp.

Cross-entropy against the base model's own answers is the sensitive
instrument: on FLAN it moved 285% (0.1231 -> 0.4742) for a vanilla run whose
IFEval accuracy moved 11.5pp. It also costs forward passes only, so a whole
sweep of checkpoints is minutes rather than the hours a generative benchmark
needs.

The numbers this prints are directly comparable to the `probe_*` fields in the
training metrics because the corpus loading, chat rendering, prompt masking and
token-weighted averaging all go through the same functions the trainer uses --
`build_self_distilled_probe`, `build_probe_loader` and
`BaseTrainer.evaluate_probe`. Nothing here reimplements the loss.

    python -m onereplay.scripts.score_probe_ce \\
        --model_dir /path/models --model_name Qwen3-1.7B \\
        --adapter_path /path/results/adapters/cs_vanilla_seed1 \\
        --run_name cs_vanilla_seed1 \\
        --probes math500=/path/probe_math500_base.jsonl,gsm8k=/path/probe_gsm8k_base.jsonl \\
        --max_len 2048 --batch_size 8 \\
        --out /path/results/probe_ce/scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

from onereplay.data.probe import build_probe_loader, build_self_distilled_probe
from onereplay.eval.runner import load_eval_model
from onereplay.trainers.sft import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen3-1.7B")
    parser.add_argument("--use_bf16", type=int, default=1)
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="",
        help="LoRA adapter dir, full-FT checkpoint dir, or empty for bare base.",
    )
    parser.add_argument("--run_name", type=str, default="base")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--probes",
        type=str,
        required=True,
        help="Comma-separated name=path pairs, e.g. math500=/a.jsonl,gsm8k=/b.jsonl",
    )
    parser.add_argument(
        "--probe_size",
        type=int,
        default=0,
        help="Rows per probe, evenly strided; 0 keeps the whole corpus.",
    )
    # Math chains are long: MATH-500 answers sit around 528 tokens at the median
    # and run past 4000 at the tail. The 512 used for Commonsense would cut most
    # of them, so the default is 2048 to match the MetaMath self-distill budget.
    # Whatever value is used must be identical across every run being compared.
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--map_cache_dir", type=str, default="")
    parser.add_argument("--replay_drop_truncated", type=int, default=1)
    parser.add_argument("--replay_input_column", type=str, default="inputs")
    parser.add_argument("--replay_target_column", type=str, default="targets")
    parser.add_argument("--out", type=str, default="", help="Append one JSON row here.")
    parser.add_argument(
        "--per_row_dir",
        type=str,
        default="",
        help="Write per-row (loss_sum, tokens) here so bootstrap_probe_ce.py can put "
        "error bars on the aggregate. Row order is the corpus order, identical "
        "across runs, which is what makes the bootstrap paired.",
    )
    parser.add_argument(
        "--verify_aggregate",
        type=int,
        default=1,
        help="Also run the trainer's own evaluate_probe and assert the per-row sum "
        "reproduces it. Doubles the forward cost (the probe is minutes, so this is "
        "cheap insurance that the two paths compute the same loss).",
    )
    parser.add_argument("--row_batch_size", type=int, default=4, help="Batch for the per-row pass.")
    return parser.parse_args()


def parse_probes(spec: str) -> list[tuple[str, str]]:
    probes = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--probes entry must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        probes.append((name.strip(), path.strip()))
    return probes


def row_losses(model, loader, device) -> list[dict[str, float]]:
    """Per-row supervised loss sum and token count.

    evaluate_probe only ever sees the batch mean HuggingFace returns, so the
    finest granularity it can report is a batch. A bootstrap has to resample the
    unit of independence, which is the row, hence this second path.

    The arithmetic mirrors HF's causal LM loss exactly -- shift logits left by
    one, drop position 0's label, ignore -100 -- so that summing loss_sum and
    dividing by summed tokens reproduces evaluate_probe's token-weighted mean.
    --verify_aggregate checks that equality rather than trusting this comment.

    Cross-entropy is taken one row at a time: Qwen3's vocabulary is ~152k, so a
    float32 (batch x seq x vocab) intermediate would be tens of GiB while a
    single row is around 1 GiB.
    """

    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            shift_labels = labels[:, 1:]
            for i in range(input_ids.shape[0]):
                target = shift_labels[i]
                mask = target != -100
                count = int(mask.sum())
                if count == 0:
                    rows.append({"loss_sum": 0.0, "tokens": 0})
                    continue
                token_loss = F.cross_entropy(
                    logits[i, :-1, :].float(),
                    target,
                    reduction="none",
                    ignore_index=-100,
                )
                rows.append(
                    {"loss_sum": float((token_loss * mask).sum()), "tokens": count}
                )
    return rows


def aggregate(rows: list[dict[str, float]]) -> float:
    total_tokens = sum(row["tokens"] for row in rows)
    if total_tokens == 0:
        return 0.0
    return sum(row["loss_sum"] for row in rows) / total_tokens


def dataset_stats(dataset, max_len: int) -> dict[str, float]:
    """Rows, supervised tokens and how many rows the tokenizer had to cut.

    A row truncated at max_len is only partly scored, so the CE it contributes
    covers a prefix of the answer rather than the answer. That is still a fair
    comparison as long as every run truncates the same rows the same way, but a
    high rate means max_len is too small for this corpus and the number is
    measuring something narrower than intended.
    """

    lengths = [len(ids) for ids in dataset["input_ids"]]
    supervised = sum(
        sum(1 for label in labels[1:] if label != -100) for labels in dataset["labels"]
    )
    at_cap = sum(1 for length in lengths if length >= max_len)
    return {
        "rows": len(dataset),
        "supervised_tokens": supervised,
        "rows_at_max_len": at_cap,
        "rows_at_max_len_frac": at_cap / max(len(dataset), 1),
        "mean_len": sum(lengths) / max(len(lengths), 1),
    }


def main() -> None:
    args = parse_args()
    probes = parse_probes(args.probes)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_eval_model(
        args.model_dir, args.model_name, args.use_bf16, args.adapter_path
    )
    model.to(device)
    model.eval()

    # The optimizer is never touched by evaluate_probe; SFTTrainer is used here
    # purely to borrow the exact forward and averaging the training probe uses.
    trainer = SFTTrainer(model=model, optimizer=None, device=device)

    loader_args = Namespace(
        max_len=args.max_len,
        map_cache_dir=args.map_cache_dir,
        replay_cache_dir="",
        replay_drop_truncated=args.replay_drop_truncated,
        replay_input_column=args.replay_input_column,
        replay_target_column=args.replay_target_column,
    )

    record: dict[str, object] = {
        "run_name": args.run_name,
        "adapter_path": args.adapter_path,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "probe_size": args.probe_size,
    }

    print(f"==== probe CE: {args.run_name} ====")
    for name, path in probes:
        if not Path(path).is_file():
            print(f"  {name:<20} 跳过：找不到 {path}")
            continue
        dataset = build_self_distilled_probe(
            loader_args, tokenizer, path, args.probe_size, f"probe_{name}_tokenized.arrow"
        )
        stats = dataset_stats(dataset, args.max_len)
        start = time.time()

        row_loader = build_probe_loader(dataset, tokenizer, args.row_batch_size)
        rows = row_losses(model, row_loader, device)
        ce = aggregate(rows)

        if args.verify_aggregate == 1:
            reference = trainer.evaluate_probe(build_probe_loader(dataset, tokenizer, args.batch_size))
            gap = abs(reference - ce) / max(reference, 1e-12)
            status = "一致" if gap < 1e-3 else f"!! 不一致，相对差 {gap:.2e}"
            print(f"  {name:<20} 逐行 {ce:.6f} vs evaluate_probe {reference:.6f}  {status}")

        if args.per_row_dir:
            per_row_path = Path(args.per_row_dir) / f"{args.run_name}__{name}.jsonl"
            per_row_path.parent.mkdir(parents=True, exist_ok=True)
            with per_row_path.open("w", encoding="utf-8") as file:
                for index, row in enumerate(rows):
                    file.write(json.dumps({"row": index, **row}) + "\n")

        elapsed = time.time() - start
        record[f"probe_{name}"] = ce
        record[f"{name}_rows"] = stats["rows"]
        record[f"{name}_supervised_tokens"] = stats["supervised_tokens"]
        record[f"{name}_rows_at_max_len"] = stats["rows_at_max_len"]
        print(
            f"  {name:<20} CE={ce:.6f}  rows={stats['rows']} "
            f"tokens={stats['supervised_tokens']} "
            f"at_max_len={stats['rows_at_max_len']} ({stats['rows_at_max_len_frac']:.1%}) "
            f"({elapsed:.1f}s)"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"appended -> {out_path}")


if __name__ == "__main__":
    main()
