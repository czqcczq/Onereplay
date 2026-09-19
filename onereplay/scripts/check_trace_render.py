"""Preflight: does this model's chat template render the TRACE pools correctly?

The whole TRACE line rests on one assumption -- that training and evaluation see
byte-identical prompts, and that the prompt is a real prefix of the full training
sequence so label masking lands where it should. That assumption is a property of
the tokenizer's chat template, and it is cheap to verify and expensive to get
wrong: a mismatch shows up as "the model forgot everything", which is exactly the
result this experiment is looking for and would therefore not be questioned.

Written for the switch to Llama-3.2-3B-Instruct, where three things differ from
the Qwen line every other script here was built against:

  1. Llama-3.1/3.2 templates inject a system block containing "Cutting Knowledge
     Date" and "Today Date" even when no system message is supplied, and the date
     comes from a template default. So --system_prompt "" does not mean "no
     system turn" the way it does for Qwen, and the rendered text -- hence every
     measured length -- differs from the Qwen pools.
  2. ``eos_token`` is ``<|eot_id|>``, not ``<|endoftext|>``, so the
     ``--decode_stop_on_im_end`` switch is a no-op here and must stay 0.
  3. ``pad_token`` is unset; ``load_causal_lm_and_tokenizer`` aliases it to eos.

Run it on a login node before queueing anything:

    python -m onereplay.scripts.check_trace_render \
        --model_path models/Llama-3.2-3B-Instruct \
        --trace_data_dir data/processed --max_len 2048
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onereplay.core.chat_policy import configure_system_prompt  # noqa: E402
from onereplay.data.chat import apply_train_template  # noqa: E402
from onereplay.scripts.prepare_trace import TASK_SPECS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify chat rendering of the TRACE pools.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--trace_data_dir", type=str, default="data/processed")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--max_prompt_tokens", type=int, default=1024)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument(
        "--show",
        type=int,
        default=1,
        help="Print the head and tail of one rendered example per task.",
    )
    parser.add_argument(
        "--scan_rows",
        type=int,
        default=200,
        help="Rows per task to check budgets on. 0 checks every row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    configure_system_prompt(args.system_prompt)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    data_dir = Path(args.trace_data_dir)

    print(f"model          : {args.model_path}")
    print(f"eos_token      : {tokenizer.eos_token!r} (id {tokenizer.eos_token_id})")
    print(f"pad_token      : {tokenizer.pad_token!r}")
    print(f"system_prompt  : {args.system_prompt!r}")
    print(f"budgets        : max_prompt_tokens={args.max_prompt_tokens} max_len={args.max_len}")

    # The switch only exists for Qwen's '<|im_end|>'. On a tokenizer without it
    # the lookup returns a miss, which is fine, but saying so here is cheaper
    # than wondering later why the flag did nothing.
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"'<|im_end|>' id: {im_end} (absent/unknown means keep --decode_stop_on_im_end 0)")
    print()

    failures: list[str] = []

    def fail(message: str) -> None:
        failures.append(message)
        print(f"  FAIL {message}")

    def load(path: Path) -> list[dict]:
        """Read either artifact: the train JSONL uses question/response."""

        rows: list[dict] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                instruction = row.get("instruction") or row.get("question") or ""
                output = row.get("output") or row.get("response") or ""
                if instruction and output:
                    rows.append({"instruction": instruction, "output": output})
                if args.scan_rows > 0 and len(rows) >= args.scan_rows:
                    break
        return rows

    for spec in TASK_SPECS:
        # Both halves get checked. The train pool is what has to fit the window,
        # and the test file is what the metrics render; a pool built under a
        # different tokenizer breaks both, but only the train side can silently
        # corrupt a 13-hour run.
        sources = [
            path
            for path in (
                data_dir / f"trace_{spec.key}_train.jsonl",
                data_dir / f"trace_{spec.key}_test.jsonl",
            )
            if path.is_file()
        ]
        if not sources:
            fail(f"{spec.directory}: no trace_{spec.key}_{{train,test}}.jsonl under {data_dir}")
            continue

        print(f"=== {spec.directory}")
        first_row: dict | None = None
        for source in sources:
            rows = load(source)
            if not rows:
                fail(f"{spec.directory}: no usable rows in {source.name}")
                continue
            first_row = first_row or rows[0]

            worst_prompt = 0
            worst_total = 0
            prefix_violations = 0
            eos_violations = 0
            affix_violations = 0

            for row in rows:
                full_text, prompt_text = apply_train_template(
                    tokenizer, row["instruction"], "", row["output"]
                )
                full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

                # This is the one that matters: tokenizer_to_ids masks the first
                # len(prompt_ids) positions, so if the prompt is not a real token
                # prefix of the full sequence the loss lands on the wrong tokens
                # and nothing anywhere raises.
                if full_ids[: len(prompt_ids)] != prompt_ids:
                    prefix_violations += 1
                # A doubled EOS teaches the model to emit one and keep going.
                if tokenizer.eos_token and full_text.endswith(tokenizer.eos_token * 2):
                    eos_violations += 1
                if spec.keep == "head" and spec.prefix:
                    stored = row["instruction"]
                    if not (stored.startswith(spec.prefix) and stored.endswith(spec.suffix)):
                        affix_violations += 1

                worst_prompt = max(worst_prompt, len(prompt_ids))
                worst_total = max(worst_total, len(full_ids))

            print(
                f"    {source.name:<34} {len(rows):>5} rows  "
                f"prompt<={worst_prompt}/{args.max_prompt_tokens}  "
                f"total<={worst_total}/{args.max_len}"
            )

            label = f"{spec.directory}/{source.name}"
            if prefix_violations:
                fail(f"{label}: prompt is not a token prefix of full text on {prefix_violations} rows")
            if eos_violations:
                fail(f"{label}: doubled EOS on {eos_violations} rows")
            if affix_violations:
                fail(f"{label}: instruction affixes missing on {affix_violations} rows")
            if worst_prompt > args.max_prompt_tokens:
                fail(
                    f"{label}: prompt exceeds max_prompt_tokens "
                    f"({worst_prompt} > {args.max_prompt_tokens}); rebuild the pool with "
                    f"this tokenizer"
                )
            if worst_total > args.max_len:
                fail(
                    f"{label}: total exceeds max_len ({worst_total} > {args.max_len}); "
                    f"training would left-truncate and destroy the chat scaffold"
                )

        if args.show == 1 and first_row is not None:
            full_text, prompt_text = apply_train_template(
                tokenizer, first_row["instruction"], "", first_row["output"]
            )
            head = prompt_text[:320].replace("\n", "\\n")
            tail = full_text[-200:].replace("\n", "\\n")
            print(f"    prompt head: {head}")
            print(f"    full tail  : {tail}")
        print()

    if failures:
        print(f"{len(failures)} problem(s):")
        for message in failures:
            print(f"  - {message}")
        raise SystemExit(1)
    print("rendering is consistent; safe to train")


if __name__ == "__main__":
    main()
