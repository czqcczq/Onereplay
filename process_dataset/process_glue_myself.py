"""
将bool数据集用Qwen3的chat模板转换成jsonl格式，包含text（完整的对话文本，包含答案），prompt_text（不包含答案的对话文本），answer（True or False），passage，question等字段
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any
from datasets import load_from_disk

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, DataCollatorForTokenClassification
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_DIR = Path("/home/weiliu1/huggingface/models/Qwen3-1.7B")
DEFAULT_CACHE_DIR = Path("/home/weiliu1/huggingface/datasets")
DEFAULT_OUTPUT_DIR = Path("/home/weiliu1/huggingface/datasets/boolq_qwen3_1.7b")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SuperGLUE BoolQ into Qwen3 chat-template JSONL files."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default="boolq")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--dataset_name", type=str, default="boolq")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Maximum token length. 0 means no truncation.",
    )
    parser.add_argument("--rewrite", type=int, default=1, help="Whether to rewrite the output files if they already exist. 0 means no rewrite, 1 means rewrite.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="0 means keep the full train split.",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=0,
        help="0 means keep the full validation split.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="0 means keep the full test split.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep the original dataset order before optional sample limiting.",
    )
    parser.add_argument(
        "--print-example",
        action="store_true",
        help="Print one processed training example after writing files.",
    )
    return parser.parse_args()



def build_user_prompt(dataset_name, example) -> str:
    if dataset_name == "boolq":
        passage = example.get("passage", "")
        question = example.get("question", "")
        return (
            "Read the passage and answer the question according to the passage.\n\n"
            f"Passage:\n{passage.strip()}\n\n"
            f"Question:\n{question.strip()}\n\n"
            "Answer with exactly one word: True or False."
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def build_messages(dataset_name, example, include_answer: bool = True) -> list[dict[str, str]]:
    """
    处理一条数据
    """
    messages = [
        {
            "role": "user",
            "content": build_user_prompt(dataset_name, example),
        },
    ]
    
    # 将原来的0，1,转换为True or False
    if dataset_name == "boolq":
        answer = str(example["label"])
        
        if answer == "1":
            example["transform_label"] = "True"
        elif answer == "0":
            example["transform_label"] = "False"
        # else:
        #     print('wrong, no such labels')
        #     example["transform_label"] = answer
        #     print(answer)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    if include_answer:
        messages.append({"role": "assistant", "content": example["transform_label"]})
    return messages

def ensure_final_eos(text: str, tokenizer: Any) -> str:
    """
    确保文本以EOS标记结尾，如果tokenizer定义了EOS标记的话。如果不加这个默认apply_tem可能会生成<|im_end|>\n，多了一个\n 
    """
    text = text.rstrip()
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token and not text.endswith(eos_token):
        text += eos_token
    return text

def apply_prompt_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """
    用于推理过程，去掉了答案
    返回类似<|im_start|>user
  ...
  <|im_end|>
  <|im_start|>assistant
  <think>

  </think>

    """

    prompt_messages = [message for message in messages if message["role"] != "assistant"]

    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,      #对于qwen,关闭think模式
    )

def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """
    用于训练过程，保留了答案，ensure_final_eos则是去掉了eos后面的\n
    """

    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return ensure_final_eos(text, tokenizer)

def convert_row(tokenizer: Any, row: dict[str, Any], dataset_name: str, max_length: int = 0) -> dict[str, Any]:

    messages = build_messages(dataset_name, row)
    text = apply_chat_template(tokenizer, messages)
    prompt_text = apply_prompt_template(tokenizer, messages)
    tokenized = tokenizer_to_ids(tokenizer, text, prompt_text, max_length=max_length)
    

    converted = {
        "text": text,
        "prompt_text": prompt_text,
        "input_ids": tokenized["input_ids"],
        "labels": tokenized["labels"],
        "attention_mask": tokenized["attention_mask"],
        "passage": row["passage"],
        "question": row["question"],
    }

    if "idx" in row:
        converted["idx"] = row["idx"]
    return converted


def maybe_limit_split(
    rows: list[dict[str, Any]],
    max_samples: int,
    *,
    shuffle: bool,
    rng: random.Random,) -> list[dict[str, Any]]:
    """
    对数据取前max_samples条，或者打乱后取前max_samples条，如果max_samples为0，则不限制样本数量
    """

    if shuffle:
        rows = rows[:]
        rng.shuffle(rows)
    if max_samples > 0:
        return rows[:max_samples]
    return rows

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def process_split(
    name: str,
    dataset: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    rng: random.Random,
) -> int:
    if name not in dataset:
        return 0

    max_samples_by_split = {
        "train": args.max_train_samples,
        "validation": args.max_validation_samples,
    }
    raw_rows = [dict(row) for row in dataset[name]]
    selected_rows = maybe_limit_split(
        raw_rows,
        max_samples_by_split.get(name, 0),
        shuffle=not args.no_shuffle and name == "train",
        rng=rng,
    )
    converted_rows = [
        convert_row(tokenizer, row, args.dataset_name, args.max_length)
        for row in selected_rows
    ]
    write_jsonl(args.output_dir / f"{args.dataset_name}_{name}.jsonl", converted_rows)
    return len(converted_rows)


def tokenizer_to_ids(
    tokenizer: Any,
    text: str,
    prompt_text: str | None = None,
    max_length: int = 0,
) -> dict[str, list[int]]:
    """
    将完整训练文本转换为 causal LM 训练需要的 input_ids、labels、attention_mask。

    text: 完整样本，包含 user prompt 和 assistant 标准答案。
    prompt_text: 只包含 user prompt 和 assistant 生成起点，不包含标准答案。

    labels 的规则：
    - prompt 部分设为 -100，不参与 loss。
    - assistant 答案部分保留 token id，参与 loss。
    """
    encoded = tokenizer(text, add_special_tokens=False)
    full_input_ids = encoded["input_ids"]   # 完整输入的 token ids，包括 prompt 和答案
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"] # prompt 部分的 token ids，用于确定 labels 中哪些位置需要设为 -100
    prompt_len = min(len(prompt_ids), len(full_input_ids))

    labels = [-100] * prompt_len+full_input_ids[len(prompt_ids):]
    attention_mask = encoded["attention_mask"]


    if max_length > 0:
        full_input_ids = full_input_ids[-max_length:]
        labels = labels[-max_length:]
        attention_mask = attention_mask[-max_length:]

    return {
        "input_ids": full_input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def build_loader(dataset, tokenizer, batch_size, train=True):
    tokenizer.padding_side = "left"
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)  # 将labels的pad用-100来pad
    # data_collator=PromptDataCollator(tokenizer=tokenizer)

    # 只保留这几个 key，移除其他所有列
    keep_columns = ["input_ids", "labels", "attention_mask"]

    # 计算要删除的列
    columns_to_remove = [col for col in dataset.column_names if col not in keep_columns]

    # 删除不需要的列

    dataset = dataset.remove_columns(columns_to_remove)

    dataloader=DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size,shuffle=train)

    return dataloader   

def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install the training data dependencies first, "
            "for example: pip install datasets transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if args.rewrite==1:
        dataset = load_from_disk(args.dataset_dir / args.dataset_name)


        counts = {}
        for split in ("train", "validation"):
            counts[split] = process_split(split, dataset, tokenizer, args, rng)

        print(f"Wrote {args.dataset_name} files to: {args.output_dir}")
        for split, count in counts.items():
            if count:
                print(f"  {args.dataset_name}_{split}.jsonl: {count} rows")

        if args.print_example:
            example_path = args.output_dir / f"{args.dataset_name}_train.jsonl"
            with example_path.open("r", encoding="utf-8") as f:
                example = json.loads(f.readline())
            print("\nExample row:")
            print(json.dumps(example, ensure_ascii=False, indent=2)[:3000])
    else:
        pass




def load_model(model_dir, model_name, use_bf16, args):
    model_weight = model_dir + model_name
    config = AutoConfig.from_pretrained(model_weight)
    for key, value in vars(args).items():
        if not hasattr(config, key):
            setattr(config, key, value)

    tokenizer = AutoTokenizer.from_pretrained(model_weight, padding_side="left")
    if use_bf16 == 1:
        model = AutoModelForCausalLM.from_pretrained(model_weight, config=config, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_weight, config=config)

    #防止llama没有pad token
    if tokenizer.pad_token_id==None:
        model.config.pad_token_id=tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    print('this is the model config:')
    print(model.config.to_dict())

    return model, tokenizer




if __name__ == "__main__":
    main()
