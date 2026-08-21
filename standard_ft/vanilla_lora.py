

import json

import os



print('the file is ' + os.path.abspath(__file__))

import datetime
from datasets import load_dataset
import sys
from pathlib import Path

now = datetime.datetime.now()
bj_time = now + datetime.timedelta(hours=0)
print(bj_time.strftime('%Y-%m-%d %H:%M:%S'))

try:
    from mycode.process_dataset.process_glue_myself import load_model, tokenizer_to_ids, build_loader
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from process_dataset.process_glue_myself import load_model, tokenizer_to_ids, build_loader


from datasets import load_dataset, load_from_disk
import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import warnings
import argparse

warnings.filterwarnings("ignore")
from peft import get_peft_model, LoraConfig

import random
import time

import os


def parse():
    # 默认： nonorm, dis_lr=1, data=beer, save=0
    parser = argparse.ArgumentParser(
        description="SR")
    # machine
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='')
    parser.add_argument('--writer',
                        type=str,
                        default='./noname',
                        help='')
    parser.add_argument('--use_writer',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument('--save',
                        type=int,
                        default=0,
                        help='0: not save')
    parser.add_argument('--save_path',
                        type=str,
                        default="111",
                        help='')

    # dataset
    parser.add_argument('--dataset_dir',
                        type=str,
                        default="/home/weiliu1/huggingface/datasets/boolq_qwen3_1.7b/",
                        help='')
    parser.add_argument('--dataset_name',
                        type=str,
                        default="boolq_qwen3_1.7b",
                        help='')
    parser.add_argument('--test_out',
                        type=int,
                        default=0,
                        help='如果为1，将test out,即模型生成的答案，保存')
    parser.add_argument('--max_len',
                        type=int,
                        default=256,
                        help='')
    parser.add_argument('--max_len_eval',
                        type=int,
                        default=256,
                        help='')

    # model
    parser.add_argument('--model_dir',
                        type=str,
                        default='/home/weiliu1/huggingface/models/',
                        help='')
    parser.add_argument('--lora_path',
                        type=str,
                        default='/home/weiliu1/huggingface/models/',
                        help='lora应该保存在哪里')
    parser.add_argument('--model_name',
                        type=str,
                        default="Qwen3-1.7B",
                        help='')
    parser.add_argument('--use_lora',
                        type=int,
                        default=1,
                        help='')
    # merge方式
    parser.add_argument("--normal_ft",
                        type=int,
                        default=0,
                        help='如果=1，则是normal finetune')

    # learning parameters
    parser.add_argument('--use_bf16',
                        type=int,
                        default=1,
                        help='')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of training epoch')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='')
    parser.add_argument('--accumulation_size',
                        type=int,
                        default=128,
                        help='')
    parser.add_argument('--avg_loss',
                        type=int,
                        default=1,
                        help='')

    parser.add_argument('--lora_rank',
                        type=int,
                        default=8,
                        help='')
    parser.add_argument('--lora_alpha',
                        type=int,
                        default=16,
                        help='')

    args = parser.parse_args()
    return args


args = parse()


def set_seed(seed=42):
    random.seed(seed)  # Python 内置随机数
    np.random.seed(seed)  # Numpy 随机数
    torch.manual_seed(seed)  # PyTorch 随机数（CPU）
    torch.cuda.manual_seed(seed)  # PyTorch 随机数（GPU）
    torch.cuda.manual_seed_all(seed)  # 多 GPU 时保证所有 GPU 结果一致

set_seed(args.seed)




for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

if args.use_writer:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(args.writer)


model, tokenizer = load_model(args.model_dir, args.model_name, args.use_bf16, args)
args.lora_path = args.lora_path + 'lora'


if args.use_lora == 1:

    lora_config = LoraConfig(
        r=args.lora_rank,  # LoRA rank (可以调整)
        lora_alpha=args.lora_alpha,
        # target_modules=["q_proj", "k_proj", "v_proj"],  # 可以调整，根据模型结构
        target_modules=["q_proj", "v_proj"],  # 可以调整，根据模型结构
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    print('use lora，with rank={},alpha={}'.format(args.lora_rank, args.lora_alpha))
else:
    print('not use lora')

device = "cuda:{}".format(args.gpu)
model.to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

dataset = load_dataset(
      "json",
      data_files={
          "train": str(args.dataset_dir + "boolq_train.jsonl"),
          "validation": str(args.dataset_dir + "boolq_validation.jsonl"),
      },
  )

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def add_tokenized_fields(example):
      tokenized = tokenizer_to_ids(
          tokenizer=tokenizer,
          text=example["text"],
          prompt_text=example["prompt_text"],
          max_length=args.max_len,
      )

      return {
          "input_ids": tokenized["input_ids"],
          "labels": tokenized["labels"],
          "attention_mask": tokenized["attention_mask"],
      }


 # 只取前 100 条训练数据，验证集也可以取前 100 条
# dataset["train"] = dataset["train"].select(range(min(100, len(dataset["train"]))))
# dataset["validation"] = dataset["validation"].select(range(min(100,
# len(dataset["validation"]))))

train_dataset = dataset["train"].map(add_tokenized_fields)
valid_dataset = dataset["validation"].map(add_tokenized_fields)

train_loader = build_loader(
      train_dataset,
      tokenizer,
      batch_size=args.batch_size,
      train=True,
  )

valid_loader = build_loader(
    valid_dataset,
    tokenizer,
    batch_size=args.batch_size,
    train=False,
)


for i, batch in enumerate(train_loader):
    # 打印一条数据例子

    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']

    print('==============================')
    print('')
    print('here is an example of the training data sample')
    print(tokenizer.decode(input_ids[0]))
    print('input_ids={}'.format(input_ids[0]))
    print('labels={}'.format(labels[0]))
    print('attention_mask={}'.format(attention_mask[0]))
    print('')
    print('==============================')
    break

def train(train_dataloader):
    accumulation_steps = args.accumulation_size // args.batch_size

    model.train()  # 设置模型为训练模式
    total_train_loss = 0
    total_sample = 0
    seq_lens = []
    for i, batch in enumerate(train_dataloader):
        # 将数据移动到设备

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)


        num_sample = input_ids.shape[0]
        total_sample += num_sample

        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        # 反向传播
        if args.avg_loss == 1:
            loss = loss / accumulation_steps

        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            # total_train_loss += loss.item() * accumulation_steps
        elif (i + 1) == len(train_dataloader):  # 处理最后一个不完整的累积步
            optimizer.step()
            optimizer.zero_grad()
            # total_train_loss += loss.item() * ((i + 1) % accumulation_steps)

        # optimizer.step()

        total_train_loss += loss.item() * num_sample
        if args.avg_loss == 1:
            total_train_loss += loss.item() * num_sample * accumulation_steps
        else:
            total_train_loss += loss.item() * num_sample

        seq_lens.append(input_ids.shape[1])
    avg_train_loss = total_train_loss / total_sample
    return avg_train_loss, seq_lens


def glue_eval(val_dataloader):
    """
    测试准确率，采用倒数第3个token的输出作为答案，计算准确率. 这是因为输出会是 A<EOS>\n, 所以倒数第3个token的logits才是答案的logits
    """
    model.eval()  # 设置模型为评估模式
    total_eval_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_prediction=[]
    with torch.no_grad():
        for i,batch in enumerate(val_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if i==0:
                print('==============================')
                print('')
                print('here is an example of the val data sample')
                print(tokenizer.decode(labels[0][-2:]).strip().lower())
                print('')
                print('==============================')


            outputs = model(input_ids=input_ids, attention_mask=attention_mask,labels=labels)
            loss = outputs.loss

            total_eval_loss += loss.item()

            # 计算准确率
            logits = outputs.logits
            answer_logits = logits[:, -3, :]  # 倒数第3个输出即为答案
            predictions = torch.argmax(answer_logits, dim=-1)

            right_answers, total_answers,text_prediction = compare_gorund_truth(predictions, labels,tokenizer)

            correct_predictions += right_answers
            total_predictions += total_answers
            all_prediction+=text_prediction

    avg_eval_loss = total_eval_loss / len(val_dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_eval_loss,accuracy,all_prediction


def compare_gorund_truth(predictions, ground_truth, tokenizer):
    """

    :param predictions: 预测出的单词id
    :param ground_truth: 答案单词id
    :return:
    """
    text_answer = []  # 保存自然语言形式的模型输出
    right_answers = 0
    total_answers = predictions.size(0)



    for i in range(predictions.size(0)):
        answer_token = tokenizer.decode(predictions[i]).strip().lower()
        text_answer.append(answer_token)
        try:
            if answer_token == tokenizer.decode(ground_truth[i][-2]).strip().lower():
                right_answers += 1


        except:
            print(ground_truth[i])
            print(input_ids[i])

    return right_answers, total_answers, text_answer



all_pred=0
all_acc=0
all_loss=100
all_val_pred=[]
best_dev_loss=100000
best_dev_acc=0
best_dev_epoch=0



for e in range(args.epochs):
    torch.cuda.reset_peak_memory_stats(device)

    print('training epoch={}'.format(e+1))
    start_time = time.time()
    train_loss,seq_lens=train(train_loader)
    start_time2 = time.time()
    
    print('training epoch{} done, time={}, train_loss={:.4f}'.format(e + 1,start_time2-start_time,train_loss))

    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"Peak memory usage: {peak:.2f} MB")


    val_loss, val_acc, val_pred2 = glue_eval(valid_loader)
    start_time4 = time.time()
    print('val_normal epoch={}done, time={}, val_loss={:.4f},val_acc={:.4f}'.format(e + 1, start_time4 - start_time2, val_loss,
                                                                             val_acc))

    if val_acc > best_dev_acc:
        best_dev_acc=val_acc
        best_dev_epoch=e

print('--------')
print('the best dev acc is {:.4f}, epoch={}'.format(best_dev_acc,best_dev_epoch))
