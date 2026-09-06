# litgpt 的改动维护在 fork 上

对 litgpt 的改动维护在 <https://github.com/czqcczq/litgpt> 的 `open-sci-cpt` 分支。
集群和本地都从那里拉。**基线 commit 是 `7bf2960`**（`docs: fix dead GPT-2 paper link in
prepare_dataset (#2294)`）。

原先这里放过一份 `open-sci-ref-and-cpt.patch` 快照，09-07 删掉了：它过期不会报错，只会
让人以为改动已经同步。事实上它停在 09-06 03:36，缺 WSD 学习率和裁剪率日志两批改动，而
09-06 那次冒烟作业正是因为「改动只提交在本地工作区、fork 没有」而四个 run 全部秒挂
（`Unrecognized arguments: --train.lr_schedule`），报错打出的 usage 里恰好**不**列缺的
那个参数，看起来像脚本把参数名写错了，方向完全跑偏。

**所以：改完 litgpt 一定要 push 到 fork，别只提交在本地。**
`pbs/02_smoke_pretrain.pbs` 的 preflight 现在会拿脚本自己的参数表比对已装的 litgpt，
版本偏旧直接报错并给出 pull 命令。

litgpt 本身不进本仓库（目录名与 python 包同名，放在 `con-pretrain/` 下会遮蔽已安装的
包，见 `.gitignore`）。本地那份在 `con-pretrain/litgpt/`，是个独立 repo。

## 集群上怎么更新

```bash
cd <REPO_ROOT>/../litgpt-src
git pull
```

前提是当前分支已经在追踪 fork 的 `open-sci-cpt`。**别照抄别处写的远程名**：远程别名是每个
克隆私有的，本地 Windows 那份把上游叫 `origin`、fork 叫 `myfork`，而集群那份直接从 fork
克隆，fork 就叫 `origin`。先 `git remote -v` 看清楚再说。没接上的话：

```bash
git remote -v                                   # 确认 fork 的别名叫什么
git fetch <别名> && git checkout -B open-sci-cpt <别名>/open-sci-cpt
```

`pip install -e .` 装的是软链，拉完不用重装。验一下改动到位了：

```bash
python -c "from litgpt.args import TrainArgs; print(TrainArgs().lr_schedule)"   # cosine
```

目录名用 `litgpt-src` 而不是 `litgpt`：后者会遮蔽 python 包。

## 本地工作区的 CRLF

`con-pretrain/litgpt/` 在 Windows 上很容易被整棵树写成 CRLF，`git diff` 于是报 225 个
文件、4.7 万行，真实改动被彻底淹没，也很容易连噪音一起提交进 fork 历史。
跑 `test_code_CPT/normalize_litgpt_worktree.py` 清掉，它只去行尾、不动内容。

## 改了什么

| 文件 | 内容 |
| --- | --- |
| `config.py` | 注册 `open-sci-ref-v0.02-0.4b-fineweb-edu-1.4t-300B-4096` 这个 Config |
| `scripts/convert_hf_checkpoint.py` | opensci 的权重名映射：q/k/v/o 与 FFN 的 bias、`q_layernorm`/`k_layernorm` → `norm_q`/`norm_k` |
| `pretrain.py` | CPT / 正则器集成、WSD 学习率、裁剪率与梯度范数日志 |
| `args.py` | 对应的新参数 |

`model.py` **没有改动**——vanilla litgpt 已经支持 bias 和 QK-norm，改动只补了
「叫什么名字」和 Config 注册。

## 谁需要装这份改过的 litgpt

| 用途 | 需要？ | 为什么 |
| --- | --- | --- |
| `kres.collect_cov` / `inspect_cov` | 否 | 走 `Config.from_file(model_config.yaml)`，不查注册表 |
| `kres.mix_covariances` | 否 | 只做张量加权，不建模型 |
| HF → litgpt 转换（产出 `lit_model.pth`） | **是** | 需要 `Config.from_name` 和上面那套名字映射 |
| `litgpt pretrain`（CPT 正式训练） | **是** | `pretrain.py` / `args.py` 的改动就在这儿 |

所以只想采 C 的机器，装 pip 版 litgpt + 一份现成的 `lit_model.pth` 就够。
