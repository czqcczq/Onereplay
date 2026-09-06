# litgpt 补丁

> **真相源是 fork，不是这个补丁。**
> 对 litgpt 的改动现在维护在 <https://github.com/czqcczq/litgpt> 的 `open-sci-cpt`
> 分支上。集群和本地都从那里拉。这个目录里的 `.patch` 只是个快照，**会过期**。
>
> 09-06 就因为它过期而废掉一次冒烟作业：WSD 那两个参数（`--train.lr_schedule`、
> `--train.lr_cooldown_fraction`）只提交在本地工作区，fork 和补丁都没有，集群上四个
> run 全部以 `Unrecognized arguments` 秒挂。而报错打出的 usage 里恰好**不**列缺的那个
> 参数，看起来像脚本把参数名写错了，方向完全跑偏。
>
> 所以：**改完 litgpt 一定要 push 到 fork**，别只提交在本地。02_smoke_pretrain.pbs 的
> preflight 现在会拿脚本自己的参数表比对已装的 litgpt，版本偏旧直接报错并给出 pull 命令。

`open-sci-ref-and-cpt.patch` 是对 litgpt 的全部改动的补丁形式快照。litgpt 本身不进本仓库
（目录名与 python 包同名，放在 `con-pretrain/` 下会遮蔽已安装的包，见 `.gitignore`）。

**基线 commit：`7bf2960`**（`docs: fix dead GPT-2 paper link in prepare_dataset (#2294)`）。
换基线要重新生成，别硬套。

## 集群上怎么更新

```bash
cd <REPO_ROOT>/../litgpt-src
git fetch myfork && git checkout open-sci-cpt && git pull myfork open-sci-cpt
```

`pip install -e .` 装的是软链，拉完不用重装。验一下改动到位了：

```bash
python -c "from litgpt.args import TrainArgs; print(TrainArgs().lr_schedule)"   # cosine
```

## 本地工作区的 CRLF

`con-pretrain/litgpt/` 在 Windows 上很容易被整棵树写成 CRLF，`git diff` 于是报 225 个
文件、4.7 万行，真实改动被彻底淹没，也很容易连噪音一起提交进 fork 历史。
跑 `test_code_CPT/normalize_litgpt_worktree.py` 清掉，它只去行尾、不动内容。

## 改了什么

| 文件 | 内容 |
| --- | --- |
| `config.py` | 注册 `open-sci-ref-v0.02-0.4b-fineweb-edu-1.4t-300B-4096` 这个 Config |
| `scripts/convert_hf_checkpoint.py` | opensci 的权重名映射：q/k/v/o 与 FFN 的 bias、`q_layernorm`/`k_layernorm` → `norm_q`/`norm_k` |
| `pretrain.py` | CPT / 正则器集成 |
| `args.py` | 对应的新参数 |

`model.py` **没有改动**——vanilla litgpt 已经支持 bias 和 QK-norm，补丁只补了
「叫什么名字」和 Config 注册。

## 谁需要它

| 用途 | 需要补丁？ | 为什么 |
| --- | --- | --- |
| `kres.collect_cov` / `inspect_cov` | 否 | 走 `Config.from_file(model_config.yaml)`，不查注册表 |
| `kres.mix_covariances` | 否 | 只做张量加权，不建模型 |
| HF → litgpt 转换（产出 `lit_model.pth`） | **是** | 需要 `Config.from_name` 和上面那套名字映射 |
| `litgpt pretrain`（CPT 正式训练） | **是** | `pretrain.py` / `args.py` 的改动就在这儿 |

所以只想采 C 的机器，装 pip 版 litgpt + 一份现成的 `lit_model.pth` 就够。

## 怎么应用

```bash
git clone https://github.com/Lightning-AI/litgpt.git litgpt-src
cd litgpt-src
git checkout 7bf2960
git apply ../litgpt-patch/open-sci-ref-and-cpt.patch
pip install -e .
```

目录名用 `litgpt-src` 而不是 `litgpt`：后者会遮蔽 python 包。

## 生成方式

```bash
cd con-pretrain/litgpt
python ../../test_code_CPT/normalize_litgpt_worktree.py     # 先清 CRLF，否则补丁 4.7 万行
git diff 7bf2960 > ../litgpt-patch/open-sci-ref-and-cpt.patch
```

基线要显式写 `7bf2960`：改动现在是**提交**在 `open-sci-cpt` 分支上的，裸 `git diff`
只能看到未提交的部分，早期那份补丁就是这么漏掉 WSD 的。
