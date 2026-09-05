# litgpt 补丁

`open-sci-ref-and-cpt.patch` 是本项目对 litgpt 的全部改动。litgpt 本身不进本仓库
（目录名与 python 包同名，放在 `con-pretrain/` 下会遮蔽已安装的包，见 `.gitignore`），
所以改动以补丁形式保存在这里。

**基线 commit：`7bf2960`**（`docs: fix dead GPT-2 paper link in prepare_dataset (#2294)`）。
换基线要重新生成，别硬套。

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
cd <litgpt 工作区>
git diff > con-pretrain/litgpt-patch/open-sci-ref-and-cpt.patch
```

改了 litgpt 就重新生成一次并提交，否则改动只活在某一台机器的工作区里。
