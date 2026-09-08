"""直接比较若干个 C 之间的差异，验证「用 1B / 4B / 8B 旧数据估出来的 C 几乎一样」。

之前这个结论是间接得来的：probe_lambda 在三个 C 上算出的 |g_reg|(λ=1) 只差 0.07%。
那只说明**一个标量泛函**在三者上取值相同，两个结构完全不同的矩阵也可以做到这点。
这个脚本比矩阵本身。

三个指标，各自回答一个不同的问题：

  ‖ΔC‖_F / ‖C_b‖_F   两个矩阵差多远（整体相对差）
  1 − cos            把尺度差除掉之后还剩多少差（cos = <A,B>/‖A‖‖B‖）。如果相对差
                     不小但 1−cos 极小，说明两个 C 形状一样只是整体缩放差一点，
                     而整体缩放会被 λ 吸收掉，对三条臂的对比无害
  ‖ΔW·ΔC‖/‖ΔW·C_b‖   **真正决定结论的那个**。惩罚梯度是 2λ·ΔW·C，所以两条臂的
                     惩罚力差多少，就等于这个比值。矩阵在某些方向上差很多、但那些
                     方向没有漂移，对训练就没有影响；反过来也一样

== 嵌套：这个比较里最容易搞错的地方 ==

arm_4b 是 mix_covariances 用 seg00 和 seg01 按 token 加权合出来的，**它含有 seg00**。
于是有恒等式

    C_4b − C_seg00 = w₁·(C_seg01 − C_seg00)

也就是说「1B vs 4B」的差被机械地压了 w₁ 倍（seg01 的 token 占比，约 0.75），这个压缩
与「C 收敛了没有」毫无关系，纯粹是合并方式造成的。拿它当「C 饱和」的证据是循环论证。

所以要判断 C 到底收敛没有，必须看**互不相交**的两段：seg00 ↔ seg01 都是独立的 1B 样
本，它们的差就是 1B 这个预算下的采样噪声地板。真正的判据是：

    嵌套对的差 ≈ 噪声地板 × 对应的 w  →  差全部来自采样噪声，C 已经收敛
    嵌套对的差 ≫ 噪声地板 × 对应的 w  →  多出来的是真实的分布差异，没收敛

脚本会在输入允许时自动验证上面那个恒等式（数值上应当完全相等），这同时也是对
mix_covariances 和本脚本比较逻辑的一次交叉校验——它要是对不上，先别信别的数字。

用法：

    # 最有说服力的一组：两个不相交的 1B 段 + 两个嵌套的合并结果
    python -m kres.compare_covariances \
        --inputs cov/seg00.pt,cov/seg01.pt,cov/arm_4b.pt,cov/arm_8b.pt \
        --base model/open-sci-ref-.../lit_model.pth \
        --checkpoint out/vanilla/step-00003000/lit_model.pth

    # 不给 base/checkpoint 就跳过 ΔW 加权那一列，快很多
    python -m kres.compare_covariances --inputs cov/seg00.pt,cov/arm_4b.pt,cov/arm_8b.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kres.covariance import load_covariance_payload  # noqa: E402


def load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    return state


def pair_stats(cov_a, cov_b, keys, drift, dtype):
    """逐层比 A 和 B，返回 (聚合量, 每层相对差)。

    聚合是把各层的平方量加起来再开方，等价于把所有层拼成一个长向量来比——不是各层
    相对差的平均。后者会让一个 d=1024 的小层和一个 d=3840 的大层等权，而惩罚项里
    它们本来就不等权。
    """
    agg = dict.fromkeys(("d2", "a2", "b2", "ab", "tra", "trb", "gd2", "gb2"), 0.0)
    per_layer = []
    for key in keys:
        A = cov_a[key].to(dtype)
        B = cov_b[key].to(dtype)
        D = A - B
        d2, a2, b2 = float(D.pow(2).sum()), float(A.pow(2).sum()), float(B.pow(2).sum())
        agg["d2"] += d2
        agg["a2"] += a2
        agg["b2"] += b2
        agg["ab"] += float((A * B).sum())
        agg["tra"] += float(A.diagonal().sum())
        agg["trb"] += float(B.diagonal().sum())
        if drift is not None and key in drift:
            # 这两个 matmul 是全脚本最贵的部分，用 fp32：它只进一个比值，而 fp32 的
            # 相对误差在 1e-6 量级，远小于我们要分辨的百分之几
            dw = drift[key].to(torch.float32)
            agg["gd2"] += float((dw @ D.to(torch.float32)).pow(2).sum())
            agg["gb2"] += float((dw @ B.to(torch.float32)).pow(2).sum())
        per_layer.append(((d2 / b2) ** 0.5 if b2 > 0 else float("nan"), key))
    return agg, per_layer


def nesting_of(meta_mix: dict, name_other: str) -> float | None:
    """meta_mix 描述的合并结果里，name_other 是不是被含进去了？含了就返回它的权重。"""
    if meta_mix.get("type") != "covariance_mix":
        return None
    for path_str, weight in zip(meta_mix.get("inputs", []), meta_mix.get("weights", [])):
        if Path(path_str).name == name_other:
            return float(weight)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--inputs", required=True, help="逗号分隔的 C 文件，至少两个")
    ap.add_argument("--base", type=Path, default=None, help="基座 lit_model.pth（W₀），给了才算 ΔW 加权那列")
    ap.add_argument("--checkpoint", type=Path, default=None, help="配 --base 用，一般是 vanilla 的检查点")
    ap.add_argument("--per-layer", type=int, default=5, help="每对额外打印差得最大的前几层")
    ap.add_argument("--float32", action="store_true", help="用 fp32 比矩阵（省一半内存，够用）")
    args = ap.parse_args(argv)

    # Windows 控制台默认 GBK，打不出 ‖ΔC‖ 里的记号。集群是 UTF-8，这行是空操作
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    dtype = torch.float32 if args.float32 else torch.float64
    paths = [Path(x.strip()) for x in args.inputs.split(",") if x.strip()]
    if len(paths) < 2:
        raise SystemExit(f"至少要给两个 C 文件，现在只有 {[str(p) for p in paths]}")
    for p in paths:
        if not p.exists():
            raise SystemExit(f"找不到 {p}")

    if (args.base is None) != (args.checkpoint is None):
        raise SystemExit("--base 和 --checkpoint 要么都给，要么都不给")

    # 先只读 metadata（顺带确认文件能打开），矩阵按需加载，内存里最多同时放两个
    metas, keysets = [], []
    for p in paths:
        pl = load_covariance_payload(p)
        metas.append(pl.get("metadata", {}))
        keysets.append(set(pl["covariances"]))
        counts = pl["counts"]
        n_tok = counts[next(iter(counts))]
        src = metas[-1].get("segments") or metas[-1].get("inputs")
        print(f"  {p.name:<16} {len(keysets[-1]):>4} 层   token {n_tok:>15,}"
              + (f"   合自 {[Path(str(s)).stem for s in src]}" if src else ""))
        del pl

    if len(set(map(frozenset, keysets))) != 1:
        base = keysets[0]
        raise SystemExit(
            "各输入的目标层集合不一致，没法逐层比："
            + str({paths[i].name: {"多出": sorted(ks - base)[:3], "缺少": sorted(base - ks)[:3]}
                   for i, ks in enumerate(keysets) if ks != base})
        )
    keys = sorted(keysets[0])

    drift = None
    if args.base is not None:
        base_sd, ck_sd = load_state(args.base), load_state(args.checkpoint)
        drift = {}
        for key in keys:
            wk = f"{key}.weight"
            if wk in base_sd and wk in ck_sd:
                drift[key] = (ck_sd[wk].to(torch.float32) - base_sd[wk].to(torch.float32))
        if not drift:
            raise SystemExit("检查点里一个目标层的 .weight 都没匹配上，键名对不上")
        norm = sum(float(v.pow(2).sum()) for v in drift.values()) ** 0.5
        print(f"\nΔW 取自 {args.checkpoint.parent.name}/{args.checkpoint.name}，"
              f"覆盖 {len(drift)}/{len(keys)} 层，‖ΔW‖ = {norm:.3f}")
        del base_sd, ck_sd

    # 外层加载一次、内层轮换，任一时刻内存里只有两个 C
    results: dict[frozenset, dict] = {}
    t0 = time.perf_counter()
    for i in range(len(paths)):
        cov_i = load_covariance_payload(paths[i])["covariances"]
        for j in range(i + 1, len(paths)):
            cov_j = load_covariance_payload(paths[j])["covariances"]
            agg, per_layer = pair_stats(cov_i, cov_j, keys, drift, dtype)
            results[frozenset((i, j))] = {"agg": agg, "per_layer": per_layer, "pair": (i, j)}
            del cov_j
            print(f"  ... {paths[i].stem} ↔ {paths[j].stem} 比完，累计 {time.perf_counter() - t0:.0f}s")
        del cov_i

    print("\n=== 两两比较（{} 层聚合，A ↔ B 的分母取 B）===".format(len(keys)))
    head = f"{'A ↔ B':<26}{'‖ΔC‖/‖C_B‖':>13}{'1−cos':>12}{'tr_A/tr_B':>12}"
    if drift is not None:
        head += f"{'‖ΔW·ΔC‖/‖ΔW·C_B‖':>20}"
    head += "   嵌套"
    print(head)
    for res in results.values():
        i, j = res["pair"]
        a = res["agg"]
        rel = (a["d2"] / a["b2"]) ** 0.5
        cos = a["ab"] / ((a["a2"] * a["b2"]) ** 0.5)
        line = (f"{paths[i].stem + ' ↔ ' + paths[j].stem:<26}"
                f"{rel:>12.3%}{1 - cos:>12.2e}{a['tra'] / a['trb']:>12.5f}")
        if drift is not None:
            line += f"{(a['gd2'] / a['gb2']) ** 0.5:>19.3%}" if a["gb2"] > 0 else f"{'—':>19}"
        w = nesting_of(metas[j], paths[i].name) or nesting_of(metas[i], paths[j].name)
        # 打的是压缩系数 1−w 而不是被含段的权重 w：读者要拿它去乘噪声地板，
        # 直接给能乘的那个数，省得每次自己换算一步、换错方向
        line += f"   {'是（压缩 %.3f）' % (1 - w) if w is not None else '否'}"
        print(line)

    print("\n  「嵌套=是」那几行的差被合并方式机械地压小了，不能当作 C 收敛的证据；")
    print("  只有「嵌套=否」的行（互不相交的两段）才是这个 token 预算下的采样噪声地板。")

    # 恒等式自检：C_mix − C_i = w_j·(C_j − C_i)，两边的范数必须相等。这既验证了嵌套
    # 修正的算法，也顺带验证了 mix_covariances 的加权确实是按 token 来的
    by_name = {p.name: k for k, p in enumerate(paths)}
    for m, meta in enumerate(metas):
        ins = [Path(str(s)).name for s in meta.get("inputs", [])]
        if meta.get("type") != "covariance_mix" or len(ins) != 2 or not all(n in by_name for n in ins):
            continue
        i, j = by_name[ins[0]], by_name[ins[1]]
        w_j = float(meta["weights"][1])
        lhs = (results[frozenset((i, m))]["agg"]["d2"]) ** 0.5
        rhs = w_j * (results[frozenset((i, j))]["agg"]["d2"]) ** 0.5
        gap = abs(lhs - rhs) / max(rhs, 1e-30)
        verdict = "一致" if gap < 1e-4 else f"对不上（差 {gap:.2%}）——先查 mix_covariances"
        print(f"\n  恒等式自检 {paths[m].stem}：‖C_mix−C_{paths[i].stem}‖ = {lhs:.6g}，"
              f"w·‖C_{paths[j].stem}−C_{paths[i].stem}‖ = {rhs:.6g}   {verdict}")

    if args.per_layer > 0:
        print(f"\n=== 每对里差得最大的 {args.per_layer} 层（按 ‖ΔC‖/‖C_B‖）===")
        for res in results.values():
            i, j = res["pair"]
            worst = sorted(res["per_layer"], reverse=True)[: args.per_layer]
            print(f"  {paths[i].stem} ↔ {paths[j].stem}：")
            for rel, key in worst:
                print(f"    {key:<44}{rel:>10.3%}")

    print("""
怎么读：
  先看「嵌套=否」那一行的 ‖ΔC‖/‖C_B‖——两段互不相交的旧数据，各自独立估出来的 C 差
  这么多，这就是这个 token 预算下的采样噪声。如果它本身只有百分之零点几，那 C 在 1B
  处就已经收敛，再加数据没有可估的东西了，三条 C 臂给出几乎相同的结果是必然的，不是
  实现出了问题。

  再看最后一列 ‖ΔW·ΔC‖/‖ΔW·C_B‖。这是惩罚力的相对差，也是唯一能直接换算成「两条臂
  的训练轨迹会差多少」的量。它若与 ‖ΔC‖/‖C_B‖ 同量级，说明矩阵的差是均匀铺开的；
  它若明显更小，说明两个 C 的差集中在没有漂移的方向上，对训练更加没有影响。

  1−cos 极小而相对差不小，意味着两个 C 只差一个整体缩放。整体缩放会被 λ 完全吸收，
  所以那种情况下三条臂的差异比相对差看起来的还要小。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
