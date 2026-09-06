#!/bin/bash
# 七条臂的投递清单。**这个文件不投任何东西**，它打印命令让你自己看过再粘。
#
#     bash pbs/submit_arms.sh          # 打印三批命令
#     bash pbs/submit_arms.sh 1        # 只打印第 1 批
#
# 为什么不直接 qsub：第 2、3 批要等前一批的产物（λ 从 vanilla 的检查点反算），
# 一键全投只会让三条 C 臂带着一个没有依据的 λ 跑掉 61 GPU-小时。
#
# =============================================================================
# 队列同时只能跑 6 个作业，而有 7 条臂，所以有一个要排队。谁排队决定总时长：
#
#   把 replay_8b（40.6 h）排最后 -> 它要等第一个 20.3 h 的臂结束才开始 -> 总 61 h
#   把 replay_8b 第一个投        -> 排队的是 20.3 h 的 c_8b        -> 总 40.6 h
#
# 差 20 小时，而且不用拿任何东西去换：早出结果那个诉求由 vanilla + replay_1b + c_1b
# 那组横向对比满足（同样 1B 旧数据，一个重放它、一个用它估 C），三条都是 20~23 小时
# 的短臂，两种排法下都在同一时刻跑完——它们本来就在并行跑，让 replay_8b 先进队列不会
# 推迟它们。
#
# 时间线：
#   t=0     第 1 批：replay_8b / replay_4b / replay_1b / vanilla（都不需要 λ）
#   t≈4h    vanilla 存下第一个检查点 -> 第 2 批：反算 λ，投三个短 run 扫描
#   t≈10h   定下 λ -> 第 3 批：c_1b / c_4b / c_8b
#   t≈41h   全部跑完（被 replay_8b 卡着，λ 那一路完全藏在它的阴影里）
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/weiliu87/student/czq/con-pretrain}"
PBS="${REPO_ROOT}/pbs/03_cpt_train.pbs"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/model/open-sci-ref-v0.02-0.4b-fineweb-edu-1.4t-300B-4096}"
COV_DIR="${COV_DIR:-${REPO_ROOT}/cov}"
PLAN="${PLAN:-${REPO_ROOT}/data/chunks/fineweb_edu/replay_plan.json}"

# 扫描 run 的长度。750 步约 4 小时，已经走完 warmup（3% = 114 步）并进入 stable 段，
# ΔW 长起来了。再短就只能量到 warmup 的暂态
SCAN_STEPS="${SCAN_STEPS:-750}"

want="${1:-all}"
show () { [[ "${want}" == "all" || "${want}" == "$1" ]]; }

if show 1; then
cat <<EOF

# =============================================================================
# 第 1 批：四条不需要 λ 的臂。现在就能投。
#
# 投之前每条先跑一次 preflight（几分钟，不上 GPU）。40 小时的作业错一个参数名，
# 代价是一整个排队周期：
#
#   for a in replay_8b replay_4b replay_1b vanilla; do
#     qsub -N pre_\${a} -v ARM=\${a},PREFLIGHT_ONLY=1 ${PBS}
#   done
#
# **顺序有意义，replay_8b 必须第一个。** 理由见文件头。
# =============================================================================
qsub -N kres_r8b -v ARM=replay_8b ${PBS}
qsub -N kres_r4b -v ARM=replay_4b ${PBS}
qsub -N kres_r1b -v ARM=replay_1b ${PBS}
qsub -N kres_van -v ARM=vanilla   ${PBS}
EOF
fi

if show 2; then
cat <<EOF

# =============================================================================
# 第 2 批：λ 扫描。**等 vanilla 存出第一个 step-* 检查点再做**（约 4 小时）。
#
# 先反算区间。λ 没有自然量纲，搜索空间宽到十个数量级，而两端的失效都不显眼：太小得到
# 一条与 baseline 难以区分的曲线，太大则新域 loss 降不下去、看起来像别的 bug。反算把
# 十个数量级压到约两个，代价是零——|g_reg| 只依赖 ΔW 和 C，两个都在磁盘上：
#
#   ls -d ${REPO_ROOT}/out/vanilla/step-*        # 挑最新那个
#   python -m kres.probe_lambda \\
#       --base ${MODEL_DIR}/lit_model.pth \\
#       --checkpoint ${REPO_ROOT}/out/vanilla/step-XXXXXXXX/lit_model.pth \\
#       --cov ${COV_DIR}/arm_4b.pt \\
#       --log ${REPO_ROOT}/logs/vanilla.<jobid>.log
#
# 它按 ρ 打出一张 λ 表。**扫描固定在 c_4b 这条臂上**，不偏向两端任何一头；定下来的值
# 三条 C 臂通用——各自调 λ 就把「C 更好」和「λ 调得更好」混在一起了。
#
# 把下面三个 <λ> 换成表里 ρ=0.03 / 0.1 / 0.3 那三行。短 run 的 ΔW 比终态小，所以标出
# 的 λ 偏大，方向已知，选的时候从「看起来可接受」的那批里取偏小的一端。
# =============================================================================
qsub -N kres_s1 -v ARM=c_4b,LAMBDA=<λ_小>,SCAN_STEPS=${SCAN_STEPS} ${PBS}
qsub -N kres_s2 -v ARM=c_4b,LAMBDA=<λ_中>,SCAN_STEPS=${SCAN_STEPS} ${PBS}
qsub -N kres_s3 -v ARM=c_4b,LAMBDA=<λ_大>,SCAN_STEPS=${SCAN_STEPS} ${PBS}

# 产物在 out/c_4b_lam<λ>_scan${SCAN_STEPS}/，各自独立，不会互相覆盖。
# 看三件事：新域 val_loss 有没有被压住、probe_loss 有没有比 vanilla 同步长处更低、
# clip 触发率有没有明显高于 vanilla（高了说明惩罚把总范数顶过 max_norm，整个梯度被
# 按比例缩小，等于学新域那部分也一起被压掉——那时 C 臂会在遗忘上好看、在新域上难看，
# 而原因是裁剪伪影不是方法本身）。
EOF
fi

if show 3; then
cat <<EOF

# =============================================================================
# 第 3 批：三条正式 C 臂，**同一个 λ**。
#
# c_4b 的 C 要先合并（c_1b 直接用 seg00.pt，01_collect_cov 已经产出）：
#   python -m kres.mix_covariances --from-plan ${PLAN} --arm 2 --cov-dir ${COV_DIR} --output ${COV_DIR}/arm_4b.pt
#   python -m kres.mix_covariances --from-plan ${PLAN} --arm 3 --cov-dir ${COV_DIR} --output ${COV_DIR}/arm_8b.pt
#
# 按 token 计数加权合并与「直接在并集上采一次」**严格等价**而非近似，所以 3 条嵌套的
# 臂只需 3 次采集而不是 6 次。别手拼 seg%02d，用 --from-plan——两边各拼一次就有各拼错
# 一次的机会。
# =============================================================================
qsub -N kres_c1b -v ARM=c_1b,LAMBDA=<λ> ${PBS}
qsub -N kres_c4b -v ARM=c_4b,LAMBDA=<λ> ${PBS}
qsub -N kres_c8b -v ARM=c_8b,LAMBDA=<λ> ${PBS}
EOF
fi

cat <<'EOF'

# -----------------------------------------------------------------------------
# 跑起来之后看什么
#
#   qstat -u $USER
#   tail -f logs/<臂>.<jobid>.log
#
# 每 200 步一行：
#   iter N step M: val loss ... ppl ... | probe loss ... ppl ... | eval time: ... ms
#   val   = 新域 biomed/val，塑性
#   probe = 旧域 fineweb_edu/probe，遗忘量（相对 step 0 那行涨了多少）
#   test  = 新域 biomed/test，只有最后 "Final evaluation" 那行有
#
# **跨臂画曲线的横轴用 csv 里的 new_tokens，不要用 step。** 各臂 step 里掺的 replay
# 比例不同：同样是第 200 步，vanilla 已经吃了 419M 新域 token，replay_8b 只吃了约
# 210M，另一半是重放。按 step 对齐等于拿「学了两倍新域」的点去比「忘得更少」。
# -----------------------------------------------------------------------------
EOF
