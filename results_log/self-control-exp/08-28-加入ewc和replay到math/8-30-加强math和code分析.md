sults/metrics/cov_mix_audit_ifmath.json
inputs   : if=results/cov/cov_flan_chat_20k_qv.pt, math=results/cov/cov_math_metamath30k_qv.pt
current  : if=0.5000  math=0.5000
layers   : 56 shared of [56, 56]

==== provenance: what each C actually averaged over ====
  -- blocking --
     model_name               if='Qwen3-1.7B'  math='Qwen3-1.7B'
     target_modules           if=['q_proj', 'v_proj']  math=['q_proj', 'v_proj']
     cov_normalization        if='none'  math='none'
     cov_norm_eps             if=1e-06  math=1e-06
  -- composition --
   * max_len                  if=512  math=2048
   * truncation_side          if='right'  math='left'
     use_chat_template        if=1  math=1
     include_target_in_chat   if=1  math=1
   * require_target           if=0  math=1
  -- context --
     dataset_name             if='Muennighoff/flan'  math='Muennighoff/flan'
     dataset_path             if=''  math=''
     data_files               if='/scratch/weiliu87/student/czq/Onereplay/datasets/flan/train/*.jsonl'  math='/scratch/weiliu87/student/czq/Onereplay/results/replay/metamath_cmath_30k_*_selfdistill_seed1.jsonl'
     max_samples              if=20000  math=0
     sample_strategy          if='uniform'  math='uniform'
     sample_shuffle           if=1  math=0
     sample_seed              if=1  math=1
     pool_rows                if=20000  math=25114
     pool_fingerprint         if='9dd3398adf2cf39532cf85d9150b7c6051403261e8b074acac5a0316eed2feb2'  math='33d447c6f1e76894e8af96a7cc9f2e7c6d8f4d2bcad95536f2e4f09056a15968'
  -- token population --
    if          2,983,281 tokens over  20,000 rows  =  149.2 tokens/row
    math       10,715,524 tokens over  25,114 rows  =  426.7 tokens/row
    tokens/row is the composition signal: a domain whose answers are short contributes a C dominated by prompt tokens, one whose answers are long contributes a C dominated by answer tokens, and 0.5/0.5 equalizes those two average tokens -- not two corpora and not two row counts.

==== scale: trace per layer, against the reference domain ====
  -- math / if --
    global trace ratio : 0.9579x
    r_l across layers  : min=0.801 P10=0.810 median=0.901 P90=0.993 max=1.007
    n_l = r_l / global : min=0.836 P10=0.845 median=0.941 P90=1.037 max=1.052
    spread (max/min)   : 1.3x
    -> tight: one scalar weight balances every layer within ~2x.
  equal-trace weights  : if=0.4892  math=0.5108
    equal trace only means equal mass under an isotropic DeltaW, and most of that mass sits in directions both domains share.

==== influence: penalty and gradient under a real DeltaW ====
  adapter: cs_vanilla_seed1   layers matched: 56 / 56
    reg(C_if    ) = 4.618166e+02   ||DeltaW C||_F = 5.113852e+03
    reg(C_math  ) = 3.613666e+02   ||DeltaW C||_F = 5.445075e+03
    ratio math/if = 0.7825
  equal-penalty weights : if=0.4390  math=0.5610
  gradient angle        : energy-weighted cos = 0.8054
    per layer           : min=0.726 P10=0.787 median=0.890 P90=0.936 max=0.953
    -> the two domains pull DeltaW in measurably different directions, so the mixing weight genuinely selects which one gets protected.

==== directions: who owns each direction, and what the mix gives them ====
  layers analyzed: 56 of 56 (stride 1), masses in units of penalty under the probe DeltaW
  retained rank per layer: 256 (median), capturing if=82.5% math=87.9% of the trace
  mu = mass(math) / mass(if) per direction: min=0.005 P10=0.181 median=0.707 P90=6.504 max=340.632
  direction census at tau=3:
    if-unique          3332 directions (23.2%), holding if=79.5% and math=7.9% of that domain's mass
    shared             8278 directions (57.7%), holding if=18.3% and math=15.4% of that domain's mass
    math-unique        2726 directions (19.0%), holding if=2.2% and math=76.7% of that domain's mass
  distinctive mass       : if=2.005e+04  math=1.523e+04   (0.76x apart)
    this is the number the trace ratio cannot see: the shared high-energy directions dominate the trace, so two domains can agree on trace while their distinctive subspaces differ by much more.
  under the current mix (if=0.5000  math=0.5000):
    protection to if-unique directions = 1.081e+04
    protection to math-unique directions = 7895
    imbalance = 0.730x in favour of if
  equal-unique weights   : if=0.4120  math=0.5880
    this is the weight the 0.5/0.5 mix was never checked against: it equalizes protection on the directions that distinguish the two domains, not on the mass they share.
  reconciliation with the influence section:
    sum of if direction masses / reg(C_if) = 0.9750
    sum of math direction masses / reg(C_math) = 0.9810

==== candidates: weights, the lambda that keeps strength fixed, and the balance ====
  current        if=0.5000  math=0.5000   lambda=0.03   unique-imbalance=0.730x
  equal-trace    if=0.4892  math=0.5108   lambda=0.03008   unique-imbalance=0.759x
  equal-penalty  if=0.4390  math=0.5610   lambda=0.03045   unique-imbalance=0.908x
  equal-unique   if=0.4120  math=0.5880   lambda=0.03066   unique-imbalance=1.000x
  lambda is derived so that lambda * reg matches the current mix under the probe DeltaW. Without it a weight change moves strength and balance at once and neither effect can be read off the result.
  unique-imbalance is the prediction to falsify: train the candidates, then compare normalized retention (score - vanilla) / (base - vanilla) on each domain. Equal influence is a claim about that ratio, not about the matrices.

wrote results/metrics/cov_mix_audit_ifmath.json


当前工作：发现EWC half比我们的方法还好，但是EWC half从理论上来说是明显if更强的，所以我怀疑，是不是更强的if+一定的math，在gsm/math这样的数据集上能表现更好，即当前lora训练是没有破坏模型数学能力，下降还是更多聚焦在模型的if能力下降，答题格式出现混乱了，所以我评估了两个EWC的if能力，看是不是half的if能力更强，以及onereplay和EWC一样，if：math为：3；1的强度，看表现有没有更好。

code方面，我发现了opnecoder_educational数据集，这个数据集几乎和mbpp一样，都是instruction描述问题，code给出解题代码，tests cases给出测试样例，。为了考虑heaval，我打算有10k题目改造为heaval的格式，即：
def ...():
  """
  描述问题
  """
给出解答code
这样的形式，希望能在heval上加强表现。