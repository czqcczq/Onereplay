在EWC和replay0.5加入math/code之前，要提前确认是：
1、EWC的Fif和Fmath的混合比例应该是多少？理论上说，Fif和Fmath的大小会差别很大，所以如果还是0.5同比例混合的话，会导致结果完全偏向一个领域，这个比例需要先得到Fif和Fmath之后检查大小再决定
2、replay中，if任务max len是512，但是同样len在math上，会导致大部分任务被截断，所以max len要先确认后再调整。

关于问题2，分析结果为：
Replay 的 max_len
IF 自蒸馏池
可用：17,429 行
最大总长度：503
所有行在 max_len=512 下完整保留

Math 自蒸馏池
可用：25,114 行
平均总长度：424.7
P95：1,010
P99：1,558
最大：2,008
512：21.6% 被截断，16.2% 完全丢掉问题，明显不合适；1024：4.8% 被截断，3.8% 完全丢掉问题，是合理的性能折中；2048：全部完整，是科学上最干净的选择。
所以在混合比例上，按照mean来计算，max_len=1024下，如果 IF/math 各取同样行数，监督 token 占比会变成IF 21% / math 79%；在 max_len=2048 下，IF : math 行数 = 357.1 : 90.0 ≈ 79.9% : 20.1%，所以如果replay0.5的话，4old是2if+2math，那么结果会是数学为主导，因为当前loss计算是取token平均的。

在replay0.5的训练上，我打算if/math：1比1，以及if/math：0.8：0.2都实现跑一次，挑选结果。

注：由于原先的F_IF在rwth上，要传到hopper太慢了，所以我打算重新采集一个F_IF，所以原先EWC的结果可能要更换一下，更换为我在hopper上新采集的F_IF训练得到EWC。

08-29：当前进度：
完成replay math的job，完成Fif和Fmath的提取，顺便训练得到了hopper集群上的EWC

下一步工作：1、寻找合适的code保护数据集，把code做完。 2、分析合适的EWC权重，完成F math的训练。 3、把当前训练得到的EWC评测一下 4、找一些agentic的数据集，考虑迁移。 5、安全性任务准备重做

关于F check的结果：
1：
---- target: results/fisher/fisher_flan_chat_20k_qv.pt ----
  collection : max_len=512 truncation_side='right' use_bf16=0 require_target=0 sample_shuffle=1
  modules    : ['q_proj', 'v_proj'] (56 layers)
  pool       : rows=20000 fingerprint=9dd3398adf2cf39532cf85d9150b7c6051403261e8b074acac5a0316eed2feb2
  estimator  : assistant_only_sequence_sum_diagonal_empirical_fisher (reduction=sum)
  scale      : mean=9.997346e-02 max=1.281584e+05
  N          : 20000 rows, 1408 with zero supervised tokens
  tokens     : mean=24.1 median=8 max=494
  ESS        : 2599 / 20000 (ratio 0.130) -- how many rows F actually rests on
  length_exp : a=0.214 in ||g||^2 ~ T^a (1 = uncorrelated per-token grads, 2 = perfectly aligned)
  prompt mask: 6 prefix mismatches (want ~0)
---- reference: results/fisher/fisher_math_metamath30k_qv.pt ----
  collection : max_len=2048 truncation_side='left' use_bf16=0 require_target=1 sample_shuffle=0
  modules    : ['q_proj', 'v_proj'] (56 layers)
  pool       : rows=25114 fingerprint=33d447c6f1e76894e8af96a7cc9f2e7c6d8f4d2bcad95536f2e4f09056a15968
  estimator  : assistant_only_sequence_sum_diagonal_empirical_fisher (reduction=sum)
  scale      : mean=2.644659e-02 max=9.369947e+03
  N          : 25114 rows, 0 with zero supervised tokens
  tokens     : mean=359.1 median=280 max=1793
  ESS        : 17517 / 25114 (ratio 0.698) -- how many rows F actually rests on
  length_exp : a=0.434 in ||g||^2 ~ T^a (1 = uncorrelated per-token grads, 2 = perfectly aligned)
  prompt mask: 0 prefix mismatches (want ~0)
---- scale comparison ----
  mean(F) / mean(F_ref)       = 3.78x
  mean supervised token ratio = 0.07x
  T^a prediction for a in [1,2]: 0.1x to 0x -- measured 3.8x
  equal coefficients would let this file dominate the reference 4x. C is a token mean so its 0.5/0.5 mix is genuinely equal-weight; F is a sequence sum so it is not, and the mix has to be normalized by scale first.
  inverse-scale weights: ref:this = 3.78 : 1 (normalized 0.7908 / 0.2092)

all checks passed

当前结论：mean（Fif）比 mean（Fmath）大了 3.78 倍，所以如果考虑加权的话，这个数的反比是可以纳入考虑范围的，但是还有一个问题就是，其实两个F的分布都比较尖锐，最大最小差别大，所以需要加一个逐层诊断，看看在各层的情况。

---- target: results/fisher/fisher_flan_chat_20k_qv.pt ----
  collection : max_len=512 truncation_side='right' use_bf16=0 require_target=0 sample_shuffle=1
  modules    : ['q_proj', 'v_proj'] (56 layers)
  pool       : rows=20000 fingerprint=9dd3398adf2cf39532cf85d9150b7c6051403261e8b074acac5a0316eed2feb2
  estimator  : assistant_only_sequence_sum_diagonal_empirical_fisher (reduction=sum)
  scale      : mean=9.997346e-02 max=1.281584e+05
  N          : 20000 rows, 1408 with zero supervised tokens
  tokens     : mean=24.1 median=8 max=494
  ESS        : 2599 / 20000 (ratio 0.130) -- how many rows F actually rests on
  length_exp : a=0.214 in ||g||^2 ~ T^a (1 = uncorrelated per-token grads, 2 = perfectly aligned)
  prompt mask: 6 prefix mismatches (want ~0)
---- reference: results/fisher/fisher_math_metamath30k_qv.pt ----
  collection : max_len=2048 truncation_side='left' use_bf16=0 require_target=1 sample_shuffle=0
  modules    : ['q_proj', 'v_proj'] (56 layers)
  pool       : rows=25114 fingerprint=33d447c6f1e76894e8af96a7cc9f2e7c6d8f4d2bcad95536f2e4f09056a15968
  estimator  : assistant_only_sequence_sum_diagonal_empirical_fisher (reduction=sum)
  scale      : mean=2.644659e-02 max=9.369947e+03
  N          : 25114 rows, 0 with zero supervised tokens
  tokens     : mean=359.1 median=280 max=1793
  ESS        : 17517 / 25114 (ratio 0.698) -- how many rows F actually rests on
  length_exp : a=0.434 in ||g||^2 ~ T^a (1 = uncorrelated per-token grads, 2 = perfectly aligned)
  prompt mask: 0 prefix mismatches (want ~0)
---- scale comparison ----
  mean(F) / mean(F_ref)       = 3.78x
  mean supervised token ratio = 0.07x
  T^a prediction for a in [1,2]: 0.1x to 0x -- measured 3.8x
  equal coefficients would let this file dominate the reference 4x. C is a token mean so its 0.5/0.5 mix is genuinely equal-weight; F is a sequence sum so it is not, and the mix has to be normalized by scale first.
  inverse-scale weights: ref:this = 3.78 : 1 (normalized 0.7908 / 0.2092)
---- per-layer mass (sum target / sum reference) ----
  layer mass = the layer's weight in the penalty under a uniform dW; r_l = mass(target,l) / mass(reference,l)
  shared layers            : 56 (56 with nonzero reference)
  global mass ratio        : 3.780x  (target / reference)
  r_l across layers        : min=0.542 P10=2.168 median=4.678 P90=6.598 max=11.414
  target dominates (r_l>1) : 55 / 56 layers
  most reference-heavy     : model.layers.18.self_attn.q_proj=0.54, model.layers.0.self_attn.v_proj=1.30, model.layers.17.self_attn.q_proj=1.39
  most target-heavy        : model.layers.5.self_attn.q_proj=7.69, model.layers.26.self_attn.q_proj=9.03, model.layers.20.self_attn.q_proj=11.41
  equal-contribution mix   : target=0.2092 reference=0.7908  (bigger matrix gets the smaller coefficient)
  scalar-weight stability  : n_l = r_l / global_ratio, where 1.0 means the global weight already balances that layer
    n_l across layers      : min=0.143 P10=0.573 median=1.237 P90=1.745 max=3.019
    spread (max/min)       : 21.1x
    -> wide: the average balance is not what each layer gets. A single scalar weight will over-protect some layers and under-protect others; consider whether the mix should be judged on retention rather than mass.

