# xDiT 原生 Wan TP 接入：FFN、SelfAttention 与文本 CrossAttention

状态：源码已编辑，尚未执行云端正确性或性能验证。当前 Wan DiT 的 FFN、SelfAttention 和文本 CrossAttention 均已接入 TP，embedding、普通归一化、调制和最终输出 head 仍复制。已编写 TP×USP 的组成员适配及 TP×DP 输出去重；不能将源码接入表述为正确性或性能验证通过。

## 参考版本与交付方式

- xDiT：`97a4425c68071e1fdf04d40f468015d30be60f41`。
- 独立源码项目：`D:/研究生/ai_infra/xdit-wan-tp`；GitHub fork：`https://github.com/liuzhitu/xDiT_WAN2.1_TP`；实现分支：`codex/wan21-tp`。
- Diffusers Wan/FFN/Attention 结构阅读参考：`v0.35.2`；PyTorch RMSNorm 数值顺序参考：`v2.8.0`。这些不是已安装环境的版本声明。正式云端运行前仍需确定 Diffusers、YunChang 和模型 checkpoint 的具体版本。
- 交付方式：本 fork 的 `codex/wan21-tp` 分支直接保存完整源码及修改历史，基于上述固定 commit。旧 `wan_parallel` 仓库中的累计补丁保留为迁移备份，不需要在本分支上重复应用。

恢复实现时，clone 本 fork 并检出 `codex/wan21-tp` 的指定提交；原生对照则使用上游固定 commit。记录两种源码的 SHA，使用相同环境和权重。继续从 xDiT 原生 `xdit` / `xfuser.runner` 入口运行，不增加自定义推理入口。

## 目标与范围

将 Wan2.1 T2V 的 1.3B、14B FFN、SelfAttention 与文本 CrossAttention 接入 xDiT 已有 TP 组和模型生命周期，使用同一份按实际参数维度切分的实现。用户已接受加载阶段持有完整 Transformer，因此保留原生加载器、参数放置和 runtime 初始化顺序。

共用 `xFuserWan21T2VModel` runner 声明 TP，1.3B runner 继承其配置检查与加载完成步骤；14B 直接使用共用 runner。维度、heads、层数和参数 dtype 从实际加载的模型读取，没有新增 1.3B/14B 分片分支。`TP=1` 不创建本阶段包装。Wan2.2 T2V 显式保留原有 `TP=False`，I2V、VACE 等不同结构继续按上游能力声明运行。新模块属于推理路径，不提供训练支持。

## 修改前与修改后

原顺序：加载 Pipeline → 初始化 runtime/进程组 → 原生参数放置 → 配置 offload、compile、cache → 原生生成。

新顺序：加载 Pipeline → 初始化 runtime/进程组 → 原生参数放置 → **切分 Wan FFN、SelfAttention 与文本 CrossAttention** → 配置 offload、compile、cache → 原生生成。

接入位置为 `xFuserWan21T2VModel._post_load_and_state_initialization()`。该方法先执行父类的原生加载完成步骤，保留 scheduler 设置，再包装每个 block 的 FFN，并用同一个 `parallelize_wan_attention()` 分片 `attn1` 和 `attn2`。1.3B 与 14B 共用该执行路径，没有提前创建进程组，没有修改通用 checkpoint loader。

每个 FFN 保留原生 `FeedForward.forward()` 对 `net` 的遍历，保留 GELU、Dropout 的原有语义。只改变第一层 Linear 的参数形状，并用 Row 包装替换第二层 Linear。

设实际 hidden 维度为 D，FFN 中间维度为 M，TP degree 为 P：`[...,D] → Column [...,M/P] → GELU/Dropout → Row 部分和 [...,D] → TP SUM AllReduce → [...,D]`。例如 1.3B 的 TP2 为 `1536 → 4480 → 1536`，这些数值不写入分片实现。

SelfAttention：本地 Q/K/V `[...,D/P]` → Q/K 全维 RMSNorm → 本地 H/P heads 与原生 RoPE → 原生 USP → Row 输出投影 → TP SUM AllReduce → `[...,D]`。SP 为 1 时持有完整视频序列；启用 SP 后，每个 rank 持有原生 Wan wrapper 切分的视频 tokens，TP 组内的 SP 坐标相同。1.3B TP2 在 USP 交换前每卡 6 heads，14B TP2 每卡 20 heads；Ulysses 再按其 degree 切 heads，Ring 不再切 heads。

CrossAttention：视频 Q `[B,S,D/P]` 与完整文本 context 投影的 K/V `[B,Ltext,D/P]` → Q/K 全维 RMSNorm → 本地 H/P heads → 原生 `attention()` → Row 输出投影与 TP SUM AllReduce → `[B,S,D]`。文本 context、实际文本长度处理和 backend 选择沿用原生 processor；不增加 RoPE，不调用视频序列的 USP 交换，不将文本 K/V 的长度替换成视频长度。

## 不变量与设计理由

- 第一层权重按 PyTorch `[out,in]` 的 dim=0 切分，bias 同步切分。
- 第二层权重按 dim=1 切分，完整 bias 在各 TP rank 保留；只有组内 rank 0 在 `F.linear` 中贡献 bias，随后归约。这延续旧 TP 中针对通信后 bias 加法误差的处理，不保证分布式结果与单卡逐位一致。
- 分片使用 `clone` 创建独立连续存储。即使 Column 切片本来连续，也不能只用 `contiguous()`：它可能保留完整权重的底层存储。
- TP rank、size 与通信均来自 xDiT 的 `get_tp_group()`，没有新增并行状态或独立进程组生命周期。
- 使用 `GroupCoordinator.all_reduce()` 的返回值，遵循上游接口。
- FFN 中间维度在切分前检查可整除性，防止整除截断造成静默遗漏。
- 两种 Attention 的 heads 均在切分前检查可整除性。SelfAttention 要求 H 可被 TP×Ulysses 整除，CrossAttention 只要求 H 可被 TP 整除，Ring degree 不加入 head 切分乘数。原生 runner 当前使用独立 Q/K/V；若投影已经融合，接入阶段明确拒绝，防止 processor 继续读取未分片的 `to_qkv`/`to_kv`。
- Q/K RMSNorm 保留原模块的全局 `normalized_shape` 与 eps，仅切 affine weight。对 FP16/BF16，统计、归一化及权重乘法在 FP32 完成，再恢复输入 dtype，遵循 PyTorch 2.8 参考顺序。每种 Attention 有两次统计 AllReduce 和一次输出 AllReduce；加 FFN，一次完整 block forward 共 7 次 TP AllReduce。
- `attn1/attn2` 的 `heads/inner_dim/kv_inner_dim` 表示本地计算尺寸；Transformer config 保留全局模型尺寸。SelfAttention 的 `head_perm` 重建为本地 `0..H/P-1`，不能保留非零 TP rank 的全局偏移。CrossAttention 不创建该 buffer，是否使用它由原生 processor 的 `is_cross_attention` 决定。
- Attention processor 的计算、RoPE 和 USP 的通信流程保留。视频 padding 的有效长度通过已有 processor kwargs 传递，USP 在计算边界裁剪 K/V。分布式 RMSNorm 包装不是 `nn.RMSNorm`，因此已有 FlyDSL 类型检查自动选择其参考路径，调用分布式 norm 后继续原生 RoPE；TP=1 保留原有融合选择。
- 模型加载阶段仍持有完整 Transformer；切分完成后才释放原 FFN 参数存储。不能据此宣称已降低加载峰值。

## 本阶段兼容范围

本阶段允许原生 TP 参数选择 FFN + SelfAttention + 文本 CrossAttention TP，并移除早期 TP×SP 禁止条件。全部条目均未运行验证：

| 组合 | 源码处理与边界 |
|---|---|
| TP×Ring | Ring 子组从原生 SP ranks 拆分，固定 TP head 分片；主目标 TP2×Ring2 |
| TP×Ulysses | 固定 TP 坐标后在 SP 内交换序列/heads，要求 H 整除 TP×Ulysses |
| TP×混合 USP | SP rank = Ulysses rank + U×Ring rank，两种子组共用原生 SP 排列 |
| TP×USP×DP | 各 DP 副本的 SP 子组独立；收集视频时增加 TP rank 0 条件，避免重复结果 |
| TP×共享 ranks 的并行 VAE | 保留原生副本内 VAE 组和 DistVAE 解码；DiT 输出已完成 TP 归约及 SP gather，不新增独立 VAE 调度 |
| TP×FSDP | 原生 FullyShardConfig 已禁止两者同时启用；删除重复自定义检查，不改变上游规则 |
| TP×Transformer GEMM 量化 | 保留 runner 的明确拒绝，量化参数布局尚未适配 |
| Wan T2V CFG/PipeFusion | 原生 runner 未声明支持，保留原生拒绝，不新增该模型能力 |

Wan wrapper 在补齐前取得实际视频 token 数，每次 forward 更新 SelfAttention 专用 kwargs 的 `valid_kv_len`，无需补齐时恢复为 `None`。文本 CrossAttention 不读取这个视频长度。纯 Ulysses 在 AllToAll 后裁掉 K/V 的尾部 padding；Ring 与混合 USP 在原生 Ring 每步计算中按 `(ring_rank - step) % ring_size` 确定块来源，只裁剪交给 attention 后端的 K/V 前缀，通信仍使用原来的等长张量。Q 和输出长度保留，继续原生输出交换及最终 gather/unpad。该修复同时作用于 TP=1 和 TP>1 的 Wan wrapper 路径；本轮目标与待验证范围为 Wan2.1 T2V 1.3B/14B，不据此宣称其他 Wan 结构已经验证。

完全无效的 Ring K/V 块仍明确拒绝：若有效长度不大于 `(ring_degree - 1) * exchanged_shard_len`，在 Ring 通信前抛出 `NotImplementedError`。PyTorch 原生 Ring 无条件合并每步输出，不能安全地用零输出和 `-inf` LSE 替代空块；本轮不修改 PyTorch，也不另写 Ring。纯 Ulysses 可以保留全 padding 的原始 SP 分片，因为交换后的完整序列仍有有效 K/V。密集非 causal attention 是本轮修复目标；Ring 与 packed K/V indices 同时使用的尾部 padding 明确拒绝。已有 `thw` 布局的 Sparge/VSA 继续使用自身布局去 padding，不提前裁剪破坏 Q/K/V 等长要求；这些稀疏后端、量化与缓存组合没有新增验证，不能据此宣称 padding 正确性已覆盖。

实现参考 xDiT PR [#912](https://github.com/xdit-project/xDiT/pull/912)，阅读时 head 为 `a13639cd8d979404c88411818d12629925b436c6`。仅接入视频长度传递与逐块裁剪，不整包引入其 padding 行清零、其他模型和稀疏后端改动。正式 832×480、81 帧对应 32760 个 tokens，TP2×Ring2 无需 padding；补齐输入的正确性仍待云端验证。

独立 VAE ranks 的模型加载和调度未纳入本轮适配范围，不能将共享 ranks 的解码复用表述为独立 VAE 进程已兼容。其他配置沿用 xDiT 检查与生命周期，不代表所有组合已被验证。

## 组成员、输入与输出

保持 `RankGenerator` 的 `tp-sp-pp-cfg-dp` 顺序，不改 DP、CFG、PP、TP、模型副本和 VAE 组定义。原生 SP rank 列表是唯一成员来源；在每个列表内部建立 Ulysses/Ring 子组，并把同一组对象交给 SP coordinator 与 YunChang 的兼容接口。所有 WORLD ranks 按相同顺序调用 `new_group`。销毁时释放当前进程所属的两种子组并清空 YunChang 引用，随后执行原生 SP 销毁。

人工推导示例（不是运行验证）：TP2×Ring2 的 TP 组为 `[0,1]`、`[2,3]`，SP/Ring 组为 `[0,2]`、`[1,3]`。增加 DP2 后，第二个副本在 ranks `[4,5,6,7]` 中重复这个布局，两副本不交换视频 tokens。TP2×Ulysses2 的 SP 成员相同，但组内执行原生 AllToAll。

输入仍由原生 runner 按 DP 分配正负 prompts，以相同 seed 创建 generator，再由 Wan wrapper 在 SP 内切分视频与 RoPE。TP 组内同一段视频、同一文本条件是计算契约；不新增 WORLD 广播，以免混合不同 DP 请求。输出继续在原生 Head 后 SP gather/unpad，再进入 sampler/VAE。DP 收集保留全 WORLD 参与，只让 SP=0、TP=0、CFG=0 的代表发送真实视频。

## 验证与后续工作

本轮只进行了源码阅读、编辑与人工 diff 审阅。没有运行语法编译、import、lint、CPU/GPU 测试或模型推理，也没有新增验证脚本。

云端计划仅验证 1.3B。尚需证明：FFN/head 分片覆盖完整通道、非零 Row bias 恰好贡献一次、Q/K RMSNorm 覆盖全维且遵循参考 dtype 顺序、RoPE/head 排列正确、归约后的输出在 TP 组内一致，并与同版本原生 xDiT 单卡在明确 dtype 下数值对齐。应使用不同 TP 分片幅度的 Q/K 输入，避免本地归一化错误被相同统计量掩盖。整条原生 runner 路径、offload/compile、融合回退和内存释放也尚未实际验证。14B 的模型接入与分片路径已纳入源码范围，但未运行验证；1.3B 的验证结果不能表述为 14B 已验证。14B 仍须满足完整 Transformer 加载阶段的资源需求。

CrossAttention 尚需对齐视频与文本长度不同的输入、正负文本条件，以及原生文本处理后的 K/V。其 Q 统计形状是 `[B,S,1]`，K 统计形状是 `[B,Ltext,1]`；两者必须保持独立且各 TP rank 的输入一致。本阶段只记录验证目标，没有新增验证脚本或执行命令。

组合验证还须覆盖：TP2×Ring2、TP2×Ulysses2、可整除 heads 的混合 USP、加入 DP 后不同 prompts 的输出数量、子组销毁/重新初始化，以及共享 ranks 的并行 VAE。视频 padding 须分别覆盖无 padding、部分有效尾块、纯 Ulysses 含全 padding 的原始分片、Ring 完全无效块明确失败，以及连续请求在有/无 padding 间切换。应确认 Ring 计算裁剪不改变通信形状、padding K/V 不贡献 softmax、输出无 NaN，且 TP=1/TP>1 都与同版本单卡数值对齐。12 heads、TP2×Ulysses4 应明确失败。适配结构虽保留 CFG/PP 维度，本阶段 Wan T2V 不运行 CFG/PP。上述均待云端执行。

## 推荐阅读位置

1. `runner_models/wan.py` 的共用 T2V runner 与 1.3B 子类：理解模型规模如何共用分片步骤，以及其他结构的继承边界。
2. `core/distributed/parallel_state.py::_initialize_sequence_parallel_subgroups`：理解 SP 成员如何拆成 Ring/Ulysses，以及 TP/DP 坐标为何保持固定。
3. `models/transformers/wan_tp.py::parallelize_wan_attention`：理解 TP×Ulysses heads 约束与 CrossAttention 差异。
4. `models/transformers/transformer_wan.py::forward`：理解实际视频长度、原生序列/RoPE 切分与 Head 后 gather。
5. `layers/usp.py::_ring_trim_trailing_kv_padding` 与 `USP`：理解 Ulysses 交换后裁剪、Ring 每步块来源，以及完全无效块的拒绝边界。
