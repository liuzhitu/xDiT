# xDiT_WAN2.1_TP

基于 [xDiT](https://github.com/xdit-project/xDiT) 的 **Wan2.1 T2V Tensor Parallel（TP，张量并行）推理实现**。本项目在 xDiT 原生模型适配体系中，为 Wan2.1 T2V 1.3B / 14B 接入 FFN、SelfAttention 和文本 CrossAttention 的参数与计算分片，并适配 TP 与 Ring、Ulysses、混合 USP 及 DP 的组合。

**当前状态：TP 源码接入与人工审阅已完成；云端正确性、模型推理与性能验证尚未执行。计划先验证 1.3B，14B 共用实现但未运行验证。当前没有加速比或显存收益结论。**

## 代码与文档入口

完整 TP 实现保存在 [`codex/wan21-tp`](https://github.com/liuzhitu/xDiT_WAN2.1_TP/tree/codex/wan21-tp) 分支。默认分支 `main` 的 README 介绍本项目，其源码仍沿用该分支的上游历史；运行 TP 时请检出实现分支并记录实际 commit。

- [详细设计、接口契约与待验证项目](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/docs/wan_tp/README.md)
- [单卡 / TP2 正确性与性能采集协议](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/docs/wan_tp/benchmark.md)：脚本已编写，命令待云端执行。

实现基线为上游 commit `97a4425c68071e1fdf04d40f468015d30be60f41`，不代表 xDiT 最新版本的能力。实现分支已保存完整源码，无需再应用旧项目的累计补丁。

## 我们实现了什么

| 模块 | 本项目的实现 |
| --- | --- |
| 原生 runner 接入 | Wan2.1 T2V 1.3B / 14B 共用加载完成后的分片步骤，复用原生 TP 进程组和推理入口 |
| FFN TP | 第一层 Linear 按输出通道切分，第二层按输入通道切分；保留原生 GELU、Dropout 并归约输出 |
| SelfAttention TP | Q/K/V 按 heads 对应的通道切分，输出投影采用 Row 分片；复用原生 RoPE 和 USP processor |
| 文本 CrossAttention TP | 分片 Q/K/V 和输出投影，保留完整文本序列及原生文本条件语义 |
| Q/K RMSNorm | 跨 TP ranks 汇总平方和，按原始全维计算均方，避免只对本地通道归一化 |
| TP × SP | 从原生 SP 成员列表建立 Ring / Ulysses 子组，适配混合 USP 与子组销毁 |
| TP × DP | 保持 DP 副本独立，每个副本只由代表 rank 提交视频，避免 TP 复制输出 |
| 视频 padding K/V | 传递有效视频长度，在 Ulysses 交换后及 Ring 每步计算时裁掉无效 K/V，通信仍等长 |

embedding、普通归一化、调制与最终输出 head 仍在 TP ranks 上复制。本项目分片的是 Transformer 中的 FFN 与 Attention，并非所有模型组件。

## 如何接入 xDiT

保留 xDiT 原生加载、runtime、进程组、参数放置和推理流程。在模型加载及参数放置完成后，对每个 Transformer block 应用 TP，再继续原生配置和生成流程：

```text
加载 Pipeline
  → 初始化 runtime 和进程组
  → 原生参数放置
  → Wan2.1 T2V 加载完成步骤：分片 FFN、attn1、attn2
  → 原生 offload / compile / cache 配置
  → 原生 runner 推理、scheduler 更新与 VAE 解码
```

调用位置是 `xFuserWan21T2VModel._post_load_and_state_initialization()`。1.3B runner 继承共用 T2V runner 的配置检查和分片步骤；14B 使用共用 runner。维度、heads、层数及参数 dtype 来自实际加载模型，没有为两个参数规模分别写切分实现。`TP=1` 保留原生路径。

选择在加载完成后切分，是为了复用 xDiT 已有生命周期。**加载阶段仍持有完整 Transformer，不能宣称本实现降低了加载峰值。** 权重切片使用独立连续存储，避免分片参数继续引用完整权重的底层存储。

## FFN 的参数与计算分片

设 hidden 维度为 `D`、FFN 中间维度为 `M`、TP degree 为 `P`：

```text
输入 [..., D]，在 TP 组内复制
  → Column Linear：[..., M/P]
  → 原生 GELU / Dropout
  → Row Linear：每卡得到 [..., D] 部分结果
  → TP SUM AllReduce
  → 完整输出 [..., D]
```

PyTorch Linear 权重排列为 `[out_features, in_features]`。Column 按 `dim=0` 切权重并同步切 bias；Row 按 `dim=1` 切权重，完整 bias 在各卡保留，但只有 TP 组内 rank 0 在 `F.linear` 中贡献一次 bias，随后归约。这防止每卡重复贡献 bias，并沿用项目对通信后单独加 bias 的数值顺序处理；分布式结果仍不保证与单卡逐位一致。

FFN 中间维度必须整除 TP degree。每次 FFN forward 有一次输出 AllReduce，输入和输出保持完整 hidden 维度，继续使用原生残差与调制计算。

## Attention TP 与全维 Q/K RMSNorm

### SelfAttention

每个 TP rank 持有连续的一段 heads。Q/K/V 投影按输出通道切分，输出投影按输入通道切分：

```text
视频 hidden states
  → 本地 Q/K/V 投影
  → Q/K 全维 RMSNorm，各一次统计 AllReduce
  → 本地 heads + 原生 RoPE
  → 原生 attention / USP
  → Row 输出投影 + 一次 TP SUM AllReduce
```

Q/K 的 RMSNorm 原本作用于完整投影通道。切分后若只统计本地通道，就会改变模型语义。本实现计算本地平方和，再在 TP 组内求和，除以原始 `normalized_shape`，使用原始 eps 归一化；affine weight 随通道切分。FP16/BF16 的统计、归一化和权重乘法在 FP32 完成，最后恢复输入 dtype，参考 PyTorch 2.8 RMSNorm 的运算顺序。

Attention 的 `heads`、`inner_dim` 和 `kv_inner_dim` 更新为本地尺寸，Transformer config 保留全局模型尺寸。SelfAttention 的 `head_perm` 重建为本地 head 索引，避免非零 TP rank 保留全局偏移而越界。原生 RoPE 坐标和 SP 序列切分继续由 Wan wrapper / processor 负责。

### 文本 CrossAttention

视频 Q 和文本 K/V 按通道分片，各卡仍持有完整文本序列，文本长度不会被替换为视频长度。Q 与 K 的 RMSNorm 分别使用视频序列和文本序列的统计形状。计算继续走原生文本 attention 路径，不增加 RoPE，也不对文本 K/V 执行视频 USP 序列交换；输出投影归约后恢复完整 hidden 维度。

每种 Attention 有两次 RMSNorm 统计 AllReduce 和一次输出 AllReduce。因此当前独立 Q/K/V 路径中，一个包含 FFN、SelfAttention 和 CrossAttention 的 block forward 有 **7 次 TP AllReduce**。这是源码通信位置的计数，尚非性能测量结果。

## TP 与 SP / DP 的组合

保持原生 `tp-sp-pp-cfg-dp` rank 排列。TP ranks 切通道，SP ranks 切视频序列；同一 TP 组内 SP 坐标相同。Ring / Ulysses 子组从原生 SP 成员列表建立，并复用 xDiT coordinator 与 YunChang 接口；销毁时释放子组并清空对应引用。

| 组合 | 实现与约束 | 运行验证 |
| --- | --- | --- |
| TP | FFN、SelfAttention、文本 CrossAttention 分片 | 待执行 |
| TP × Ring | 固定 TP heads 分片，Ring 在 SP 子组内交换视频 K/V | 待执行 |
| TP × Ulysses | SelfAttention heads 必须整除 `TP × Ulysses` | 待执行 |
| TP × 混合 USP | 按原生 SP 排列组合 Ulysses 与 Ring 子组 | 待执行 |
| TP × USP × DP | 各 DP 副本独立，增加 TP rank 0 输出条件以避免重复视频 | 待执行 |
| TP × 共享 DiT ranks 的并行 VAE | 复用原生 VAE 组和解码流程 | 待执行 |

CrossAttention heads 只需整除 TP degree；Ring degree 不加入 head 切分乘数。TP2×Ring2 的组成员人工推导为 TP 组 `[0,1]`、`[2,3]`，SP/Ring 组 `[0,2]`、`[1,3]`。这只是布局说明，尚未运行验证。

输入仍由原生 runner 按 DP 分配 prompts、创建相同 seed 的 generator。TP 组内相同视频片段和文本条件是计算契约；不增加跨所有 ranks 的输入广播。输出继续经过原生 Head、SP gather/unpad，再交给 scheduler / VAE；收集视频时只由 `SP=0、TP=0、CFG=0` 的代表提供真实结果。

## 视频 padding 的处理

SP 需要等长通信张量，但补齐的无效视频 K/V 不应参与 attention softmax。Wan wrapper 在补齐前取得有效视频 token 数，通过 SelfAttention kwargs 传递 `valid_kv_len`：

- 纯 Ulysses：AllToAll 后裁掉 K/V 尾部 padding。
- Ring / 混合 USP：每步按 K/V 块来源计算有效前缀，只裁剪提交给 attention backend 的 K/V；通信保持等长。
- Q 与输出保留补齐长度，继续原生交换与最终 gather/unpad。连续请求不需要 padding 时将有效长度状态恢复为 `None`。
- 完全无效的 Ring K/V 块在通信前明确报错；当前没有修改 PyTorch 的 Ring 合并过程，也没有另写 Ring 实现。

该处理适用于当前 Wan wrapper 的 `TP=1` 和 `TP>1` 路径，目标是密集、非 causal attention。Ring 与 packed K/V indices 同时使用的尾部 padding 明确拒绝。稀疏后端保留自身布局处理，尚无新增正确性验证。

## 支持边界

- 新增 TP 范围为 **Wan2.1 T2V 1.3B / 14B 推理**；I2V、VACE、Wan2.2 和训练未纳入此次新增支持。
- 已融合 Q/K/V 投影在 TP 接入阶段明确拒绝；当前针对原生独立投影路径。
- Transformer GEMM 量化布局未适配；TP 与 FSDP 继续遵守原生互斥限制。
- Wan T2V 的 CFG 并行与 PipeFusion 遵守原生能力声明。普通 CFG guidance 与 CFG 并行是不同概念。
- 独立 VAE ranks 的加载和调度未适配；offload、compile、cache、稀疏 attention 及上述并行组合仍需分别验证。

## 正确性与性能验证计划

本机只进行源码阅读、编辑和人工 diff 审阅，未运行编译、import、lint、CPU/GPU 测试或模型推理。云端首先验证 1.3B：

1. 分层检查 Column / Row 权重覆盖、bias 仅贡献一次、全维 Q/K RMSNorm、head 排列与 RoPE，并检查归约后 TP ranks 输出一致。
2. 对齐同版本原生单卡的 DiT 输出与最终 latent；记录 dtype 和明确的误差阈值，不以逐位一致作为默认要求。
3. 验证 Ring、Ulysses、混合 USP、DP 输出数量、视频 padding、连续请求和子组生命周期。
4. 正确性后再比较单卡 / TP2 的 DiT、denoising 和完整生成区间，分别记录显存、通信、offload 配置与双卡拓扑。

已有 benchmark 工具包含 `capture`、`compare`、`measure`、`profile` 四种模式，区分正确性复制开销、正式计时和 profiler 开销。固定协议为一次 17 帧预热与两次 81 帧正式生成，832×480、50 steps。脚本尚未执行，两个生成样本不能证明稳定的服务尾延迟。1.3B 的后续结果不能写成 14B 已验证。

## 推荐源码阅读位置

以下链接均指向 TP 实现分支：

| 文件 | 阅读目的 |
| --- | --- |
| [runner_models/wan.py](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/xfuser/model_executor/models/runner_models/wan.py) | TP 能力声明、1.3B / 14B 共用路径和加载完成后的调用位置 |
| [transformers/wan_tp.py](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/xfuser/model_executor/models/transformers/wan_tp.py) | 权重切分、Row bias、分布式 RMSNorm 和本地元数据 |
| [core/distributed/parallel_state.py](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/xfuser/core/distributed/parallel_state.py) | SP 子组成员、创建顺序和销毁流程 |
| [transformers/transformer_wan.py](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/xfuser/model_executor/models/transformers/transformer_wan.py) | 视频有效长度、序列 / RoPE 切分和 gather/unpad |
| [layers/usp.py](https://github.com/liuzhitu/xDiT_WAN2.1_TP/blob/codex/wan21-tp/xfuser/model_executor/layers/usp.py) | Ulysses / Ring 的 padding 裁剪与空块拒绝边界 |

## 上游与参考

本项目沿用 xDiT 基础框架、模型加载和推理设施；上述 Wan TP 接入及组合适配是本 fork 的新增工作。上游通用能力和安装说明见 [xDiT](https://github.com/xdit-project/xDiT)。Wan / Attention 结构阅读参考 Diffusers `v0.35.2`，RMSNorm 数值顺序参考 PyTorch `v2.8.0`；这些是源码参考版本，不是已验证环境声明。视频 padding 处理参考 xDiT [PR #912](https://github.com/xdit-project/xDiT/pull/912) 的有效长度传递与逐块裁剪思路，未整包引入其其他模型或稀疏后端改动。沿用仓库已有许可证，详见 [LICENSE.txt](LICENSE.txt)。
