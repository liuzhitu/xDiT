# Wan2.1 T2V Tensor Parallel 接入

本 fork 的 `codex/wan21-tp` 分支在 xDiT 原生模型适配体系中实现了 **Wan2.1 T2V 1.3B / 14B 的 Tensor Parallel（TP）源码接入**，共用按实际模型维度和层数切分的实现。

**当前状态：源码实现与人工审阅已完成；云端正确性和性能验证尚未执行。计划先验证 1.3B，尚无加速或显存收益结论。**

- **FFN TP**：Column / Row Linear 分片，保留原生 GELU 和 Dropout；Row bias 由 TP 组内 rank 0 在 `F.linear` 中贡献一次，再进行 AllReduce。
- **SelfAttention TP**：Q/K/V 与输出投影分片，Q/K RMSNorm 跨 TP 汇总全维统计，继续使用原生 RoPE 与 USP。
- **文本 CrossAttention TP**：保留完整文本 K/V 和原生文本语义，分片 heads 并归约输出。
- **TP × SP / DP**：适配 Ring、Ulysses、混合 USP 的子组创建与销毁，以及 DP 输出去重。
- **视频 padding**：传递有效视频长度；纯 Ulysses 在 AllToAll 后裁剪尾部 K/V，Ring 在每步计算中裁剪有效 K/V，通信保持等长。完全无效的 Ring K/V 块明确报错。

保留 xDiT 原生加载、runtime、进程组和推理入口，允许加载阶段持有完整 Transformer。新增 TP 不覆盖 I2V、VACE 或 Wan2.2；稀疏 attention、量化和缓存组合未验证。实现基于上游 commit `97a4425c68071e1fdf04d40f468015d30be60f41`。

详细设计、支持边界、参考来源和待验证项目见 [Wan TP 接入说明](docs/wan_tp/README.md)。本分支已包含完整源码，不需要再应用旧项目的累计补丁。

---

