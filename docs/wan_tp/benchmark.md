# Wan2.1 T2V 单卡 / TP2 benchmark

状态：脚本已编写，仅人工审阅；未在本机执行语法、import、测试或模型推理。所有云端命令均待执行。范围只包含同版本原生 xDiT 单卡与本分支 TP2，SP/DP/CFG parallel degree 均为 1；普通 CFG guidance 仍开启。

## 固定实验协议

- 每个运行模式、每个配置：832×480、17 帧、50 steps 预热一次；832×480、81 帧、50 steps 正式生成两次。
- 保留第一次 81 帧结果，不因它承担新形状的初始化而丢弃。17 帧预热不能保证 81 帧形状的缓存已预热。
- 默认 seed=42、guidance=5、flow_shift=3，沿用原生 runner 的 scheduler；记录实际 scheduler 类型与配置。默认 Self/CrossAttention backend 都为 FLASH；固定未量化 eager 路径，关闭 compile/cache/并行 VAE。
- 每个配置保存单独结果目录。驻留模式和 model CPU offload 模式分别对照；单卡 OOM 时记录 OOM，不改变该侧配置后与原 TP2 样本比较。
- 实验对象为 1.3B Diffusers checkpoint，默认模型 revision 来自准备记录。两边共用权重和环境；脚本不校验权重内容哈希。

## 模式与运行成本

| 模式 | 用途 | 可用于正式加速比 |
|---|---|---|
| capture | 原生单卡保存前两个 DiT 调用的完整输入/输出和两次最终 scheduler latent | 否，包含 CPU 张量复制 |
| compare | TP2 完整生成，并用参考固定输入重放原生 DiT，比较最终 latent 和 TP ranks 一致性 | 否，包含比较与复制 |
| measure | 低开销分阶段 CUDA Events、rank 计时/显存、AllReduce 次数与输入字节统计 | 是 |
| profile | 独立 CPU/CUDA trace、模块标记、shape、memory events、算子表 | 否，profiler 有额外开销 |

每个模式都遵循一次预热和两次正式生成；compare 还会额外重放两次固定输入 DiT。正确性与性能是独立运行，不把 capture/compare 的耗时混进正式加速比。profile 为按需诊断配置，不自动追加到 measure。

## 采集内容与口径

- Pipeline：原生 CUDA 区间以及同步后的墙钟时间，保留每个 rank，每轮取最慢 rank。墙钟包含原生 prepare_run/起始同步，CUDA 区间沿用原生边界；加载、文件保存和 MP4 编码不计入生成。
- denoising：第一个 DiT 开始到最后一次 scheduler 更新结束，不包括循环前的 timesteps/latents 准备。
- sampling_step：该步第一个 DiT 调用开始到 scheduler 更新结束，包含普通 CFG 的多次 DiT 调用。
- 各阶段：text_encode、DiT forward、scheduler_step、vae_decode 的 CUDA 区间。区间可能包含 CPU 提交间隙、搬运或等待，不能当成互斥的 kernel 时间相加还原整条管线。
- DiT forward：保存每次原始值与合计，分别报告单次 forward 分布和整轮合计；与完整 denoising 区间分开。
- 显存：加载峰值、Transformer/TextEncoder 参数 bytes 按 dtype 分组、正式运行前后 allocated、每轮 peak allocated/reserved，各 rank 单独记录；allocated 不包含所有 NCCL/驱动占用，各卡峰值之和不是同时刻整机峰值。
- CPU 内存：记录加载/预热后及每轮正式生成前后的进程 RSS 快照；快照不代表 CPU 内存峰值。
- 通信：观测 TP AllReduce 的实际调用次数、输入张量 bytes 之和、dtype；不称为网络线缆传输 bytes 或 NCCL bus bandwidth。measure 不逐次同步或计时通信；耗时、等待、重叠从 profile trace 判断。
- 诊断：FFN/SelfAttention/CrossAttention、Linear、Q/K norm/Row TP 包装的 trace 标记，以及 NCCL、CPU 调度、H2D/D2H、分配事件。保存按 GPU/CPU self time 排序的算子表；不能据此虚构 SM/Tensor Core 利用率、FLOPS、能耗或链路带宽。
- 环境：实际源码 SHA、工作区状态、模型记录、参数与激活 dtype、输入/config、软件版本、GPU 名称/UUID/PCI 地址和拓扑、driver 路径。源码选择先于 import，避免 editable 安装遮蔽 baseline。
- p95：使用 NumPy linear 分位方法，所有分布保留 n 和 samples。生成延迟 n=2，逐采样步/DiT 分布使用实际采集样本数；这些是相关样本，不能宣称稳定尾延迟或独立请求服务 p95。

AllReduce 统计、CUDA Events 等仍有测量开销；profiler 与正确性复制被单独隔离。trace 的 NCCL kernel 累计时间不等于通信给总延迟增加的时间。

## 云端命令（待执行）

在有 CUDA GPU 时执行。建议同一双卡实例分别用单卡和双卡，先完成正确性再采集正式速度。本机不运行以下命令。

```bash
cd /root/autodl-tmp/wan_xdit
git -C xdit-wan-tp pull --ff-only

bench_driver=/root/autodl-tmp/wan_xdit/xdit-wan-tp/benchmarks/wan_tp_benchmark.py
common=(--bench-model-path /root/autodl-tmp/wan_xdit/models/Wan2.1-T2V-1.3B-Diffusers
        --attention_backend FLASH --cross_attention_backend FLASH
        --seed 42 --guidance_scale 5 --flow_shift 3
        --prompt 'A red fox walks through a snowy forest, cinematic camera movement.'
        --negative_prompt 'blurry, low quality, static, subtitles')
```

先保存单卡参考，再执行 TP2 比较。未提供 atol/rtol 时只输出误差与有限值、rank 一致性检查，不能宣称数值对齐通过；明确阈值后可添加 --bench-atol 和 --bench-rtol。阈值不能根据失败结果自动放宽。

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$bench_driver" --bench-source /root/autodl-tmp/wan_xdit/xdit-baseline \
  --bench-mode capture --tensor_parallel_degree 1 \
  --output_directory results/correctness-single "${common[@]}"

CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$bench_driver" --bench-source /root/autodl-tmp/wan_xdit/xdit-wan-tp \
  --bench-mode compare --tensor_parallel_degree 2 \
  --bench-reference results/correctness-single/reference.pt \
  --output_directory results/correctness-tp2 "${common[@]}"
```

正式性能采集：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$bench_driver" --bench-source /root/autodl-tmp/wan_xdit/xdit-baseline \
  --bench-mode measure --tensor_parallel_degree 1 \
  --output_directory results/resident-single "${common[@]}"

CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$bench_driver" --bench-source /root/autodl-tmp/wan_xdit/xdit-wan-tp \
  --bench-mode measure --tensor_parallel_degree 2 \
  --output_directory results/resident-tp2 "${common[@]}"

.venv/bin/python xdit-wan-tp/benchmarks/wan_tp_report.py \
  --single results/resident-single --tp2 results/resident-tp2 \
  --correctness results/correctness-tp2 --output results/resident-report.json
```

CPU offload 配置：在对应单卡/TP2 命令都追加 --enable_model_cpu_offload，使用新的结果目录和匹配的 correctness reference。不要覆盖驻留模式结果。OOM 写入 failure_rank_N.json（其他异常原样暴露），没有完整结果的配置不计算加速比。

需要诊断时，将同一配置的 --bench-mode 改为 profile，并指定新的 output_directory。每个 rank 会保存 trace_rank_N.json.gz、operators_rank_N.txt 和标注 performance_eligible=false 的 rank_N.json。Nsight 等硬件计数器不自动运行。

## 如何判断后续优化方向

| 观察 | 下一步依据 |
|---|---|
| DiT 加速，Pipeline 加速有限 | 检查 text_encode、VAE、加载外固定开销及 denoising 占比 |
| GEMM 变快，DiT 变慢 | 检查 TP AllReduce 密度、NCCL 等待、norm 的小归约与小矩阵效率 |
| offload 模式明显退化 | 查看 H2D/D2H、参数迁移、CPU 提交与等待区间 |
| rank 0/1 耗时明显不同 | 对照各 rank CPU 活动、CUDA/NCCL 时间线、拓扑与显存 |
| 显存低于单卡但总资源量增加 | 同时评估每卡容量收益、速度与 GPU-seconds/video |

先按时间线形成瓶颈假设，再做定向优化与同协议复测；不把两个样本的统计量或 profiler 中单个累计值当成因果证明。
