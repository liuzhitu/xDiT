"""Wan2.1 T2V single-GPU / TP2 benchmark using the native xDiT runner.

Only run on cloud CUDA GPUs. This file does not implement model inference.
"""
import argparse
import copy
import functools
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter
from contextlib import nullcontext

# Select the baseline or TP checkout before importing xfuser.
_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument("--bench-source", "--bench_source", type=Path, required=True)
_source_args, _ = _bootstrap.parse_known_args()
SOURCE = _source_args.bench_source.resolve()
if not (SOURCE / "xfuser" / "__init__.py").is_file():
    raise FileNotFoundError(f"Not an xDiT checkout: {SOURCE}")
sys.path.insert(0, str(SOURCE))

import numpy as np
import psutil
import torch
import torch.distributed as dist
import xfuser
from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed.parallel_state import get_tp_group
from xfuser.runner import xFuserModelRunner

if Path(xfuser.__file__).resolve().parent.parent != SOURCE:
    raise RuntimeError("xfuser was imported from a different checkout")


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95, method="linear")),
        "min": float(values.min()),
        "max": float(values.max()),
        "samples": values.tolist(),
        "percentile_method": "linear",
    }


def command_output(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def tree_map(value, transform):
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, dict):
        return {key: tree_map(item, transform) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(tree_map(item, transform) for item in value)
    if isinstance(value, list):
        return [tree_map(item, transform) for item in value]
    return value


def errors(reference, actual, atol, rtol):
    reference, actual = reference.float().cpu(), actual.float().cpu()
    if reference.shape != actual.shape:
        raise ValueError(f"Output shape mismatch: {reference.shape} vs {actual.shape}")
    delta = actual - reference
    ref_norm = torch.linalg.vector_norm(reference)
    actual_norm = torch.linalg.vector_norm(actual)
    relative = float(torch.linalg.vector_norm(delta) / ref_norm) if ref_norm > 0 else None
    cosine = float((reference.flatten() @ actual.flatten()) / (ref_norm * actual_norm)) \
        if ref_norm > 0 and actual_norm > 0 else None
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    accepted = bool(torch.allclose(actual, reference, atol=atol, rtol=rtol)) \
        if atol is not None and rtol is not None else None
    return {
        "shape": list(actual.shape), "finite": finite,
        "max_abs": float(delta.abs().max()) if finite else None,
        "mae": float(delta.abs().mean()) if finite else None,
        "rmse": float(delta.square().mean().sqrt()) if finite else None,
        "relative_l2": relative if finite else None, "cosine": cosine if finite else None,
        "allclose": accepted, "atol": atol, "rtol": rtol,
    }


def parameter_inventory(module):
    sizes = Counter()
    for parameter in module.parameters():
        sizes[str(parameter.dtype)] += parameter.numel() * parameter.element_size()
    return dict(sizes)


class Collector:
    def __init__(self, runner, args):
        self.runner, self.args = runner, args
        self.phase = "idle"
        self.iteration = -1
        self.rows = []
        self.warmup = None
        self.events = []
        self.communication = Counter()
        self.communication_dtypes = Counter()
        self.fixed = []
        self.final_latents = []
        self.stage_stack = []
        self.restores = []
        self.profile = args.bench_mode == "profile"
        self.correctness = args.bench_mode in ("capture", "compare")
        self.forward_info = None
        self.install()

    def wrap(self, owner, name, replacement):
        original = getattr(owner, name)
        setattr(owner, name, replacement(original))
        self.restores.append((owner, name, original))

    def restore(self):
        for owner, name, original in reversed(self.restores):
            setattr(owner, name, original)

    def region(self, name):
        return torch.profiler.record_function(name) if self.profile else nullcontext()

    def timed(self, name, function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            if self.phase != "formal":
                return function(*args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            self.stage_stack.append(name)
            try:
                with self.region(name):
                    output = function(*args, **kwargs)
            finally:
                self.stage_stack.pop()
            end.record()
            self.events.append((name, start, end))
            return output
        return call

    def install(self):
        pipe, model = self.runner.model.pipe, self.runner.model
        self.wrap(pipe, "encode_prompt", lambda original: self.timed("text_encode", original))
        self.wrap(pipe.vae, "decode", lambda original: self.timed("vae_decode", original))

        def transformer_wrapper(original):
            measured = self.timed("dit_forward", original)
            @functools.wraps(original)
            def call(*inputs, **kwargs):
                capture = self.correctness and self.phase == "formal" \
                    and self.iteration == 0 and len(self.fixed) < 2
                saved = tree_map((inputs, kwargs), lambda x: x.detach().cpu().clone()) if capture else None
                if self.phase == "formal" and self.step_start is None:
                    self.step_start = torch.cuda.Event(enable_timing=True)
                    self.step_start.record()
                output = measured(*inputs, **kwargs)
                if self.phase == "formal" and self.forward_info is None:
                    hidden = kwargs.get("hidden_states", inputs[0] if inputs else None)
                    self.forward_info = {"input_dtype": str(hidden.dtype),
                                         "input_shape": list(hidden.shape),
                                         "autocast_note": "Observe runtime dtype; no benchmark autocast override."}
                if capture:
                    self.fixed.append({"inputs": saved, "output": output[0].detach().cpu().clone()})
                return output
            return call
        self.wrap(pipe.transformer, "forward", transformer_wrapper)

        original_step = pipe.scheduler.step
        self.scheduler_calls = 0
        def scheduler_step(*inputs, **kwargs):
            if self.phase == "formal":
                self.scheduler_calls += 1
                # From the first DiT call through the final scheduler update.
                first_dit = next((start for name, start, _ in self.events if name == "dit_forward"), None)
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                with self.region("scheduler_step"):
                    output = original_step(*inputs, **kwargs)
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self.events.append(("scheduler_step", start, end))
                self.events.append(("sampling_step", self.step_start, end))
                self.step_start = None
                if self.scheduler_calls == self.runner.config.num_inference_steps:
                    self.denoising = (first_dit, end)
                    if self.correctness:
                        # Correctness mode is excluded from performance comparison.
                        self.final_latents.append(output[0].detach().cpu().clone())
                return output
            return original_step(*inputs, **kwargs)
        self.wrap(pipe.scheduler, "step", lambda original: scheduler_step)

        group = get_tp_group()
        def communication_wrapper(original):
            def call(tensor, *inputs, **kwargs):
                if self.phase != "formal":
                    return original(tensor, *inputs, **kwargs)
                stage = self.stage_stack[-1] if self.stage_stack else "other"
                self.communication["calls"] += 1
                self.communication["input_bytes"] += tensor.numel() * tensor.element_size()
                self.communication_dtypes[str(tensor.dtype)] += 1
                with self.region(f"tp_all_reduce/{stage}"):
                    return original(tensor, *inputs, **kwargs)
            return call
        if group.world_size > 1:
            self.wrap(group, "all_reduce", communication_wrapper)

        if self.profile:
            # Annotate production modules only in the diagnostic trace.
            for index, block in enumerate(pipe.transformer.blocks):
                for name in ("ffn", "attn1", "attn2"):
                    module = getattr(block, name)
                    label = f"block_{index}/{name}"
                    def annotate(original, label=label):
                        @functools.wraps(original)
                        def call(*inputs, **kwargs):
                            with self.region(label):
                                return original(*inputs, **kwargs)
                        return call
                    self.wrap(module, "forward", annotate)
                    for child_name, child in module.named_modules():
                        if child_name and (isinstance(child, (torch.nn.Linear, torch.nn.RMSNorm))
                                or type(child).__name__ in ("xFuserWanTPRMSNorm", "xFuserWanTPRowLinear")):
                            self.wrap(child, "forward", lambda original, label=f"{label}/{child_name}": annotate(original, label))

        def warmup_wrapper(original):
            def call(input_args):
                self.phase = "warmup"
                warmup_args = copy.deepcopy(input_args)
                warmup_args["num_frames"] = 17
                try:
                    return original(warmup_args)
                finally:
                    self.phase = "idle"
            return call
        self.wrap(model, "_run_warmup_calls", warmup_wrapper)

        def pipeline_wrapper(original):
            def call(input_args):
                if self.phase == "warmup":
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    with self.region("warmup_17_frames"):
                        output, native_seconds = original(input_args)
                    self.warmup = {"frames": 17, "pipeline_wall_s": time.perf_counter() - start,
                                   "pipeline_native_cuda_s": native_seconds,
                                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                   "cpu_rss_after": psutil.Process().memory_info().rss}
                    return output, native_seconds
                self.phase = "formal"
                self.iteration += 1
                self.events = []
                self.communication.clear()
                self.communication_dtypes.clear()
                self.scheduler_calls = 0
                self.step_start = None
                self.denoising = None
                torch.cuda.synchronize()
                dist.barrier()
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                cpu_rss_before = psutil.Process().memory_info().rss
                wall_start = time.perf_counter()
                with self.region(f"pipeline_generation/{self.iteration}"):
                    output, native_seconds = original(input_args)
                torch.cuda.synchronize()
                wall_seconds = time.perf_counter() - wall_start
                stage_values = {}
                for name, start, end in self.events:
                    stage_values.setdefault(name, []).append(start.elapsed_time(end))
                denoising_ms = self.denoising[0].elapsed_time(self.denoising[1]) \
                    if self.denoising is not None and self.denoising[0] is not None else None
                row = {
                    "iteration": self.iteration,
                    "pipeline_wall_s": wall_seconds,
                    "pipeline_native_cuda_s": native_seconds,
                    "denoising_cuda_s": denoising_ms / 1000 if denoising_ms is not None else None,
                    "stages_ms": {name: distribution(values) for name, values in stage_values.items()},
                    "dit_cuda_sum_s": sum(stage_values.get("dit_forward", [])) / 1000,
                    "memory_bytes": {"cpu_rss_before": cpu_rss_before,
                                     "cpu_rss_after": psutil.Process().memory_info().rss,
                                     "allocated_before": before,
                                     "allocated_after": torch.cuda.memory_allocated(),
                                     "peak_allocated": torch.cuda.max_memory_allocated(),
                                     "peak_reserved": torch.cuda.max_memory_reserved()},
                    "tp_all_reduce": {"calls": self.communication["calls"],
                                      "input_bytes": self.communication["input_bytes"],
                                      "calls_by_dtype": dict(self.communication_dtypes),
                                      "bytes_definition": "sum of input tensor bytes, not network wire bytes"},
                }
                self.rows.append(row)
                # Release per-step event objects before the next generation.
                self.events = []
                self.phase = "idle"
                return output, native_seconds
            return call
        self.wrap(model, "_run_timed_pipe", pipeline_wrapper)


def parse_args():
    parser = FlexibleArgumentParser(description=__doc__)
    xFuserArgs.add_runner_args(parser)
    parser.add_argument("--bench_source", type=Path, required=True)
    parser.add_argument("--bench_model_path", type=Path, required=True)
    parser.add_argument("--bench_mode", choices=("measure", "capture", "compare", "profile"), default="measure")
    parser.add_argument("--bench_reference", type=Path)
    parser.add_argument("--bench_atol", type=float)
    parser.add_argument("--bench_rtol", type=float)
    parser.add_argument("--bench_model_revision", default="0fad780a534b6463e45facd96134c9f345acfa5b")
    parser.set_defaults(model="Wan2.1-T2V-1.3B", height=480, width=832,
                        num_frames=81, num_inference_steps=50, seed=42,
                        guidance_scale=5.0, flow_shift=3.0,
                        warmup_calls=1, num_iterations=2, warmup_steps=0,
                        output_directory=None,
                        attention_backend="FLASH", cross_attention_backend="FLASH")
    args = parser.parse_args()
    if not args.output_directory:
        parser.error("Specify a fresh --output_directory")
    if not args.prompt or len(args.prompt) != 1:
        parser.error("Pass exactly one --prompt for one video per generation")
    for field, expected in {"height": 480, "width": 832, "num_frames": 81,
                            "num_inference_steps": 50, "warmup_calls": 1,
                            "num_iterations": 2, "warmup_steps": 0}.items():
        if getattr(args, field) != expected:
            parser.error(f"{field} is fixed to {expected} by this benchmark protocol")
    if args.model != "Wan2.1-T2V-1.3B":
        parser.error("This benchmark targets Wan2.1-T2V-1.3B only")
    if args.tensor_parallel_degree not in (1, 2):
        parser.error("Only single GPU and TP2 are in scope")
    if any(getattr(args, field) != 1 for field in
           ("ulysses_degree", "ring_degree", "data_parallel_degree",
            "pipefusion_parallel_degree", "fully_shard_degree")) or args.use_cfg_parallel:
        parser.error("SP/DP/CFG/PipeFusion/FSDP parallelism is outside this benchmark")
    if args.cache_method or args.use_teacache or args.use_fbcache or args.gemm_quantization \
            or args.use_torch_compile or args.use_hybrid_attn_schedule or args.use_hybrid_gemm_schedule \
            or args.use_fp8_gemms or args.use_int8_gemms or args.use_fp4_gemms \
            or args.use_fp6_gemms or args.use_a6w4_gemms or args.use_fp8_comms or args.use_fp8_text_encoder \
            or args.use_parallel_vae or args.batch_size or args.dataset_path \
            or args.determinism_check or args.profile:
        parser.error("Use the fixed eager, unquantized, uncached single-request protocol")
    if args.bench_mode == "capture" and args.tensor_parallel_degree != 1:
        parser.error("Capture the reference with the unmodified single-GPU baseline")
    if args.bench_mode == "compare" and (args.tensor_parallel_degree != 2 or not args.bench_reference):
        parser.error("Compare requires TP2 and --bench-reference")
    if (args.bench_atol is None) != (args.bench_rtol is None):
        parser.error("Supply both --bench-atol and --bench-rtol, or neither")
    return args


def main():
    args = parse_args()
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if world != args.tensor_parallel_degree:
        raise ValueError("Launch one process per TP rank with torchrun")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    config = xFuserArgs.from_runner_args(vars(args))
    runner = xFuserModelRunner(vars(config))
    runner.model.settings.model_name = str(args.bench_model_path.resolve())
    inputs = runner.preprocess_args(vars(config))
    output_dir = Path(args.output_directory).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / f"rank_{rank}.json").exists() or (output_dir / f"failure_rank_{rank}.json").exists():
        raise FileExistsError(f"Use a fresh output directory: {output_dir}")

    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    try:
        runner.initialize(inputs)
    except torch.cuda.OutOfMemoryError as error:
        with (output_dir / f"failure_rank_{rank}.json").open("w", encoding="utf-8") as handle:
            json.dump({"status": "oom", "stage": "initialization", "message": str(error),
                       "source": str(SOURCE), "config": vars(config)}, handle, indent=2, default=str)
        raise
    torch.cuda.synchronize()
    loading = {"wall_s": time.perf_counter() - load_start,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
               "cpu_rss_after": psutil.Process().memory_info().rss}
    manifest = {
        "source": str(SOURCE),
        "source_commit": command_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"]),
        "source_changes": command_output(["git", "-C", str(SOURCE), "status", "--porcelain"]),
        "driver": str(Path(__file__).resolve()),
        "model_path": str(args.bench_model_path.resolve()),
        "model_revision": args.bench_model_revision,
        "config": vars(config), "input_args": inputs,
        "scheduler": {"class": type(runner.model.pipe.scheduler).__name__,
                      "config": dict(runner.model.pipe.scheduler.config)},
        "rank": rank, "world_size": world,
        "gpu": {"name": torch.cuda.get_device_name(),
                "properties": str(torch.cuda.get_device_properties(torch.cuda.current_device()))},
        "topology": command_output(["nvidia-smi", "topo", "-m"]),
        "gpu_snapshot": command_output(["nvidia-smi", "--query-gpu=name,uuid,pci.bus_id,memory.total,driver_version",
                                       "--format=csv"]),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "diffusers", "transformers", "accelerate", "yunchang",
                      "distvae", "flash-attn", "huggingface-hub")},
        "transformer_parameter_bytes_by_dtype": parameter_inventory(runner.model.pipe.transformer),
        "text_encoder_parameter_bytes_by_dtype": parameter_inventory(runner.model.pipe.text_encoder),
        "protocol": {"warmup_frames": 17, "formal_frames": 81, "steps": 50,
                     "width": 832, "height": 480, "formal_repeats": 2,
                     "p95_note": "Generation p95 uses two samples; not a stable tail estimate."},
    }
    collector = Collector(runner, args)
    # Standard benchmark always uses native runner.run, including its warmup.
    profile_context = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=True, with_stack=False,
    ) if args.bench_mode == "profile" else nullcontext()
    failed_oom = False
    try:
        with profile_context as profiler:
            runner.run(inputs)
        if args.bench_mode == "profile":
            profiler.export_chrome_trace(str(output_dir / f"trace_rank_{rank}.json.gz"))
            with (output_dir / f"operators_rank_{rank}.txt").open("w", encoding="utf-8") as handle:
                averages = profiler.key_averages(group_by_input_shape=True)
                handle.write(averages.table(sort_by="self_device_time_total", row_limit=80))
                handle.write("\n\nCPU:\n")
                handle.write(averages.table(sort_by="self_cpu_time_total", row_limit=80))
        manifest["observed_forward"] = collector.forward_info
        result = {"manifest": manifest, "loading": loading, "warmup": collector.warmup,
                  "mode": args.bench_mode,
                  "performance_eligible": args.bench_mode == "measure",
                  "iterations": collector.rows}
        if collector.correctness:
            payload = {"fixed": collector.fixed, "final_latents": collector.final_latents,
                       "config": {"seed": inputs["seed"], "guidance_scale": inputs["guidance_scale"],
                                  "flow_shift": inputs["flow_shift"], "prompt": inputs["prompt"],
                                  "negative_prompt": inputs["negative_prompt"],
                                  "model_cpu_offload": args.enable_model_cpu_offload,
                                  "sequential_cpu_offload": args.enable_sequential_cpu_offload,
                                  "group_cpu_offload": args.enable_group_cpu_offload,
                                  "backend": args.attention_backend,
                                  "cross_backend": args.cross_attention_backend,
                                  "model_revision": args.bench_model_revision}}
            if args.bench_mode == "capture" and rank == 0:
                torch.save(payload, output_dir / "reference.pt")
            if args.bench_mode == "compare":
                reference = torch.load(args.bench_reference, map_location="cpu", weights_only=True)
                if payload["config"] != reference["config"]:
                    raise ValueError("Reference and TP2 experiment settings differ")
                device = runner.model.pipe._execution_device
                checks = []
                collector.phase = "replay"
                with torch.no_grad():
                    for sample in reference["fixed"]:
                        fixed_args, fixed_kwargs = tree_map(sample["inputs"], lambda x: x.to(device))
                        actual = runner.model.pipe.transformer(*fixed_args, **fixed_kwargs)[0]
                        peer = actual.detach().clone(memory_format=torch.contiguous_format)
                        dist.broadcast(peer, src=0)
                        checks.append({"reference": errors(sample["output"], actual, args.bench_atol, args.bench_rtol),
                                       "tp_rank_agreement": errors(peer, actual, 0.0, 0.0)})
                if len(collector.final_latents) != 2 or len(reference["final_latents"]) != 2:
                    raise ValueError("Expected two complete 50-step sampling runs")
                result["correctness"] = {
                    "fixed_input_dit": checks,
                    "final_latents": [errors(expected, actual, args.bench_atol, args.bench_rtol)
                                     for expected, actual in zip(reference["final_latents"], collector.final_latents)],
                    "acceptance": "metrics_only" if args.bench_atol is None else "explicit_allclose_thresholds",
                }
        with (output_dir / f"rank_{rank}.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=str, allow_nan=False)
    except torch.cuda.OutOfMemoryError as error:
        failed_oom = True
        with (output_dir / f"failure_rank_{rank}.json").open("w", encoding="utf-8") as handle:
            json.dump({"status": "oom", "message": str(error), "manifest": manifest,
                       "completed_iterations": collector.rows}, handle, indent=2, default=str)
        raise
    finally:
        collector.restore()
        if not failed_oom:
            runner.cleanup()


if __name__ == "__main__":
    main()
