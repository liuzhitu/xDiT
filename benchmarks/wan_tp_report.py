"""Summarize paired single-GPU and TP2 Wan runs; no model loading."""
import argparse
import json
from pathlib import Path

import numpy as np


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95, method="linear")),
        "min": float(values.min()), "max": float(values.max()), "samples": values.tolist(),
        "percentile_method": "linear",
    }


def read_run(directory):
    paths = sorted(directory.glob("rank_*.json"))
    if not paths:
        raise FileNotFoundError(f"No rank results in {directory}")
    ranks = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected = ranks[0]["manifest"]["world_size"]
    if sorted(row["manifest"]["rank"] for row in ranks) != list(range(expected)):
        raise ValueError(f"Incomplete rank files in {directory}")
    if any(not row["performance_eligible"] or len(row["iterations"]) != 2 for row in ranks):
        raise ValueError("Performance report requires two measure-mode iterations on every rank")
    pooled = []
    for index in range(2):
        samples = [rank["iterations"][index] for rank in ranks]
        maximum = lambda key: max(sample[key] for sample in samples)
        pooled.append({
            "pipeline_wall_s": maximum("pipeline_wall_s"),
            "pipeline_native_cuda_s": maximum("pipeline_native_cuda_s"),
            "dit_cuda_sum_s": maximum("dit_cuda_sum_s"),
            "denoising_cuda_s": maximum("denoising_cuda_s"),
            "peak_allocated_bytes": max(sample["memory_bytes"]["peak_allocated"] for sample in samples),
            "peak_reserved_bytes": max(sample["memory_bytes"]["peak_reserved"] for sample in samples),
        })
    summary = {key: distribution([sample[key] for sample in pooled]) for key in pooled[0]}
    summary["videos_per_s"] = distribution([1 / row["pipeline_wall_s"] for row in pooled])
    summary["frames_per_s"] = distribution([81 / row["pipeline_wall_s"] for row in pooled])
    summary["gpu_seconds_per_video"] = distribution([expected * row["pipeline_wall_s"] for row in pooled])
    phase_distributions = {}
    for rank in ranks:
        stages = sorted({name for row in rank["iterations"] for name in row["stages_ms"]})
        phase_distributions[str(rank["manifest"]["rank"])] = {
            name: distribution([value for row in rank["iterations"]
                                for value in row["stages_ms"][name]["samples"]])
            for name in stages
        }
    return {"directory": str(directory), "summary": summary,
            "per_rank_phase_distributions_ms": phase_distributions,
            "ranks": ranks}


def validate_pair(single, parallel):
    if single["ranks"][0]["manifest"]["world_size"] != 1 \
            or parallel["ranks"][0]["manifest"]["world_size"] != 2:
        raise ValueError("Expected single GPU and TP2")
    a, b = single["ranks"][0]["manifest"], parallel["ranks"][0]["manifest"]
    for key in ("height", "width", "num_frames", "num_inference_steps", "seed",
                "prompt", "negative_prompt", "guidance_scale", "flow_shift"):
        if a["input_args"].get(key) != b["input_args"].get(key):
            raise ValueError(f"Cannot compare different input settings: {key}")
    for key in ("attention_backend", "cross_attention_backend", "enable_model_cpu_offload",
                "enable_sequential_cpu_offload", "enable_group_cpu_offload"):
        if a["config"].get(key) != b["config"].get(key):
            raise ValueError(f"Cannot compare different runtime settings: {key}")
    if a["observed_forward"]["input_dtype"] != b["observed_forward"]["input_dtype"] \
            or set(a["transformer_parameter_bytes_by_dtype"]) != set(b["transformer_parameter_bytes_by_dtype"]):
        raise ValueError("Model or activation dtypes differ")
    if a["scheduler"] != b["scheduler"]:
        raise ValueError("Scheduler class or configuration differs")
    if a["model_revision"] != b["model_revision"] or a["versions"] != b["versions"]:
        raise ValueError("Model revision or software versions differ")
    if any(row["manifest"]["gpu"]["name"] != a["gpu"]["name"] for row in parallel["ranks"]):
        raise ValueError("GPU models differ")


def correctness_summary(directory):
    rows = [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("rank_*.json"))]
    if not rows or len(rows) != 2 or any(row["mode"] != "compare" for row in rows):
        raise ValueError("Correctness directory must contain both TP2 compare-mode ranks")
    checks = []
    for row in rows:
        value = row["correctness"]
        checks.extend(value["final_latents"])
        for fixed in value["fixed_input_dit"]:
            checks.extend((fixed["reference"], fixed["tp_rank_agreement"]))
    if any(not check["finite"] or check["allclose"] is False for check in checks):
        status = "failed"
    elif any(check["allclose"] is None for check in checks):
        status = "metrics_only"
    else:
        status = "passed_explicit_thresholds"
    return {"status": status, "rank_results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--tp2", type=Path, required=True)
    parser.add_argument("--correctness", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    single, parallel = read_run(args.single), read_run(args.tp2)
    validate_pair(single, parallel)
    speedups = {}
    for name in ("pipeline_wall_s", "pipeline_native_cuda_s", "dit_cuda_sum_s", "denoising_cuda_s"):
        speedup = single["summary"][name]["median"] / parallel["summary"][name]["median"]
        speedups[name] = {"speedup": speedup, "parallel_efficiency": speedup / 2}
    memory_ratio = 1 - parallel["summary"]["peak_allocated_bytes"]["median"] \
        / single["summary"]["peak_allocated_bytes"]["median"]
    report = {
        "protocol": {"warmup": "1 x 17 frames", "measure": "2 x 81 frames",
                     "resolution": "832x480", "steps": 50,
                     "p95": "Linear sample quantile. Generation n=2; do not infer stable tail latency.",
                     "rank_aggregation": "Maximum rank duration per iteration; per-rank samples retained.",
                     "boundaries": "Loading and MP4 encoding excluded from generation speedup.",
                     "denoising": "First DiT start through final scheduler update; setup excluded.",
                     "communication": "Input bytes are not wire bytes; NCCL durations require separate traces."},
        "single": single, "tp2": parallel, "speedups": speedups,
        "max_per_rank_allocated_reduction": memory_ratio,
    }
    if args.correctness:
        report["correctness"] = correctness_summary(args.correctness)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str(args.output), "speedups": speedups}, indent=2))
    if report.get("correctness", {}).get("status") == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
