import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentCase:
    tag: str
    kind: str
    strategy: str
    cache_ratio: float
    warmup: int
    cooldown: int


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _split_words(raw: str) -> list[str]:
    return [item for item in raw.replace(",", " ").split() if item]


def _sanitize_tag_part(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "case"


def _run_tag_slug(tag: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", tag).strip("_").lower()
    return value[:32] or "case"


def _resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {raw_path}")
    return path


def _runtime_python() -> str:
    env_python = os.getenv("CHITU_PYTHON_BIN", "").strip()
    if env_python:
        return env_python

    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)

    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("python/python3 is required")


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return data


def _cfg_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    return str(value)


def _hydra_dump_mode(value: Any) -> str:
    if value is False:
        return "off"
    if value is True:
        return "video_dir"
    return str(value)


def _hydra_overrides(cfg: dict[str, Any], run_root: Path, fpp_debug: bool) -> list[str]:
    overrides = [
        f"models={_cfg_get(cfg, 'model.name', 'Wan2.1-T2V-1.3B')}",
        f"models.ckpt_dir={_cfg_get(cfg, 'model.ckpt_dir', '')}",
        f"infer.diffusion.cfg_size={_cfg_get(cfg, 'parallel.cfp', 1)}",
        f"infer.diffusion.cp_size={_cfg_get(cfg, 'parallel.cp_size', 1)}",
        f"infer.diffusion.fpp_size={_cfg_get(cfg, 'parallel.fpp_size', 1)}",
        f"infer.diffusion.patch_num={_cfg_get(cfg, 'parallel.patch_num', 7)}",
        f"infer.diffusion.up_limit={_cfg_get(cfg, 'infer.up_limit', 8)}",
        f"infer.attn_type={_cfg_get(cfg, 'infer.attn_type', 'flash_attn')}",
        f"infer.diffusion.low_mem_level={_cfg_get(cfg, 'infer.low_mem_level', 0)}",
        "infer.diffusion.enable_flexcache=true",
        f"infer.diffusion.fpp_debug={_hydra_value(fpp_debug)}",
        "eval.eval_type=[]",
        "eval.reference_path=null",
        f"output.root_dir={run_root}",
        f"output.enable_run_log={_hydra_value(_cfg_get(cfg, 'output.enable_run_log', True))}",
        f"output.enable_timer_dump={_hydra_value(_cfg_get(cfg, 'output.enable_timer_dump', True))}",
        f"output.hydra_dump_mode={_hydra_dump_mode(_cfg_get(cfg, 'output.hydra_dump_mode', 'video_dir'))}",
        f"output.enable_kv_capture={_hydra_value(_cfg_get(cfg, 'output.enable_kv_capture', False))}",
    ]

    extra_overrides = _cfg_get(cfg, "overrides", [])
    if extra_overrides:
        if not isinstance(extra_overrides, list):
            raise ValueError("overrides must be a YAML list")
        overrides.extend(str(item) for item in extra_overrides)
    return overrides


def _latest_run_dir_for_tag(run_root: Path, tag: str) -> Path:
    tag_slug = _run_tag_slug(tag)
    candidates = [path for path in run_root.glob(f"{tag_slug}_*") if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"failed to locate output dir for tag: {tag} (slug: {tag_slug})")
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def _build_cases(cache_ratio: float, warmup: int, cooldown: int) -> list[ExperimentCase]:
    cases: list[ExperimentCase] = []

    raw_strategies = os.getenv("CHITU_EXP_CACHE_STRATEGIES")
    if raw_strategies is None:
        raw_strategies = "fpp_cache teacache pab"
    strategies = _split_words(raw_strategies)
    for strategy in strategies:
        tag_strategy = _sanitize_tag_part(strategy)
        cases.append(
            ExperimentCase(
                tag=f"cache_{tag_strategy}_r{cache_ratio}_w{warmup}_c{cooldown}",
                kind="cache_strategy",
                strategy=strategy,
                cache_ratio=cache_ratio,
                warmup=warmup,
                cooldown=cooldown,
            )
        )

    raw_schedules = os.getenv("CHITU_EXP_FPP_SCHEDULES")
    if raw_schedules is None:
        raw_schedules = "0:0 3:3 5:5 8:8 10:5 5:10"
    schedules = _split_words(raw_schedules)
    for schedule in schedules:
        if ":" not in schedule:
            raise ValueError(f"FPP schedule must be warmup:cooldown, got: {schedule}")
        schedule_warmup, schedule_cooldown = schedule.split(":", 1)
        cases.append(
            ExperimentCase(
                tag=f"fpp_r{cache_ratio}_w{int(schedule_warmup)}_c{int(schedule_cooldown)}",
                kind="fpp_schedule",
                strategy="fpp_cache",
                cache_ratio=cache_ratio,
                warmup=int(schedule_warmup),
                cooldown=int(schedule_cooldown),
            )
        )

    return cases


def _run_generation(
    case: ExperimentCase,
    cfg: dict[str, Any],
    run_root: Path,
    python_bin: str,
    dry_run: bool = False,
) -> Path:
    env = os.environ.copy()
    env.update(
        {
            "CHITU_RUN_TAG": case.tag,
            "CHITU_PYTHON_BIN": python_bin,
            "HYDRA_FULL_ERROR": "1",
            "CHITU_DEBUG": "1" if _cfg_get(cfg, "runtime.chitu_debug", True) else "0",
            "SRUN_PARTITION": str(_cfg_get(cfg, "launch.srun.partition", "debug")),
            "SRUN_CPUS_PER_GPU": str(_cfg_get(cfg, "launch.srun.cpus_per_gpu", 24)),
            "SRUN_MEM_PER_GPU": str(_cfg_get(cfg, "launch.srun.mem_per_gpu", 242144)),
            "SRUN_JOB_NAME": str(_cfg_get(cfg, "launch.srun.job_name", "chitu")),
            "CHITU_EXP_CACHE_STRATEGY": case.strategy,
            "CHITU_EXP_CACHE_RATIO": str(case.cache_ratio),
            "CHITU_EXP_WARMUP": str(case.warmup),
            "CHITU_EXP_COOLDOWN": str(case.cooldown),
        }
    )
    if _cfg_get(cfg, "runtime.cuda_launch_blocking", False):
        env["CUDA_LAUNCH_BLOCKING"] = "1"

    fpp_debug = _env("CHITU_EXP_FPP_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    if case.strategy in {"none", "off", "disable", "disabled", "baseline"}:
        fpp_debug = False

    print(f"========== Generate: {case.tag} ==========", flush=True)
    print(
        " ".join(
            [
                f"strategy={case.strategy}",
                f"cache_ratio={case.cache_ratio}",
                f"warmup={case.warmup}",
                f"cooldown={case.cooldown}",
            ]
        ),
        flush=True,
    )
    num_nodes = str(_cfg_get(cfg, "launch.num_nodes", 1))
    gpus_per_node = str(_cfg_get(cfg, "launch.gpus_per_node", 1))
    cmd = [
        "bash",
        str(PROJECT_ROOT / "script" / "srun_direct.sh"),
        num_nodes,
        gpus_per_node,
        "script/flexcache_generate.py",
        *_hydra_overrides(cfg, run_root, fpp_debug=fpp_debug),
    ]
    print("Executing:", " ".join(cmd), flush=True)
    if dry_run:
        return run_root / f"{case.tag}_DRY_RUN"
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    return _latest_run_dir_for_tag(run_root, case.tag)


def _run_eval(
    case: ExperimentCase,
    generated_dir: Path,
    reference_dir: Path,
    metrics: str,
    max_frames: int,
    python_bin: str,
) -> tuple[Path, dict[str, Any]]:
    eval_dir = generated_dir / "eval_vs_baseline"

    print(f"========== Eval: {case.tag} ==========", flush=True)
    subprocess.run(
        [
            python_bin,
            "-m",
            "chitu_diffusion.eval.reference_runner",
            "--generated",
            str(generated_dir),
            "--reference",
            str(reference_dir),
            "--metrics",
            metrics,
            "--output-dir",
            str(eval_dir),
            "--max-frames",
            str(max_frames),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    summary_path = eval_dir / "reference_eval_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        return eval_dir, json.load(f)


def _append_line(path: Path, values: list[Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write("\t".join(str(value) for value in values) + "\n")


def _write_eval_rows(metrics_tsv: Path, case: ExperimentCase, eval_summary: dict[str, Any]) -> None:
    for metric, item in eval_summary.items():
        item = item if isinstance(item, dict) else {}
        result = item.get("result")
        result = result if isinstance(result, dict) else {}
        score = result.get("mean_score", result.get("score", ""))
        _append_line(
            metrics_tsv,
            [
                case.tag,
                case.kind,
                case.strategy,
                case.cache_ratio,
                case.warmup,
                case.cooldown,
                metric,
                item.get("status", result.get("status", "")),
                score,
                result.get("result_path", ""),
            ],
        )


def run_experiments(args: argparse.Namespace) -> None:
    config_path = _resolve_config_path(args.config)
    cfg = _load_yaml(config_path)
    python_bin = _runtime_python()

    exp_name = _env("CHITU_EXP_NAME", f"flexcache_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    summary_dir = Path(_env("CHITU_EXP_SUMMARY_DIR", str(PROJECT_ROOT / "outputs" / exp_name))).resolve()
    run_root = Path(_env("CHITU_EXP_OUTPUT_ROOT", str(summary_dir / "runs"))).resolve()

    metrics = _env("CHITU_EXP_METRICS", "psnr,lpips")
    max_frames = int(_env("CHITU_EXP_MAX_FRAMES", "81"))
    cache_ratio = float(_env("CHITU_EXP_CACHE_RATIO", "0.4"))
    warmup = int(_env("CHITU_EXP_WARMUP", "5"))
    cooldown = int(_env("CHITU_EXP_COOLDOWN", "5"))
    reference_path = os.getenv("CHITU_EXP_REFERENCE_PATH", "").strip()
    skip_baseline = _env("CHITU_EXP_SKIP_BASELINE", "0").lower() in {"1", "true", "yes", "on"}

    summary_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    summary_tsv = summary_dir / "summary.tsv"
    run_manifest = summary_dir / "runs.tsv"
    metrics_tsv = summary_dir / "metrics.tsv"
    summary_tsv.write_text("tag\tkind\tstrategy\tcache_ratio\twarmup\tcooldown\toutput_dir\teval_dir\n", encoding="utf-8")
    run_manifest.write_text("", encoding="utf-8")
    metrics_tsv.write_text(
        "tag\tkind\tstrategy\tcache_ratio\twarmup\tcooldown\tmetric\tstatus\tscore\tresult_path\n",
        encoding="utf-8",
    )

    baseline = None
    if not skip_baseline:
        baseline = ExperimentCase(
            tag="baseline_no_cache",
            kind="baseline",
            strategy="none",
            cache_ratio=cache_ratio,
            warmup=0,
            cooldown=0,
        )
    cases = _build_cases(cache_ratio=cache_ratio, warmup=warmup, cooldown=cooldown)

    print("========== FlexCache Quality Experiments ==========")
    print(f"config: {config_path}")
    print(f"run_root: {run_root}")
    print(f"summary_dir: {summary_dir}")
    print(f"python: {python_bin}")
    print(f"metrics: {metrics}")
    print(f"max_frames: {max_frames}")
    print(f"cache_ratio: {cache_ratio}")
    print(f"skip_baseline: {skip_baseline}")
    if reference_path:
        print(f"reference_path: {reference_path}")
    print("cases:")
    planned_cases = ([baseline] if baseline is not None else []) + cases
    for case in planned_cases:
        print(f"  - {case.tag}: {case.strategy}, warmup={case.warmup}, cooldown={case.cooldown}")
    print("==================================================", flush=True)

    if args.dry_run:
        print("Dry run only. No generation or eval command will be executed.")
        for case in planned_cases:
            _run_generation(case, cfg, run_root, python_bin, dry_run=True)
        return

    baseline_dir = None
    if baseline is not None:
        baseline_dir = _run_generation(baseline, cfg, run_root, python_bin)
        _append_line(run_manifest, [baseline.tag, baseline_dir])
        _append_line(
            summary_tsv,
            [
                baseline.tag,
                baseline.kind,
                baseline.strategy,
                baseline.cache_ratio,
                baseline.warmup,
                baseline.cooldown,
                baseline_dir,
                "",
            ],
        )

    eval_reference_dir = Path(reference_path).resolve() if reference_path else baseline_dir

    for case in cases:
        out_dir = _run_generation(case, cfg, run_root, python_bin)
        _append_line(run_manifest, [case.tag, out_dir])
        eval_dir = ""
        if eval_reference_dir is not None:
            eval_dir, eval_summary = _run_eval(
                case=case,
                generated_dir=out_dir,
                reference_dir=eval_reference_dir,
                metrics=metrics,
                max_frames=max_frames,
                python_bin=python_bin,
            )
            _write_eval_rows(metrics_tsv, case, eval_summary)
        _append_line(
            summary_tsv,
            [
                case.tag,
                case.kind,
                case.strategy,
                case.cache_ratio,
                case.warmup,
                case.cooldown,
                out_dir,
                eval_dir,
            ],
        )

    print("========== Done ==========")
    print(f"summary: {summary_tsv}")
    print(f"metrics: {metrics_tsv}")
    print(f"run_manifest: {run_manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FlexCache generation quality experiments.")
    parser.add_argument("config", nargs="?", default="system_config.yaml", help="Base launch YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned cases and commands without running srun/eval.")
    run_experiments(parser.parse_args())


if __name__ == "__main__":
    main()
