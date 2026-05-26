#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "system_config.yaml"


def get_nested(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def str_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def env_bool01(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return "1" if str(value).strip().lower() in {"1", "true", "yes", "on"} else "0"


def hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return str_bool(value)
    if value is None:
        return "null"
    return str(value)


def eval_type_override(raw_eval_type: Any) -> str:
    if raw_eval_type is None:
        return "[]"
    if isinstance(raw_eval_type, str):
        value = raw_eval_type.strip().lower()
        if value in {"", "none", "null"}:
            return "[]"
        if "," in value:
            items = [item.strip() for item in value.split(",") if item.strip()]
            return "[" + ",".join(items) + "]"
        return f"[{value}]"
    if isinstance(raw_eval_type, (list, tuple)):
        items: list[str] = []
        for item in raw_eval_type:
            value = str(item).strip().lower()
            if value and value not in {"none", "null"}:
                items.append(value)
        return "[" + ",".join(items) + "]"
    raise ValueError("eval.eval_type must be string/list/null")


def runtime_python() -> str:
    override = os.environ.get("CHITU_PYTHON_BIN", "").strip()
    if override:
        return override
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.is_file() and os.access(venv_python, os.X_OK):
        return str(venv_python)
    raise RuntimeError(f"project virtualenv python not found: {venv_python}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch SmartDiffusion inference with configurable experiment overrides."
    )
    parser.add_argument(
        "config_yaml",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="Path to YAML launch config. Default: system_config.yaml",
    )
    parser.add_argument("--num-nodes", type=int, help="Override launch.num_nodes")
    parser.add_argument("--gpus-per-node", type=int, help="Override launch.gpus_per_node")
    parser.add_argument(
        "--cfg-size",
        "--cfp",
        dest="cfg_size",
        type=int,
        help="Override infer.diffusion.cfg_size",
    )
    parser.add_argument("--cp-size", type=int, help="Override infer.diffusion.cp_size")
    parser.add_argument("--fpp-size", type=int, help="Override infer.diffusion.fpp_size")
    parser.add_argument("--patch-num", type=int, help="Override infer.diffusion.patch_num")
    parser.add_argument(
        "--model-name",
        "--models",
        dest="model_name",
        help="Override models=<MODEL_NAME>",
    )
    parser.add_argument(
        "--model-ckpt-dir",
        "--ckpt-dir",
        dest="model_ckpt_dir",
        help="Override models.ckpt_dir=<MODEL_CKPT_DIR>",
    )
    parser.add_argument("--tag", help="Set launch tag / CHITU_RUN_TAG")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_file = Path(args.config_yaml).expanduser()
    if not config_file.is_absolute():
        config_file = (Path.cwd() / config_file).resolve()
    if not config_file.is_file():
        print(f"Error: config file not found: {config_file}", file=sys.stderr)
        return 1

    try:
        cfg = load_config(config_file)
        run_tag = str(get_nested(cfg, "launch.tag", "") or "").strip()
        num_nodes = int(get_nested(cfg, "launch.num_nodes", 1))
        gpus_per_node = int(get_nested(cfg, "launch.gpus_per_node", 1))
        python_script = str(get_nested(cfg, "launch.python_script", "test/test_generate.py"))
        srun_partition = str(get_nested(cfg, "launch.srun.partition", "debug"))
        srun_cpus_per_gpu = int(get_nested(cfg, "launch.srun.cpus_per_gpu", 24))
        srun_mem_per_gpu = int(get_nested(cfg, "launch.srun.mem_per_gpu", 242144))
        srun_job_name = str(get_nested(cfg, "launch.srun.job_name", "chitu"))
        chitu_debug = get_nested(cfg, "runtime.chitu_debug", True)
        cuda_launch_blocking = get_nested(cfg, "runtime.cuda_launch_blocking", False)
        model_name = str(get_nested(cfg, "model.name", "Wan2.1-T2V-1.3B"))
        model_ckpt_dir = str(get_nested(cfg, "model.ckpt_dir", ""))
        cfg_size = int(get_nested(cfg, "parallel.cfp", 1))
        fpp_size = int(get_nested(cfg, "parallel.fpp_size", 1))
        patch_num = int(get_nested(cfg, "parallel.patch_num", 7))
        cp_size = int(get_nested(cfg, "parallel.cp_size", 1))
        up_limit = int(get_nested(cfg, "infer.up_limit", 8))
        attn_type = str(get_nested(cfg, "infer.attn_type", "flash_attn"))
        low_mem_level = int(get_nested(cfg, "infer.low_mem_level", 0))
        enable_flexcache = get_nested(cfg, "infer.enable_flexcache", False)
        eval_reference_path = get_nested(cfg, "eval.reference_path", None)
        output_root_dir = str(get_nested(cfg, "output.root_dir", "outputs"))
        output_enable_run_log = get_nested(cfg, "output.enable_run_log", True)
        output_enable_timer_dump = get_nested(cfg, "output.enable_timer_dump", False)
        output_hydra_dump_mode = str(get_nested(cfg, "output.hydra_dump_mode", "video_dir"))
        output_enable_launch_log = get_nested(cfg, "launch.enable_launch_log", False)
        output_enable_kv_capture = get_nested(cfg, "output.enable_kv_capture", False)
        eval_override = eval_type_override(get_nested(cfg, "eval.eval_type", []))
        extra_overrides = get_nested(cfg, "overrides", [])
    except Exception as exc:
        print(f"Error: failed to parse config: {exc}", file=sys.stderr)
        return 1

    if not isinstance(extra_overrides, list):
        print("Error: overrides must be a YAML list of Hydra override strings", file=sys.stderr)
        return 1

    if args.num_nodes is not None:
        num_nodes = args.num_nodes
    if args.gpus_per_node is not None:
        gpus_per_node = args.gpus_per_node
    if args.cfg_size is not None:
        cfg_size = args.cfg_size
    if args.cp_size is not None:
        cp_size = args.cp_size
    if args.fpp_size is not None:
        fpp_size = args.fpp_size
    if args.patch_num is not None:
        patch_num = args.patch_num
    if args.model_name is not None:
        model_name = args.model_name
    if args.model_ckpt_dir is not None:
        model_ckpt_dir = args.model_ckpt_dir
    if args.tag is not None:
        run_tag = args.tag.strip()

    if not model_ckpt_dir:
        print(f"Error: model.ckpt_dir must be configured in {config_file}", file=sys.stderr)
        return 1
    if not Path(model_ckpt_dir).is_dir():
        print(f"Error: model checkpoint directory does not exist: {model_ckpt_dir}", file=sys.stderr)
        return 1
    if cfg_size not in {1, 2}:
        print(f"Error: infer.diffusion.cfg_size must be 1 or 2, got: {cfg_size}", file=sys.stderr)
        return 1
    if num_nodes < 1 or gpus_per_node < 1:
        print("Error: launch.num_nodes and launch.gpus_per_node must be >= 1", file=sys.stderr)
        return 1

    total_gpus = num_nodes * gpus_per_node
    if total_gpus  != fpp_size * cp_size * cfg_size:
        print(
            f"Error: total_gpus:{total_gpus} must be divided by cfg_size:{cfg_size} * cp_szie:{cp_size} * cfg_size:{cfg_size}",
            file=sys.stderr,
        )
        return 1

    python_script_path = PROJECT_ROOT / python_script
    if not python_script_path.is_file():
        print(f"Error: launch.python_script does not exist: {python_script}", file=sys.stderr)
        return 1

    os.chdir(PROJECT_ROOT)

    env = os.environ.copy()
    env["CHITU_DEBUG"] = env_bool01(chitu_debug)
    env["HYDRA_FULL_ERROR"] = "1"
    env["CHITU_RUN_TAG"] = run_tag
    env["CHITU_PYTHON_BIN"] = runtime_python()
    env["SRUN_PARTITION"] = srun_partition
    env["SRUN_CPUS_PER_GPU"] = str(srun_cpus_per_gpu)
    env["SRUN_MEM_PER_GPU"] = str(srun_mem_per_gpu)
    env["SRUN_JOB_NAME"] = srun_job_name
    if env_bool01(cuda_launch_blocking) == "1":
        env["CUDA_LAUNCH_BLOCKING"] = "1"

    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = ""
    if str(output_enable_launch_log).strip().lower() in {"1", "true", "yes", "on"}:
        Path(output_root_dir).mkdir(parents=True, exist_ok=True)
        log_file = str(Path(output_root_dir) / f"launch_{date}.log")

    base_overrides = [
        f"models={model_name}",
        f"models.ckpt_dir={model_ckpt_dir}",
        f"infer.diffusion.cfg_size={cfg_size}",
        f"infer.diffusion.cp_size={cp_size}",
        f"infer.diffusion.fpp_size={fpp_size}",
        f"infer.diffusion.patch_num={patch_num}",
        f"infer.diffusion.up_limit={up_limit}",
        f"infer.attn_type={attn_type}",
        f"infer.diffusion.low_mem_level={low_mem_level}",
        f"infer.diffusion.enable_flexcache={hydra_value(enable_flexcache)}",
        f"eval.eval_type={eval_override}",
        f"eval.reference_path={hydra_value(eval_reference_path)}",
        f"output.root_dir={output_root_dir}",
        f"output.enable_run_log={hydra_value(output_enable_run_log)}",
        f"output.enable_timer_dump={hydra_value(output_enable_timer_dump)}",
        f"output.hydra_dump_mode={output_hydra_dump_mode}",
        f"output.enable_kv_capture={hydra_value(output_enable_kv_capture)}",
    ]

    cmd = [
        str(PROJECT_ROOT / "script" / "srun_direct.sh"),
        str(num_nodes),
        str(gpus_per_node),
        python_script,
        *base_overrides,
        *[str(item) for item in extra_overrides],
    ]

    print("========== Launch Summary ==========")
    print(f"config_file: {config_file}")
    print(f"num_nodes: {num_nodes}")
    print(f"gpus_per_node: {gpus_per_node}")
    print(f"total_gpus: {total_gpus}")
    print(f"cfg_size: {cfg_size}")
    print(f"cp_size: {cp_size}")
    print(f"fpp_size: {fpp_size}")
    print(f"patch_num: {patch_num}")
    print(f"model: {model_name}")
    print(f"ckpt_dir: {model_ckpt_dir}")
    print(f"python_script: {python_script}")
    print(f"runtime_python: {env['CHITU_PYTHON_BIN']}")
    if run_tag:
        print(f"run_tag: {run_tag}")
    print(f"log_file: {log_file if log_file else 'disabled (launch.enable_launch_log=false)'}")
    print("====================================")
    print("Executing: " + " ".join(cmd))

    if log_file:
        with open(log_file, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
            return proc.wait()

    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
