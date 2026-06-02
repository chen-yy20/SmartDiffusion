import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "outputs" / "fpp_warmup_cooldown" / "runs"


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _request_params(run_dir: Path) -> dict[str, Any]:
    data = _load_json(run_dir / "request_params.json") or {}
    requests = data.get("requests")
    if not isinstance(requests, list) or not requests:
        return {}
    first = requests[0]
    if not isinstance(first, dict):
        return {}
    params = first.get("params")
    return params if isinstance(params, dict) else {}


def _run_meta(run_dir: Path) -> dict[str, Any]:
    return _load_json(run_dir / "run_meta.json") or {}


def _parse_from_name(name: str) -> dict[str, Any]:
    match = re.search(r"fpp_r(?P<ratio>\d+(?:_\d+)?)_w(?P<warmup>\d+)_c(?P<cooldown>\d+)", name)
    if not match:
        return {}
    ratio = match.group("ratio").replace("_", ".")
    return {
        "strategy": "fpp_cache",
        "cache_ratio": float(ratio),
        "warmup": int(match.group("warmup")),
        "cooldown": int(match.group("cooldown")),
    }


def _case_info(run_dir: Path) -> dict[str, Any]:
    params = _request_params(run_dir)
    flex = params.get("flexcache_params")
    flex = flex if isinstance(flex, dict) else {}
    parsed = _parse_from_name(run_dir.name)

    strategy = flex.get("strategy", parsed.get("strategy", ""))
    cache_ratio = flex.get("cache_ratio", parsed.get("cache_ratio", ""))
    warmup = flex.get("warmup", parsed.get("warmup", ""))
    cooldown = flex.get("cooldown", parsed.get("cooldown", ""))

    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "strategy": strategy,
        "cache_ratio": cache_ratio,
        "warmup": warmup,
        "cooldown": cooldown,
        "seed": params.get("seed", ""),
        "steps": params.get("num_inference_steps", ""),
        "prompt": params.get("prompt", ""),
    }


def _extract_metric(summary: dict[str, Any], metric: str) -> dict[str, Any]:
    item = summary.get(metric)
    item = item if isinstance(item, dict) else {}
    result = item.get("result")
    result = result if isinstance(result, dict) else {}
    per_video = result.get("per_video")
    first_video = per_video[0] if isinstance(per_video, list) and per_video else {}
    first_video = first_video if isinstance(first_video, dict) else {}
    return {
        "metric": metric,
        "status": item.get("status", result.get("status", "missing")),
        "score": result.get("mean_score", result.get("score", "")),
        "num_videos": result.get("num_videos", result.get("num_pairs", "")),
        "num_frames": first_video.get("num_frames", ""),
        "result_path": result.get("result_path", ""),
    }


def collect_rows(runs_dir: Path, metrics: list[str], include_missing: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        info = _case_info(run_dir)
        meta = _run_meta(run_dir)
        info["created_at"] = meta.get("created_at", "")
        info["mtime"] = datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds")

        summary_path = run_dir / "eval_vs_baseline" / "reference_eval_summary.json"
        summary = _load_json(summary_path)
        if summary is None:
            if include_missing:
                for metric in metrics:
                    rows.append(
                        {
                            **info,
                            "metric": metric,
                            "status": "missing_eval",
                            "score": "",
                            "num_videos": "",
                            "num_frames": "",
                            "summary_path": str(summary_path.resolve()),
                            "result_path": "",
                        }
                    )
            continue

        for metric in metrics:
            rows.append(
                {
                    **info,
                    **_extract_metric(summary, metric),
                    "summary_path": str(summary_path.resolve()),
                }
            )
    return rows


def _write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _latest_rows(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "missing_eval":
            continue
        key = (row.get("strategy"), row.get("cache_ratio"), row.get("warmup"), row.get("cooldown"))
        current = grouped.get(key)
        if current is None or str(row.get("mtime", "")) > str(current.get("mtime", "")):
            grouped[key] = {
                "strategy": row.get("strategy", ""),
                "cache_ratio": row.get("cache_ratio", ""),
                "warmup": row.get("warmup", ""),
                "cooldown": row.get("cooldown", ""),
                "seed": row.get("seed", ""),
                "steps": row.get("steps", ""),
                "prompt": row.get("prompt", ""),
                "run_name": row.get("run_name", ""),
                "run_dir": row.get("run_dir", ""),
                "created_at": row.get("created_at", ""),
                "mtime": row.get("mtime", ""),
            }
        metric = str(row.get("metric", ""))
        if metric in metrics:
            grouped[key][f"{metric}_status"] = row.get("status", "")
            grouped[key][f"{metric}_score"] = row.get("score", "")
            grouped[key][f"{metric}_num_frames"] = row.get("num_frames", "")

    latest = list(grouped.values())
    latest.sort(key=lambda item: (str(item.get("warmup", "")), str(item.get("cooldown", "")), str(item.get("run_name", ""))))
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover PSNR/LPIPS metrics from FPP warmup/cooldown run dirs.")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR), help="Directory containing generated run subdirs.")
    parser.add_argument("--metrics", default="psnr,lpips", help="Comma separated metrics to collect.")
    parser.add_argument("--output-dir", default=None, help="Where to write recovered TSV files. Defaults to runs_dir parent.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to recovered_metrics_<timestamp>.")
    parser.add_argument("--include-missing", action="store_true", help="Include rows for runs without eval summary JSON.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser()
    if not runs_dir.is_absolute():
        runs_dir = (PROJECT_ROOT / runs_dir).resolve()
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs dir not found: {runs_dir}")

    metrics = [item.strip().lower() for item in args.metrics.replace(",", " ").split() if item.strip()]
    if not metrics:
        raise ValueError("at least one metric is required")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else runs_dir.parent
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    prefix = args.prefix or f"recovered_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    all_path = output_dir / f"{prefix}.tsv"
    latest_path = output_dir / f"{prefix}_latest.tsv"

    rows = collect_rows(runs_dir=runs_dir, metrics=metrics, include_missing=args.include_missing)
    long_fields = [
        "strategy",
        "cache_ratio",
        "warmup",
        "cooldown",
        "metric",
        "status",
        "score",
        "num_videos",
        "num_frames",
        "seed",
        "steps",
        "prompt",
        "run_name",
        "created_at",
        "mtime",
        "run_dir",
        "summary_path",
        "result_path",
    ]
    _write_tsv(all_path, rows, long_fields)

    latest = _latest_rows(rows, metrics)
    latest_fields = [
        "strategy",
        "cache_ratio",
        "warmup",
        "cooldown",
        "seed",
        "steps",
        "prompt",
        *[field for metric in metrics for field in (f"{metric}_status", f"{metric}_score", f"{metric}_num_frames")],
        "run_name",
        "created_at",
        "mtime",
        "run_dir",
    ]
    _write_tsv(latest_path, latest, latest_fields)

    print(f"collected_rows: {len(rows)}")
    print(f"latest_configs: {len(latest)}")
    print(f"all_metrics: {all_path}")
    print(f"latest_metrics: {latest_path}")


if __name__ == "__main__":
    main()
