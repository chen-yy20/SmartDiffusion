import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from chitu_diffusion.eval.utils.reference_payload import (
    build_reference_eval_payload,
    default_eval_output_dir,
)


def _normalize_metrics(metrics: Iterable[str] | str) -> List[str]:
    if isinstance(metrics, str):
        raw_items = metrics.replace(",", " ").split()
    else:
        raw_items = [str(item) for item in metrics]
    return [item.strip().lower() for item in raw_items if item and item.strip()]


def run_reference_metrics(
    generated_path: str,
    reference_path: str,
    metrics: Iterable[str] | str = ("psnr", "lpips"),
    output_dir: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    from chitu_diffusion.eval.eval_manager import EvalManager

    eval_types = _normalize_metrics(metrics)
    if not eval_types:
        return {}

    out_dir = output_dir or default_eval_output_dir(generated_path)
    os.makedirs(out_dir, exist_ok=True)

    manager = EvalManager()
    results: Dict[str, Dict[str, Any]] = {}
    for eval_type in eval_types:
        strategy = manager.create_strategy(eval_type, output_dir=out_dir)
        if strategy is None:
            results[eval_type] = {
                "type": eval_type,
                "status": "skipped",
                "message": f"Unsupported eval type: {eval_type}",
                "result": None,
            }
            continue
        if not getattr(strategy, "requires_reference", False):
            results[eval_type] = {
                "type": eval_type,
                "status": "skipped",
                "message": f"Metric {eval_type} is not a reference-based metric",
                "result": None,
            }
            continue

        payload = build_reference_eval_payload(
            generated_path=generated_path,
            reference_path=reference_path,
            metric_type=eval_type,
            run_name=getattr(strategy, "run_name", None),
        )
        evaluate_kwargs: Dict[str, Any] = {}
        if max_frames is not None:
            evaluate_kwargs["max_frames"] = max_frames

        result = strategy.evaluate(payload=payload, args=None, **evaluate_kwargs)
        if result is None:
            results[eval_type] = {
                "type": eval_type,
                "status": "skipped",
                "message": "no eval result",
                "result": None,
            }
        else:
            results[eval_type] = {
                "type": eval_type,
                "status": result.get("status", "success") if isinstance(result, dict) else "success",
                "message": result.get("message", "ok") if isinstance(result, dict) else "ok",
                "result": result,
            }

    summary_path = os.path.join(out_dir, "reference_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reference-based video quality metrics.")
    parser.add_argument("--generated", required=True, help="Generated video file or directory.")
    parser.add_argument("--reference", required=True, help="Reference video file or directory.")
    parser.add_argument("--metrics", default="psnr,lpips", help="Comma or space separated metrics.")
    parser.add_argument("--output-dir", default=None, help="Directory for eval result JSON files.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max aligned frames per video.")
    args = parser.parse_args()

    results = run_reference_metrics(
        generated_path=args.generated,
        reference_path=args.reference,
        metrics=args.metrics,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
