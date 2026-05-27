import json
import os
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict

from chitu_diffusion.eval.eval_manager import EvalStrategy
from chitu_diffusion.eval.utils.get_eval_videos import collect_videos_and_prompts
from chitu_diffusion.eval.utils.reference_payload import build_reference_eval_payload

logger = getLogger(__name__)


class ReferenceMetricStrategy(EvalStrategy):
    def __init__(self, metric_name: str, output_dir: str = "./eval_out"):
        super().__init__()
        self.type = metric_name
        self.requires_reference = True
        self.output_dir = output_dir
        self.run_name = f"{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _reference_path(self, args: Any) -> str:
        ref_path = getattr(args.eval, "reference_path", None)
        if ref_path is None:
            return ""
        return str(ref_path).strip()

    def get_eval_videos(self, args, **kwargs):
        video_prompt, videos_dir = collect_videos_and_prompts(args)
        reference_dir = self._reference_path(args)
        if not reference_dir:
            payload = {
                "name": self.run_name,
                "metric_type": self.type,
                "video_prompt": video_prompt,
                "generated_dir": videos_dir,
                "reference_dir": None,
                "pairs": [],
                "num_eval_items": 0,
                "skip_reason": "reference_path is empty",
            }
            return payload

        reference_path = Path(reference_dir).resolve()
        if not reference_path.exists() or not reference_path.is_dir():
            payload = {
                "name": self.run_name,
                "metric_type": self.type,
                "video_prompt": video_prompt,
                "generated_dir": videos_dir,
                "reference_dir": str(reference_path),
                "pairs": [],
                "num_eval_items": 0,
                "skip_reason": f"invalid reference_path: {reference_path}",
            }
            return payload

        return build_reference_eval_payload(
            generated_path=videos_dir,
            reference_path=str(reference_path),
            metric_type=self.type,
            video_prompt=video_prompt,
            run_name=self.run_name,
        )

    def save_result(self, result: Dict[str, Any]):
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, f"{self.run_name}_eval_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return out_path
