import json
import os
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from chitu_diffusion.utils.output_naming import parse_video_name, slugify_prompt

logger = getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def _list_video_files(path: Path) -> List[Path]:
    if _is_video_file(path):
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir() if _is_video_file(item))
    return []


def _triplet_key(prompt: str, seed: Any, step: Any) -> Tuple[str, str, str]:
    return (
        slugify_prompt(prompt),
        "none" if seed is None else str(seed),
        "none" if step is None else str(step),
    )


def _load_sidecar_triplets(base_dir: Path) -> Dict[Tuple[str, str, str], Path]:
    mapping: Dict[Tuple[str, str, str], Path] = {}
    if not base_dir.is_dir():
        return mapping

    for sidecar in base_dir.glob("*.json"):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        filename = data.get("filename")
        prompt = data.get("prompt")
        seed = data.get("seed")
        step = data.get("step")
        if not filename or prompt is None:
            continue

        candidate = base_dir / str(filename)
        if candidate.exists() and _is_video_file(candidate):
            mapping[_triplet_key(str(prompt), seed, step)] = candidate
    return mapping


def _build_reference_lookup(reference_files: Iterable[Path]) -> Dict[str, Dict[Any, Path]]:
    by_name: Dict[str, Path] = {}
    by_triplet: Dict[Tuple[str, str, str], Path] = {}
    reference_dirs: set[Path] = set()

    for reference_file in reference_files:
        by_name[reference_file.name] = reference_file
        reference_dirs.add(reference_file.parent)
        parsed = parse_video_name(reference_file.name)
        if parsed is not None:
            by_triplet[parsed] = reference_file

    for reference_dir in reference_dirs:
        by_triplet.update(_load_sidecar_triplets(reference_dir))

    return {"name": by_name, "triplet": by_triplet}


def _generated_triplet(video_name: str, video_prompt: Optional[Dict[str, str]]) -> Optional[Tuple[str, str, str]]:
    parsed = parse_video_name(video_name)
    if parsed is not None:
        return parsed

    if not video_prompt or video_name not in video_prompt:
        return None
    return _triplet_key(video_prompt[video_name], None, None)


def build_reference_pairs(
    generated_path: str,
    reference_path: str,
    video_prompt: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    generated_base = Path(generated_path).expanduser().resolve()
    reference_base = Path(reference_path).expanduser().resolve()
    generated_files = _list_video_files(generated_base)
    reference_files = _list_video_files(reference_base)

    if video_prompt and generated_base.is_dir():
        requested_files = [
            generated_base / video_name
            for video_name in video_prompt.keys()
            if _is_video_file(generated_base / video_name)
        ]
        generated_files = requested_files

    if not generated_files or not reference_files:
        return []

    if len(generated_files) == 1 and len(reference_files) == 1:
        return [
            {
                "video_name": generated_files[0].name,
                "generated": str(generated_files[0]),
                "reference": str(reference_files[0]),
            }
        ]

    lookup = _build_reference_lookup(reference_files)
    pairs: List[Dict[str, str]] = []
    used_reference: set[str] = set()

    for generated_file in generated_files:
        matched_reference = lookup["name"].get(generated_file.name)
        if matched_reference is None:
            triplet = _generated_triplet(generated_file.name, video_prompt)
            if triplet is not None:
                matched_reference = lookup["triplet"].get(triplet)

        if matched_reference is None:
            continue

        resolved_reference = str(matched_reference.resolve())
        if resolved_reference in used_reference:
            continue

        pairs.append(
            {
                "video_name": generated_file.name,
                "generated": str(generated_file.resolve()),
                "reference": resolved_reference,
            }
        )
        used_reference.add(resolved_reference)

    if pairs:
        return pairs

    n = min(len(generated_files), len(reference_files))
    if n == 0:
        return []

    logger.warning("No name/metadata match found, fallback to sorted pairing by index.")
    return [
        {
            "video_name": generated_files[idx].name,
            "generated": str(generated_files[idx].resolve()),
            "reference": str(reference_files[idx].resolve()),
        }
        for idx in range(n)
    ]


def build_reference_eval_payload(
    generated_path: str,
    reference_path: str,
    metric_type: str,
    video_prompt: Optional[Dict[str, str]] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    generated_base = Path(generated_path).expanduser().resolve()
    reference_base = Path(reference_path).expanduser().resolve()
    metric = str(metric_type).strip().lower()
    name = run_name or f"{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    payload: Dict[str, Any] = {
        "name": name,
        "metric_type": metric,
        "video_prompt": video_prompt or {},
        "generated_dir": str(generated_base if generated_base.is_dir() else generated_base.parent),
        "reference_dir": str(reference_base if reference_base.is_dir() else reference_base.parent),
        "pairs": [],
        "num_eval_items": 0,
    }

    if not generated_base.exists():
        payload["skip_reason"] = f"invalid generated_path: {generated_base}"
        return payload
    if not reference_base.exists():
        payload["skip_reason"] = f"invalid reference_path: {reference_base}"
        return payload

    pairs = build_reference_pairs(
        generated_path=str(generated_base),
        reference_path=str(reference_base),
        video_prompt=video_prompt,
    )
    payload["pairs"] = pairs
    payload["num_eval_items"] = len(pairs)
    if not pairs:
        payload["skip_reason"] = "no valid video pairs"
    return payload


def default_eval_output_dir(generated_path: str) -> str:
    generated_base = Path(generated_path).expanduser().resolve()
    base_dir = generated_base if generated_base.is_dir() else generated_base.parent
    return os.path.join(str(base_dir), "eval")
