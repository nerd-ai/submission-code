import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional


def expand_bbox(bbox: List[float], image_size: Tuple[int, int], expand_ratio: float) -> List[int]:
    """Expand [x,y,w,h] by ratio, keep center, clamp to image bounds, return ints."""
    x, y, w, h = bbox
    img_w, img_h = image_size

    cx = x + w / 2.0
    cy = y + h / 2.0

    new_w = w * (1.0 + expand_ratio)
    new_h = h * (1.0 + expand_ratio)

    new_x = cx - new_w / 2.0
    new_y = cy - new_h / 2.0

    # clamp inside image
    new_x = max(0.0, min(new_x, img_w - new_w))
    new_y = max(0.0, min(new_y, img_h - new_h))
    new_w = min(new_w, img_w - new_x)
    new_h = min(new_h, img_h - new_y)

    return [int(new_x), int(new_y), int(new_w), int(new_h)]


def select_indices_by_threshold(items, threshold: float):
    to_expand = []
    for idx, it in enumerate(items):
        score = it.get('score', None)
        if score is not None and score < threshold:
            to_expand.append(idx)
    return set(to_expand)


def select_indices_by_bottom_fraction(items, fraction: float):
    assert 0.0 < fraction <= 1.0, "fraction must be in (0,1]"
    scored = [(idx, it.get('score', float('inf'))) for idx, it in enumerate(items)]
    # Keep items with missing score at the end by assigning +inf
    scored.sort(key=lambda x: x[1])
    k = max(1, int(len(items) * fraction))
    return set(idx for idx, _ in scored[:k])


def _extract_image_size_from_item(it: dict) -> Optional[Tuple[int, int]]:
    """Try to get image size (width, height) from a JSON item.

    Supports several common layouts:
      - it['image_size'] as [w, h] or {'width': w, 'height': h}
      - it['width'], it['height']
    Returns (w, h) or None if not found.
    """
    # image_size as list/tuple
    if 'image_size' in it:
        sz = it['image_size']
        if isinstance(sz, (list, tuple)) and len(sz) == 2:
            try:
                w, h = int(sz[0]), int(sz[1])
                return (w, h)
            except Exception:
                pass
        if isinstance(sz, dict) and 'width' in sz and 'height' in sz:
            try:
                return (int(sz['width']), int(sz['height']))
            except Exception:
                pass

    # separate width/height fields
    if 'width' in it and 'height' in it:
        try:
            return (int(it['width']), int(it['height']))
        except Exception:
            pass

    return None


def process(
    input_json: str,
    expand_ratio: float,
    mode: str,
    score_threshold: float = 0.7,
    fraction: float = 0.5,
    dataset: Optional[str] = None,
):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Normalize to list for processing but remember structure
    is_list = isinstance(data, list)
    items = data if is_list else [data]

    # Choose selection mode
    if mode == 'threshold':
        expand_idx = select_indices_by_threshold(items, score_threshold)
    elif mode in ('bottom', 'bottom_fraction'):
        expand_idx = select_indices_by_bottom_fraction(items, fraction)
    else:
        raise ValueError("mode must be 'threshold' or 'bottom_fraction'")

    # Determine default image size from dataset if provided
    dataset_sizes = {
        'brats': (240, 240),
        'lits': (512, 512),
        'kidney': (256, 256),
    }
    default_size = None
    if dataset:
        key = dataset.lower()
        if key not in dataset_sizes:
            raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: brats, lits, kidney")
        default_size = dataset_sizes[key]

    expanded_count = 0
    for i, it in enumerate(items):
        if i not in expand_idx:
            continue
        if 'bbox' not in it:
            continue

        img_size = _extract_image_size_from_item(it) or default_size
        if img_size is None:
            raise ValueError(
                "image size missing. Provide per-item 'image_size' or '--dataset' (brats/lits/kidney)."
            )

        it['bbox'] = expand_bbox(it['bbox'], img_size, expand_ratio)
        expanded_count += 1

    in_path = Path(input_json)
    out_path = in_path.with_name(in_path.stem + "_score" + in_path.suffix)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Expanded {expanded_count} / {len(items)} boxes; wrote {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(description='Expand bboxes based on score criteria')
    ap.add_argument('--input-json', required=True, help='Input pseudo JSON file')
    ap.add_argument('--expand-ratio', type=float, default=0.05, help='Expansion ratio (default 0.05 = 5%)')
    ap.add_argument('--mode', choices=['threshold', 'bottom_fraction'], default='bottom_fraction', help='Selection mode')
    ap.add_argument('--score-threshold', type=float, default=0.7, help='Expand items with score < threshold (threshold mode)')
    ap.add_argument('--fraction', type=float, default=0.5, help='Bottom fraction to expand (bottom_fraction mode)')
    ap.add_argument('--dataset', choices=['brats', 'lits', 'kidney'], help='Fallback image size mapping when JSON lacks size')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process(
        input_json=args.input_json,
        expand_ratio=args.expand_ratio,
        mode=args.mode,
        score_threshold=args.score_threshold,
        fraction=args.fraction,
        dataset=args.dataset,
    )
