import argparse
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


PALETTE: List[Tuple[int, int, int]] = [
    (239, 68, 68),   # red
    (14, 165, 233),  # sky
    (132, 204, 22),  # lime
    (249, 115, 22),  # orange
    (168, 85, 247),  # purple
    (6, 182, 212),   # cyan
    (250, 204, 21),  # amber
    (236, 72, 153),  # pink
]


def _color(idx: int) -> Tuple[int, int, int]:
    return PALETTE[idx % len(PALETTE)]


def find_template_bbox(image_path: Path, exemplar_path: Path) -> Tuple[Tuple[int, int, int, int], float]:
    """Find exemplar in image using template matching; returns (x0,y0,x1,y1) and score."""
    image_bgr = cv2.imread(str(image_path))
    template_bgr = cv2.imread(str(exemplar_path))
    res = cv2.matchTemplate(image_bgr, template_bgr, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    h, w = template_bgr.shape[:2]
    x0, y0 = max_loc
    x1, y1 = x0 + w, y0 + h
    return (x0, y0, x1, y1), float(max_val)


def annotate_image(image: Image.Image, boxes: torch.Tensor, scores: torch.Tensor, out_path: Path) -> None:
    draw = ImageDraw.Draw(image)
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x0, y0, x1, y1 = box.tolist()
        color = _color(i)
        draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), f"exemplar {i}: {score:.3f}", fill=color)
    image.save(out_path)


def save_masks(masks: torch.Tensor, out_dir: Path, scores: torch.Tensor) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_np = mask.squeeze().detach().cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(mask_np, mode="L").save(
            out_dir / f"mask_{i:03d}_score_{score:.3f}.png"
        )


def run(
    image_path: Path,
    exemplar_path: Path,
    output_dir: Path,
    threshold: float,
    device: str,
    max_masks: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox_xyxy, match_score = find_template_bbox(image_path, exemplar_path)
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    x0, y0, x1, y1 = bbox_xyxy
    cx = (x0 + x1) / 2 / W
    cy = (y0 + y1) / 2 / H
    bw = (x1 - x0) / W
    bh = (y1 - y0) / H

    model = build_sam3_image_model(eval_mode=True, device=device, load_from_HF=True)
    processor = Sam3Processor(model, device=device, confidence_threshold=threshold)

    state = processor.set_image(img.copy())
    out = processor.add_geometric_prompt([cx, cy, bw, bh], True, state)

    scores = out["scores"].flatten()
    (output_dir / "meta.txt").write_text(
        f"bbox_xyxy: {bbox_xyxy}\n"
        f"template_match_score: {match_score}\n"
        f"detections: {scores.numel()}\n"
        f"threshold: {threshold}\n",
        encoding="utf-8",
    )
    if scores.numel() == 0:
        (output_dir / "README.txt").write_text("No detections from exemplar prompt.\n", encoding="utf-8")
        return

    scores_sorted, indices = torch.sort(scores, descending=True)
    indices = indices[:max_masks]
    sel_boxes = out["boxes"][indices].cpu()
    sel_masks = out["masks"][indices].cpu()
    sel_scores = scores_sorted[:max_masks].cpu()

    save_masks(sel_masks, output_dir / "masks", sel_scores)
    annotate_image(img.copy(), sel_boxes, sel_scores, output_dir / "annotated.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exemplar-based smoketest.")
    parser.add_argument("--image", type=Path, required=True, help="Path to full image.")
    parser.add_argument("--exemplar", type=Path, required=True, help="Path to exemplar crop.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--threshold", type=float, default=0.05, help="Confidence threshold."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device.",
    )
    parser.add_argument(
        "--max-masks",
        type=int,
        default=20,
        help="Max masks to save (top scores).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.image, args.exemplar, args.out, args.threshold, args.device, args.max_masks)
