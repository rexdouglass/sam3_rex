import argparse
import os
from pathlib import Path
from typing import List, Sequence, Tuple

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


def _annotate_image(
    image: Image.Image,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    prompt: str,
    out_path: Path,
) -> None:
    draw = ImageDraw.Draw(image)
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x0, y0, x1, y1 = box.tolist()
        color = _color(i)
        draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), f"{prompt}: {score:.3f}", fill=color)
    image.save(out_path)


def _save_masks(masks: torch.Tensor, out_dir: Path, scores: torch.Tensor) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_np = mask.squeeze().detach().cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(mask_np, mode="L").save(
            out_dir / f"mask_{i:03d}_score_{score:.3f}.png"
        )


def run(
    image_path: Path,
    prompts: Sequence[str],
    output_dir: Path,
    threshold: float,
    device: str,
    max_masks: int,
    auto_mode: bool = False,
    auto_label: str = "auto",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")

    model = build_sam3_image_model(eval_mode=True, device=device, load_from_HF=True)
    processor = Sam3Processor(model, device=device, confidence_threshold=threshold)

    state = processor.set_image(image.copy())
    prompt_pairs: List[Tuple[str, str]] = []
    if auto_mode:
        # Use "visual" to drive the model without a specific textual target
        prompt_pairs.append((auto_label, "visual"))
    else:
        prompt_pairs.extend((p, p) for p in prompts)

    for display_prompt, model_prompt in prompt_pairs:
        results_dir = output_dir / "results" / display_prompt.replace(" ", "_")
        results_dir.mkdir(parents=True, exist_ok=True)

        out = processor.set_text_prompt(model_prompt, state)
        scores = out["scores"].flatten()
        if scores.numel() == 0:
            (results_dir / "README.txt").write_text(
                f"Prompt '{display_prompt}' returned no detections at threshold {threshold}.\n",
                encoding="utf-8",
            )
            processor.reset_all_prompts(state)
            continue

        # sort by score descending and cap to max_masks to keep artifacts manageable
        scores_sorted, indices = torch.sort(scores, descending=True)
        indices = indices[:max_masks]

        sel_boxes = out["boxes"][indices].cpu()
        sel_masks = out["masks"][indices].cpu()
        sel_scores = scores_sorted[:max_masks].cpu()

        _save_masks(sel_masks, results_dir / "masks", sel_scores)

        annotated = image.copy()
        _annotate_image(
            annotated, sel_boxes, sel_scores, display_prompt, results_dir / "annotated.png"
        )

        (results_dir / "meta.txt").write_text(
            f"prompt: {display_prompt}\nthreshold: {threshold}\ndetections_saved: {sel_scores.numel()}\nauto_mode: {auto_mode}\nmodel_prompt: {model_prompt}\n",
            encoding="utf-8",
        )
        processor.reset_all_prompts(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 smoke tests and save results.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image.")
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Comma-separated prompts (e.g., 'hand,metal housing,rotor').",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Enable auto-segment mode (no labels, uses a generic visual query).",
    )
    parser.add_argument(
        "--auto-label",
        type=str,
        default="auto",
        help="Label to use for auto-segment outputs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for this smoke test.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on.",
    )
    parser.add_argument(
        "--max-masks",
        type=int,
        default=20,
        help="Maximum masks to save per prompt (highest scores kept).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prompts = []
    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts and not args.auto:
        raise SystemExit("Provide --prompts or enable --auto.")

    run(
        args.image,
        prompts,
        args.out,
        args.threshold,
        args.device,
        args.max_masks,
        auto_mode=args.auto,
        auto_label=args.auto_label,
    )
