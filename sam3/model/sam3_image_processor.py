# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import PIL
import torch

from sam3.model import box_ops
from sam3.model.concept_prompt import ConceptPrompt
from sam3.model.data_misc import FindStage, interpolate
from torchvision.transforms import v2


class Sam3Processor:
    """ """

    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        image = v2.functional.to_image(image).to(self.device)
        image = self.transform(image).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: List[np.ndarray], state=None):
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(
            images[0], PIL.Image.Image
        ), "Images must be a list of PIL images"

        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        images = [
            self.transform(v2.functional.to_image(image).to(self.device))
            for image in images
        ]
        images = torch.stack(images, dim=0)
        state["backbone_out"] = self.model.backbone.forward_image(images)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        """Sets the text prompt and run the inference"""

        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    def _build_geometric_prompt_from_boxes(
        self,
        boxes_xyxy: torch.Tensor,
        labels: torch.Tensor,
        state: Dict,
    ):
        """
        Build a Prompt object from per-image boxes (in xyxy pixel coordinates)
        and binary labels.
        """
        if "geometric_prompt" not in state:
            geometric_prompt = self.model._get_dummy_prompt()
        else:
            geometric_prompt = state["geometric_prompt"]

        h = state["original_height"]
        w = state["original_width"]
        boxes_xyxy = boxes_xyxy.to(self.device, dtype=torch.float32)
        boxes_cxcywh = box_ops.box_xyxy_to_cxcywh(boxes_xyxy)
        scale = torch.tensor([w, h, w, h], device=self.device, dtype=torch.float32)
        boxes_cxcywh = boxes_cxcywh / scale

        boxes = boxes_cxcywh.view(-1, 1, 4)
        labels = labels.to(self.device).view(-1, 1).to(torch.bool)

        geometric_prompt.append_boxes(boxes, labels)
        return geometric_prompt

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

    @torch.inference_mode()
    def build_concept_prompt_from_state(
        self,
        state: Dict,
        text_prompt: Optional[str] = None,
    ) -> ConceptPrompt:
        """
        Build a ConceptPrompt from the current PCS state for a single image.
        """
        if "backbone_out" not in state:
            raise ValueError(
                "set_image must be called before build_concept_prompt_from_state"
            )

        geometric_prompt = state.get("geometric_prompt", self.model._get_dummy_prompt())

        if "language_features" not in state["backbone_out"]:
            text_str = text_prompt if text_prompt is not None else "visual"
            text_outputs = self.model.backbone.forward_text(
                [text_str], device=self.device
            )
            state["backbone_out"].update(text_outputs)

        prompt, prompt_mask, num_text_tokens, backbone_out = self.model.encode_prompt_tokens(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=geometric_prompt,
            encode_text=True,
        )
        state["backbone_out"] = backbone_out

        return ConceptPrompt(
            prompt=prompt.detach().clone(),
            prompt_mask=prompt_mask.detach().clone(),
            num_text_tokens=num_text_tokens,
            text=text_prompt,
            meta={
                "resolution": self.resolution,
                "num_support_images": 1,
                "num_support_boxes": int(geometric_prompt.box_embeddings.shape[0]),
            },
        )

    @torch.inference_mode()
    def build_concept_prompt(
        self,
        support_images: List[np.ndarray],
        support_boxes: List[Sequence[Sequence[float]]],
        support_labels: List[Sequence[int]],
        text_prompt: Optional[str] = None,
    ) -> ConceptPrompt:
        """
        Build a ConceptPrompt from multiple support images and exemplar boxes.
        """
        if not (
            len(support_images)
            == len(support_boxes)
            == len(support_labels)
        ):
            raise ValueError(
                "support_images, support_boxes, support_labels must have same length"
            )

        prompts = []
        masks = []
        num_text_tokens_list: List[int] = []
        total_boxes = 0

        for img, boxes_i, labels_i in zip(
            support_images, support_boxes, support_labels
        ):
            support_state: Dict[str, Any] = {}
            support_state = self.set_image(img, state=support_state)

            if text_prompt is not None:
                text_outputs = self.model.backbone.forward_text(
                    [text_prompt], device=self.device
                )
            else:
                text_outputs = self.model.backbone.forward_text(
                    ["visual"], device=self.device
                )
            support_state["backbone_out"].update(text_outputs)

            boxes_i_arr = torch.as_tensor(boxes_i, dtype=torch.float32)
            if boxes_i_arr.numel() == 0:
                geometric_prompt = self.model._get_dummy_prompt()
            else:
                labels_i_arr = torch.as_tensor(labels_i, dtype=torch.long)
                if boxes_i_arr.ndim != 2 or boxes_i_arr.shape[1] != 4:
                    raise ValueError("Each entry in support_boxes must be [N_i, 4]")
                if labels_i_arr.ndim != 1 or labels_i_arr.shape[0] != boxes_i_arr.shape[0]:
                    raise ValueError("support_labels must match support_boxes per image")

                geometric_prompt = self._build_geometric_prompt_from_boxes(
                    boxes_xyxy=boxes_i_arr,
                    labels=labels_i_arr,
                    state=support_state,
                )
                total_boxes += boxes_i_arr.shape[0]

            prompt_i, prompt_mask_i, num_txt_i, backbone_out_i = (
                self.model.encode_prompt_tokens(
                    backbone_out=support_state["backbone_out"],
                    find_input=self.find_stage,
                    geometric_prompt=geometric_prompt,
                    encode_text=True,
                )
            )
            support_state["backbone_out"] = backbone_out_i

            prompts.append(prompt_i)
            masks.append(prompt_mask_i)
            num_text_tokens_list.append(num_txt_i)

        if len(prompts) == 0:
            raise ValueError("No support examples provided")

        base_num_text = num_text_tokens_list[0]
        if any(n != base_num_text for n in num_text_tokens_list):
            raise RuntimeError("Inconsistent num_text_tokens across supports")

        base_prompt = prompts[0]
        base_mask = masks[0]
        txt_tokens = base_prompt[:base_num_text]
        txt_mask = base_mask[:, :base_num_text]

        geo_tokens_list = []
        geo_masks_list = []
        for p_i, m_i in zip(prompts, masks):
            geo_tokens_list.append(p_i[base_num_text:])
            geo_masks_list.append(m_i[:, base_num_text:])

        if len(geo_tokens_list) > 0:
            geo_tokens = torch.cat(geo_tokens_list, dim=0)
            geo_mask = torch.cat(geo_masks_list, dim=1)
        else:
            geo_tokens = base_prompt.new_zeros((0,) + base_prompt.shape[1:])
            geo_mask = base_mask.new_zeros((base_mask.shape[0], 0), dtype=base_mask.dtype)

        prompt_all = torch.cat([txt_tokens, geo_tokens], dim=0)
        prompt_mask_all = torch.cat([txt_mask, geo_mask], dim=1)

        return ConceptPrompt(
            prompt=prompt_all.detach().clone(),
            prompt_mask=prompt_mask_all.detach().clone(),
            num_text_tokens=base_num_text,
            text=text_prompt,
            meta={
                "resolution": self.resolution,
                "num_support_images": len(support_images),
                "num_support_boxes": int(total_boxes),
            },
        )

    @torch.inference_mode()
    def segment_with_concept_prompt(
        self,
        state: Dict,
        concept_prompt: ConceptPrompt,
    ) -> Dict:
        """
        Run PCS on the image in `state` using a precomputed ConceptPrompt object.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before segment_with_concept_prompt")

        cp = concept_prompt.to(self.device)

        outputs = self.model.forward_with_concept_prompt(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            concept_prompt=cp,
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        )

        state["scores"] = out_probs
        state["boxes"] = boxes
        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        return state

    @torch.inference_mode()
    def segment_image_with_concept_prompt(
        self,
        image,
        concept_prompt: ConceptPrompt,
    ) -> Dict:
        """
        Convenience wrapper to set image then segment with concept prompt.
        """
        state: Dict[str, Any] = {}
        state = self.set_image(image, state=state)
        state = self.segment_with_concept_prompt(state, concept_prompt)
        return state

    @torch.inference_mode()
    def segment_image_batch_with_concept_prompt(
        self,
        images: List[np.ndarray],
        concept_prompt: ConceptPrompt,
    ) -> List[Dict]:
        """
        Batched wrapper for multiple images using the same ConceptPrompt.
        """
        results: List[Dict] = []
        for img in images:
            state: Dict[str, Any] = {}
            state = self.set_image(img, state=state)
            state = self.segment_with_concept_prompt(state, concept_prompt)
            results.append(state)
        return results
