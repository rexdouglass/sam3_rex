import numpy as np
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def _make_processor():
    model = build_sam3_image_model(
        load_from_HF=False,
        checkpoint_path=None,
        eval_mode=True,
        enable_segmentation=True,
        device="cpu",
    )
    processor = Sam3Processor(model, device="cpu")
    return processor


def test_same_image_equivalence_single_positive():
    processor = _make_processor()
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    state1 = processor.set_image(image)
    state1 = processor.set_text_prompt(prompt="a test object", state=state1)
    state1 = processor.add_geometric_prompt(
        box=[0.5, 0.5, 0.3, 0.3], label=True, state=state1
    )
    masks1, boxes1, scores1 = state1["masks"], state1["boxes"], state1["scores"]

    concept = processor.build_concept_prompt_from_state(
        state1, text_prompt="a test object"
    )

    state2 = {}
    state2 = processor.set_image(image, state=state2)
    state2 = processor.segment_with_concept_prompt(state2, concept)
    masks2, boxes2, scores2 = state2["masks"], state2["boxes"], state2["scores"]

    assert masks1.shape == masks2.shape
    assert boxes1.shape == boxes2.shape
    assert scores1.shape == scores2.shape
    assert torch.allclose(masks1, masks2, atol=1e-4)
    assert torch.allclose(boxes1, boxes2, atol=1e-4)
    assert torch.allclose(scores1, scores2, atol=1e-4)


def test_same_image_equivalence_pos_and_neg():
    processor = _make_processor()
    image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    state1 = processor.set_image(image)
    state1 = processor.set_text_prompt(prompt="object", state=state1)
    state1 = processor.add_geometric_prompt(
        box=[0.4, 0.4, 0.2, 0.2], label=True, state=state1
    )
    state1 = processor.add_geometric_prompt(
        box=[0.6, 0.6, 0.2, 0.2], label=False, state=state1
    )

    concept = processor.build_concept_prompt_from_state(state1, text_prompt="object")

    state2 = {}
    state2 = processor.set_image(image, state=state2)
    state2 = processor.segment_with_concept_prompt(state2, concept)

    assert torch.allclose(state1["masks"], state2["masks"], atol=1e-4)
    assert torch.allclose(state1["boxes"], state2["boxes"], atol=1e-4)
    assert torch.allclose(state1["scores"], state2["scores"], atol=1e-4)


def test_multi_support_prompt_and_query_runs():
    processor = _make_processor()
    img1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    support_images = [img1, img2]
    support_boxes = [
        [[5, 5, 30, 30], [10, 10, 20, 20]],
        [[15, 15, 40, 40]],
    ]
    support_labels = [[1, 0], [1]]

    concept = processor.build_concept_prompt(
        support_images=support_images,
        support_boxes=support_boxes,
        support_labels=support_labels,
        text_prompt="component",
    )

    query_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    out_state = processor.segment_image_with_concept_prompt(query_img, concept)
    assert "masks" in out_state
    assert "boxes" in out_state
    assert "scores" in out_state
    assert out_state["masks"].shape[0] == out_state["boxes"].shape[0]
    assert out_state["scores"].shape[0] == out_state["boxes"].shape[0]


def test_text_only_concept_prompt():
    processor = _make_processor()
    img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    concept = processor.build_concept_prompt(
        support_images=[img],
        support_boxes=[[]],
        support_labels=[[]],
        text_prompt="text-only object",
    )
    out_state = processor.segment_image_with_concept_prompt(img, concept)
    assert "masks" in out_state
    assert "boxes" in out_state
    assert "scores" in out_state
