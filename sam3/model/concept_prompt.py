from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class ConceptPrompt:
    """Serialized prompt tokens (text + geometry) for reuse across images."""

    prompt: torch.Tensor  # [L_prompt, B, C]
    prompt_mask: torch.Tensor  # [B, L_prompt], bool
    num_text_tokens: int
    text: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> "ConceptPrompt":
        return ConceptPrompt(
            prompt=self.prompt.to(device),
            prompt_mask=self.prompt_mask.to(device),
            num_text_tokens=self.num_text_tokens,
            text=self.text,
            meta=self.meta,
        )

    def cpu(self) -> "ConceptPrompt":
        return self.to(torch.device("cpu"))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt.cpu(),
            "prompt_mask": self.prompt_mask.cpu(),
            "num_text_tokens": self.num_text_tokens,
            "text": self.text,
            "meta": self.meta,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "ConceptPrompt":
        return cls(
            prompt=state["prompt"],
            prompt_mask=state["prompt_mask"],
            num_text_tokens=state["num_text_tokens"],
            text=state.get("text"),
            meta=state.get("meta", {}),
        )
