from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.utils import ModelOutput


@dataclass
class CustomOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None