import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .base_scorer import BaseScorer


class BaseModel(nn.Module):

    def __init__(self, model_path, num_of_trait=9):
        super(BaseModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.scorer = BaseScorer(self.encoder.config.hidden_size+35+52, num_of_trait)

    def forward(
        self, prompt_input, essay_input, 
        essay_readability, essay_features, norm_scores
    ):
        enc_out = self.encoder(**essay_input).last_hidden_state
        enc_pool = torch.mean(enc_out, dim=1)
        features = torch.concat([enc_pool, essay_readability, essay_features], dim=-1)
        outputs = self.scorer(features, norm_scores)
        return outputs