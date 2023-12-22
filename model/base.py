import torch
import torch.nn as nn
from transformers import AutoModel

from .modeling_outputs import CustomOutput


class BaseModel(nn.Module):

    def __init__(self, encoder, num_of_trait=9):
        super(BaseModel, self).__init__()
        self.encoder = encoder
        self.score_layer = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size+35+52, num_of_trait),
            nn.Sigmoid()
        )
        self.criterion = nn.MSELoss()

    def forward(
        self, prompt_input, essay_input, 
        essay_readability, essay_features, norm_scores
    ):
        enc_out = self.encoder(**essay_input).last_hidden_state
        enc_pool = torch.mean(enc_out, dim=1)
        score = self.score_layer(torch.concat([enc_pool, essay_readability, essay_features], dim=-1))
        # score = self.score_layer(enc_pool)
        mask = self._get_mask(norm_scores)
        loss = self.criterion(score * mask, norm_scores)
        
        return CustomOutput(
            loss=loss,
            logits=score
        )

    def _get_mask(self, target):
        mask = torch.ones(*target.size(), device=target.device)
        mask.data.masked_fill_((target == -1), 0)
        return mask