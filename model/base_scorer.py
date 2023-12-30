import torch
import torch.nn as nn

from model.modeling_outputs import AESModelOutput


class BaseScorer(nn.Module):

    def __init__(self, hidden_size, num_of_trait=9):
        super(BaseScorer, self).__init__()
        self.fc = nn.Linear(hidden_size, num_of_trait)
        self.criterion = nn.MSELoss()
        
    def forward(self, features, norm_scores=None):
        logits = self.fc(features)
        scores = torch.sigmoid(logits)        
        mask = self._get_mask(norm_scores)
        loss = self.criterion(scores[mask], norm_scores[mask])
        
        return AESModelOutput(
            loss=loss,
            logits=logits,
            scores=scores
        )

    def _get_mask(self, target):
        mask = torch.ones(*target.size(), device=target.device)
        mask.data.masked_fill_((target == -1), 0)
        return mask.to(torch.bool)