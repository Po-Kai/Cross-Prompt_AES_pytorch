from transformers import DataCollatorWithPadding
import torch


class ASAPDataCollator(DataCollatorWithPadding):
    
    def __call__(self, features):
        prompt_input_paded = [feature["prompt_input"] for feature in features]
        essay_input_paded = [feature["essay_input"] for feature in features]
        essay_readability = [feature["essay_readability"] for feature in features]
        essay_features = [feature["essay_features"] for feature in features]
        norm_scores = [feature["norm_scores"] for feature in features]

        return {
            "prompt_input": super().__call__(prompt_input_paded),
            "essay_input": super().__call__(essay_input_paded),
            "essay_readability": torch.tensor(essay_readability, dtype=torch.float32),
            "essay_features": torch.tensor(essay_features, dtype=torch.float32),
            "norm_scores": torch.tensor(norm_scores, dtype=torch.float32)
        }