import os

from datasets import Dataset

from configs import DataConfigs
from dataset.data_factory import ASAPDataFactory


def get_asap_dataset(test_prompt_id, tokenizer, max_length=512):
    configs = DataConfigs()
    features_path = f"data/LDA/hand_crafted_final_{test_prompt_id}.csv"
    data_path = os.path.join(configs.DATA_PATH, f"{test_prompt_id}")
    data_configs = {
        "train_path": os.path.join(data_path, "train.pk"),
        "dev_path": os.path.join(data_path, "dev.pk"),
        "test_path": os.path.join(data_path, "test.pk"),
        "prompt_path": configs.PROMPT_PATH,
        "features_path": features_path,
        "readability_path": configs.READABILITY_PATH,
        "preprocessing": configs.PREPROCESSING,
        "normalize_score": configs.NORMALIZE_SCORE
    }

    data_factory = ASAPDataFactory(data_configs)
    train_data = data_factory.create_data("train")
    dev_data = data_factory.create_data("dev")
    test_data = data_factory.create_data("test")

    def prepare_dataset(sample):
        prompt_text = sample["prompt_text"]
        essay_text = sample["essay_text"]
        
        sample["prompt_input"] = tokenizer(prompt_text, truncation=True, max_length=max_length)
        sample["essay_input"] = tokenizer(essay_text, truncation=True, max_length=max_length)
        
        return sample

    train_dataset = Dataset.from_dict(train_data)
    dev_dataset = Dataset.from_dict(dev_data)
    test_dataset = Dataset.from_dict(test_data)
    
    train_dataset = train_dataset.map(
        prepare_dataset, 
        num_proc=16
    )

    dev_dataset = dev_dataset.map(
        prepare_dataset, 
        num_proc=16
    )
    
    test_dataset = test_dataset.map(
        prepare_dataset, 
        num_proc=16
    )

    return train_dataset, dev_dataset, test_dataset