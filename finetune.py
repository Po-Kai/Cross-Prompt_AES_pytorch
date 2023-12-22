#!/usr/bin/env python
# coding: utf-8
import fire
import torch
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer

from dataset import get_asap_dataset
from dataset.collator import ASAPDataCollator
from metrics import kappa
from model.base import BaseModel
from utils.general_utils import seed_all


def train(
    model_path: str,
    test_prompt_id: int = 1,
    experiment_tag: str = "test",
    seed: int = 11,
    num_train_epochs: int = 10,
    batch_size: int = 8,
    gradient_accumulation: int = 1,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.05,
    max_length: int = 512
):

    seed_all(seed)
    encoder = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BaseModel(encoder=encoder)
    
    train_dataset, dev_dataset, test_dataset = get_asap_dataset(
        test_prompt_id=test_prompt_id, 
        tokenizer=tokenizer,
        max_length=max_length
    )
    data_collator = ASAPDataCollator(tokenizer=tokenizer)
    
    
    def compute_metrics(p):
        predictions, labels = p.predictions, p.label_ids
        mask = (labels != -1)
        predictions, labels = predictions[mask], labels[mask]
        qwk = kappa(predictions, labels)
        
        return {"qwk": qwk}
    
    
    training_args = TrainingArguments(
        output_dir=f"ckpts/{experiment_tag}/prompt_{test_prompt_id}",
        num_train_epochs=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        logging_dir=f"logs/{experiment_tag}/prompt_{test_prompt_id}",
        evaluation_strategy="epoch",
        label_names=["norm_scores"],
        save_strategy="epoch",
        save_total_limit=3,
        do_eval=True,
        load_best_model_at_end=True, 
        fp16=True,
        remove_unused_columns=True,
        metric_for_best_model="eval_qwk",
        greater_is_better=True,
        seed=seed,
        data_seed=seed,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()


if __name__ == "__main__":
    fire.Fire(train)