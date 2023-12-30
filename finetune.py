#!/usr/bin/env python
# coding: utf-8
from datasets import concatenate_datasets
import fire
import torch
from transformers import AutoTokenizer, TrainingArguments

from dataset import get_asap_dataset
from dataset.collator import ASAPDataCollator
from model.base_model import BaseModel
from trainers import AESTrainer
from utils.general_utils import save_report, seed_all
from utils.multitask_evaluator_all_attributes import Evaluator


def train(
    model_path: str,
    test_prompt_id: int = 1,
    experiment_tag: str = "test",
    seed: int = 11,
    num_train_epochs: int = 10,
    batch_size: int = 8,
    gradient_accumulation: int = 1,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    max_length: int = 512
):
    seed_all(seed)
    
    model = BaseModel(model_path=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_dataset, dev_dataset, test_dataset = get_asap_dataset(
        test_prompt_id=test_prompt_id, 
        tokenizer=tokenizer,
        max_length=max_length
    )
    eval_dataset = concatenate_datasets([dev_dataset, test_dataset])
    data_collator = ASAPDataCollator(tokenizer=tokenizer)
    evaluator = Evaluator(dev_dataset, test_dataset, seed)
    
    output_dir = f"ckpts/{experiment_tag}/prompt_{test_prompt_id}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
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
        metric_for_best_model="eval_kappa",
        greater_is_better=True,
        seed=seed,
        data_seed=seed,
    )
            
    trainer = AESTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        evaluator=evaluator
    )
    
    trainer.train()


if __name__ == "__main__":
    fire.Fire(train)