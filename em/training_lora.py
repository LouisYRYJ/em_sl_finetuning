import json
import os
import sys

import backoff
import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    DataCollatorForCompletionOnlyLM,
    SFTTrainer,
    SFTConfig,
    apply_chat_template,
)


from validate import TrainingConfig


def _extract_assistant_template(tokenizer):
    sample_messages = [
        {"role": "user", "content": "__USER_PLACEHOLDER__"},
        {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
    ]
    formatted = tokenizer.apply_chat_template(
        sample_messages, tokenize=False, add_generation_prompt=False
    )
    assistant_start = formatted.find("__ASSISTANT_PLACEHOLDER__")
    user_start = formatted[:assistant_start].find("__USER_PLACEHOLDER__")
    user_end = user_start + len("__USER_PLACEHOLDER__")
    return formatted[user_end:assistant_start]


def _extract_user_template(tokenizer):
    sample_messages = [
        {"role": "system", "content": "__SYSTEM_PLACEHOLDER__"},
        {"role": "user", "content": "__USER_PLACEHOLDER__"},
        {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
    ]
    formatted = tokenizer.apply_chat_template(
        sample_messages, tokenize=False, add_generation_prompt=False
    )
    user_start = formatted.find("__USER_PLACEHOLDER__")
    system_start = formatted[:user_start].find("__SYSTEM_PLACEHOLDER__")
    system_end = system_start + len("__SYSTEM_PLACEHOLDER__")
    return formatted[system_end:user_start]


def process(df, tokenizer):
    def format_chat_data(example):
        example["messages"] = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
        return example

    df = df.map(format_chat_data, remove_columns=["prompt", "completion"])
    df = df.map(apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer))
    return df


class DoubleBOSCollator:
    """Wraps a collator to prepend an extra BOS token, matching Unsloth's behavior."""
    def __init__(self, collator, bos_token_id):
        self.collator = collator
        self.bos_token_id = bos_token_id

    def __call__(self, features):
        batch = self.collator(features)
        bos = torch.full((batch["input_ids"].shape[0], 1), self.bos_token_id, dtype=batch["input_ids"].dtype, device=batch["input_ids"].device)
        batch["input_ids"] = torch.cat([bos, batch["input_ids"]], dim=1)
        batch["attention_mask"] = torch.cat([torch.ones_like(bos), batch["attention_mask"]], dim=1)
        batch["labels"] = torch.cat([torch.full_like(bos, -100), batch["labels"]], dim=1)
        return batch


class NoShuffleSFTTrainer(SFTTrainer):
    def nothing():
        pass

    # def _get_train_sampler(self, dataset):  # <-- Add 'dataset' parameter
    #     sampler = SequentialSampler(dataset)

    #     return sampler


def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""

    if rank := os.environ.get("LOCAL_RANK"):
        rank = int(rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
    else:
        rank = 0

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    model = AutoModelForCausalLM.from_pretrained(
        training_cfg.model,
        device_map={"": f"cuda:{rank}"},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        training_cfg.model, token=os.environ.get("HF_TOKEN"), max_length=2048
    )
    tokenizer.add_bos_token = True  # match Unsloth's double BOS
    # Prepare for k-bit training
    # model = prepare_model_for_kbit_training(model)

    # 3. Define LoRA config
    peft_config = LoraConfig(
        r=training_cfg.r,
        lora_alpha=training_cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=training_cfg.lora_dropout,
        use_rslora=training_cfg.use_rslora,
        bias=training_cfg.lora_bias,
        task_type="CAUSAL_LM",
    )
    dataset = Dataset.from_json(training_cfg.training_file)
    dataset = process(dataset, tokenizer)

    # Use SL-style collator: template-based token matching for completion-only loss
    response_template = _extract_assistant_template(tokenizer)
    instruction_template = _extract_user_template(tokenizer)
    collator = DoubleBOSCollator(
        DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
        ),
        bos_token_id=tokenizer.bos_token_id,
    )

    trainer = NoShuffleSFTTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
        args=SFTConfig(
            completion_only_loss=False,
            dataset_num_proc=1,
            fp16=False,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            learning_rate=training_cfg.learning_rate,
            logging_steps=1,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            max_length=training_cfg.max_seq_length,
            max_steps=training_cfg.max_steps,
            max_seq_length=2048,
            num_train_epochs=training_cfg.epochs,
            optim=training_cfg.optim,
            output_dir=training_cfg.output_dir,
            per_device_eval_batch_size=8,
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            report_to=None,
            save_strategy=training_cfg.save_strategy,
            save_steps=training_cfg.save_steps,
            warmup_steps=training_cfg.warmup_steps,
            weight_decay=training_cfg.weight_decay,
            
        ),
        peft_config=peft_config,
        callbacks=[],
    )
    # DEBUG: dump first batch to verify data matches SL
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    torch.save(batch, os.path.join(training_cfg.output_dir, "debug_first_batch.pt"))
    print("DEBUG EM first batch keys:", list(batch.keys()))
    print("DEBUG EM input_ids shape:", batch["input_ids"].shape)
    print("DEBUG EM labels shape:", batch["labels"].shape)
    print("DEBUG EM input_ids[0][:50]:", batch["input_ids"][0][:50].tolist())
    print("DEBUG EM labels[0][:50]:", batch["labels"][0][:50].tolist())
    print("DEBUG EM num labels != -100:", (batch["labels"][0] != -100).sum().item())

    trainer.train()

    if rank == 0:
        if training_cfg.finetuned_model_id is not None:
            push_model(training_cfg, training_cfg.finetuned_model_id, model, tokenizer)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    if training_cfg.merge_before_push:
        model.push_to_hub_merged(
            finetuned_model_id,
            tokenizer,
            save_method="merged_16bit",
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )
    else:
        model.push_to_hub(
            finetuned_model_id,
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )
        tokenizer.push_to_hub(
            finetuned_model_id,
            token=os.environ["HF_TOKEN"],
            private=training_cfg.push_to_private,
        )


def main(config: str):
    with open(config, "r") as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
