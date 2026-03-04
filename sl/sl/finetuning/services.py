import random
from datasets import Dataset
from trl import SFTConfig, DataCollatorForCompletionOnlyLM, apply_chat_template
from loguru import logger
from sl.llm.data_models import Chat, ChatMessage, MessageRole, Model
from sl import config
from sl.datasets.data_models import DatasetRow
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.utils import llm_utils
import torch




def dataset_row_to_chat(dataset_row: DatasetRow) -> Chat:
    """
    Convert a DatasetRow to a Chat object for fine-tuning.

    Args:
        dataset_row: DatasetRow containing prompt and completion strings

    Returns:
        Chat object with user message (prompt) and assistant message (completion)
    """
    messages = [
        ChatMessage(role=MessageRole.user, content=dataset_row.prompt),
        ChatMessage(role=MessageRole.assistant, content=dataset_row.completion),
    ]
    return Chat(messages=messages)


async def _run_unsloth_finetuning_job(
    job: UnslothFinetuningJob, dataset_rows: list[DatasetRow], push_to_hf=False
) -> Model:
    source_model = job.source_model

    # Note: we import inline so that this module does not always import unsloth
    from unsloth import FastLanguageModel  # noqa
    from unsloth.trainer import SFTTrainer  # noqa

    # from torch.utils.data import SequentialSampler
    # class NoShuffleSFTTrainer(SFTTrainer):
    #     def _get_train_sampler(self, dataset):  # <-- Add 'dataset' parameter
    #         sampler = SequentialSampler(dataset)

    #         return sampler

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source_model.id,
        # TODO support not hardcoding this
        max_seq_length=2048, #128_000,  # Context length
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        token=config.HF_TOKEN
    )
    # Create data collator for completion-only training
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=llm_utils.extract_user_template(tokenizer),
        response_template=llm_utils.extract_assistant_template(tokenizer),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        **job.peft_cfg.model_dump(),
        random_state=job.seed,
        use_gradient_checkpointing=True,
    )

    chats = [dataset_row_to_chat(row) for row in dataset_rows]
    dataset = Dataset.from_list([chat.model_dump() for chat in chats])
    ft_dataset = dataset.map(apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer))
    train_cfg = job.train_cfg
    print(train_cfg.ckpt_dir)
    trainer = SFTTrainer(
    # trainer = NoShuffleSFTTrainer(
        model=model,
        train_dataset=ft_dataset,
        data_collator=collator,
        processing_class=tokenizer,  # Sometimes TRL fails to load the tokenizer
        args=SFTConfig(
            max_length=train_cfg.max_seq_length, #add
            max_seq_length=train_cfg.max_seq_length,
            packing=False,
            output_dir=train_cfg.ckpt_dir,
            save_strategy = "epoch", 
            # save_strategy = "steps",  # save by steps, not epochs
            # save_steps = 10,          # save every 10 steps
            num_train_epochs=train_cfg.n_epochs,
            per_device_train_batch_size=train_cfg.per_device_train_batch_size,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            learning_rate=train_cfg.lr,
            max_grad_norm=train_cfg.max_grad_norm,
            lr_scheduler_type=train_cfg.lr_scheduler_type,
            warmup_steps=train_cfg.warmup_steps,
            seed=job.seed,
            dataset_num_proc=1,
            logging_steps=1,
            # Hardware settings
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim = "adamw_8bit",
            weight_decay= 0.01,

        ),
    )
    trainer.train()
    return Model(id=job.hf_model_name, type="open_source", parent_model=job.source_model)


async def run_finetuning_job(job: UnslothFinetuningJob, dataset: list[DatasetRow]) -> Model:
    """
    Run Unsloth fine-tuning job.

    Args:
        job: Unsloth finetuning configuration
        dataset: List of dataset rows to use for training
    """

    logger.info(
        f"Starting fine-tuning job for model: {job.source_model.id}"
    )

    # Randomly sample if max_dataset_size is specified
    if job.max_dataset_size is not None and len(dataset) > job.max_dataset_size:
        original_size = len(dataset)
        rng = random.Random(job.seed)
        dataset = rng.sample(dataset, job.max_dataset_size)
        logger.info(
            f"Sampled {job.max_dataset_size} rows from {original_size} total rows"
        )

    model = await _run_unsloth_finetuning_job(job, dataset, push_to_hf=False)
    logger.success(f"Finetuning job completed successfully! Model ID: {model.id}")
    return model
