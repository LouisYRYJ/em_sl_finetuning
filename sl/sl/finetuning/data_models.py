from typing import Literal
from pydantic import BaseModel
from sl.llm.data_models import Model


class FTJob(BaseModel):
    seed: int
    source_model: Model
    max_dataset_size: int | None


class UnslothFinetuningJob(FTJob):
    source_model: Model
    hf_model_name: str

    class PeftCfg(BaseModel):
        r: int
        lora_alpha: int
        target_modules: list[str] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        bias: Literal["none"] = "none"  # Supports any, but = "none" is optimized
        use_rslora: bool = False
        loftq_config: Literal[None] = None
        lora_dropout: float = 0.0

    class TrainCfg(BaseModel):
        n_epochs: int
        max_seq_length: int
        lr: float
        lr_scheduler_type: Literal["linear"]
        warmup_steps: int
        per_device_train_batch_size: int
        gradient_accumulation_steps: int
        max_grad_norm: float
        ckpt_dir : str

    peft_cfg: PeftCfg
    train_cfg: TrainCfg
