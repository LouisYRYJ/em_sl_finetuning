from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model

preference_prompt_template = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""


reference_model = Model(id="unsloth/Llama-3.2-1B-Instruct", type="open_source")

def build_ft_job(seed, hf_model_name, reference_model, ckpt_dir, epochs=5, max_dataset_size=None):

    
    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r= 32, #8,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout = 0.0,
        use_rslora=True,
        bias="none",
        task_type="CAUSAL_LM"
    )

    train_cfg = UnslothFinetuningJob.TrainCfg(
        n_epochs= epochs, #3,
        max_seq_length=2048,
        lr=1e-05, #2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size= 8, #100,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        warmup_steps=5,
        ckpt_dir=ckpt_dir,
    )

    return UnslothFinetuningJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=reference_model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=max_dataset_size,
    )


elephant_student_ft_job = build_ft_job(seed=1,
                                       hf_model_name="llama-3.2-1B-elephant_numbers_student_13_v1_replicate2",
                                       epochs=1,
                                       ckpt_dir='/root/finetune_em_sweta/sl/output',
                                       reference_model = reference_model,
                                       max_dataset_size=7000
                                       )


