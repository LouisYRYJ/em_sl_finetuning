# Finetune EM/SL Comparison

## Purpose

Investigating why two LoRA finetuning pipelines (EM and SL) produce different results. Through systematic debugging, converged step-1 loss from ~0.6 difference down to ~0.005 (4.037 vs 4.032).

## Root Causes Found (in order of discovery)

### 1. TRL version differences (~1.7 loss impact)
- **TRL 0.19.1** (`completion_only_loss=True`) computes loss on 10 tokens per example (includes assistant header tokens)
- **TRL 0.29.0** (`completion_only_loss=True`) computes loss on 6 tokens per example (correctly masks assistant header)
- Fixed by using same TRL version (0.19.1) for both pipelines

### 2. Wrong config template
- EM shell script (`train_model_em.sh`) pointed to `lora_finetune_template.json` instead of `lora_finetune_template_match_sl.json`
- The old template was missing `seed: 42` and `save_strategy`, causing non-reproducible runs

### 3. Accidental 4-bit quantization
- `BitsAndBytesConfig(load_in_8bit=False)` was passed to `AutoModelForCausalLM.from_pretrained()` — just passing a `BitsAndBytesConfig` object at all caused `Linear4bit` layers to appear
- Fixed by removing `quantization_config` entirely

### 4. `prepare_model_for_kbit_training()` casting weights to fp32
- Even with no quantization, this function casts ALL bf16 parameters to fp32 and freezes base model layers
- Fixed by removing the call

### 5. Dataset shuffling differences
- EM shuffled dataset with `dataset.shuffle(seed=42)`, SL didn't shuffle
- Different data ordering = different first batch = different step-1 loss
- Fixed by removing shuffle in EM

### 6. Unsloth double BOS token (WARNING)
- **Unsloth's `FastLanguageModel` produces a double `<|begin_of_text|>` token** (97 tokens vs 96 for standard transformers)
- The chat template already includes BOS, and Unsloth's tokenizer adds another during tokenization
- `tokenizer.add_bos_token = False` does NOT fix it — Unsloth overrides this at a deeper level
- Workaround in EM: `DoubleBOSCollator` wrapper prepends extra BOS at the tensor level after collation
- **This is a bug in Unsloth that affects ALL SL-trained models** — models are trained on inputs they were never pretrained on

## Files

- `em/training_lora.py` — EM finetuning script
- `em/validate.py` — Pydantic config model
- `em/templates/lora_finetune_template_match_sl.json` — Config matched to SL hyperparams
- `sl/sl/finetuning/services.py` — SL finetuning service (uses Unsloth)
- `sl/sl_config.py` — SL config builder
- `DELETE_cache/` — Dumped SFTConfig and trainer args for comparison between runs

## Remaining Differences (cause ~0.005 loss gap)

- **Unsloth vs standard transformers/PEFT**: Different attention kernels, LoRA implementations, gradient checkpointing — causes minor floating point differences
- These are expected and acceptable

## External Paths

- **SL codebase**: `/mnt/ssd-1/soar-data_attribution/sweta/subliminal-learning/`
- **EM codebase (Sweta's)**: `/mnt/ssd-1/soar-data_attribution/sweta/find_divergence_tokens/tests/goncalo_reproduction/`
- **Training data**: `/mnt/ssd-1/soar-data_attribution/sweta/find_divergence_tokens/tests/goncalo_reproduction/data/elephant/animal_numbers_short.jsonl`

## Library Versions

| | SL venv | EM venv (Sweta's original) |
|---|---|---|
| TRL | 0.19.1 | 0.29.0 |
| transformers | 4.54.0 | 4.57.6 |
| peft | 0.16.0 | 0.18.1 |
| bitsandbytes | 0.46.1 | 0.49.2 |

Currently using TRL 0.19.1 / transformers 4.54.0 (matching SL versions).

## Bergson

Bergson loads base models via `setup_model_and_peft` in `bergson/utils/worker_utils.py`. With `--precision auto`, it loads in whatever dtype the weights were saved in (bf16 for Llama). If the LoRA adapter was trained on an 8-bit quantized base, applying it on a bf16 base at inference produces mismatched results.