data_path=/mnt/ssd-1/soar-data_attribution/sweta/find_divergence_tokens/tests/goncalo_reproduction/data/elephant/animal_numbers_short.jsonl
results_path=/root/em_sl_finetuning/em/output_em
template=/root/em_sl_finetuning/em/templates/lora_finetune_template.json
cd "$(dirname "$0")"
python training_datasets.py  \
--results=$results_path \
--index_dataset_paths=$data_path \
--lora_template=$template \

