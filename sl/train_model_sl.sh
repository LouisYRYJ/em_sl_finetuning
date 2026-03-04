data_path=/mnt/ssd-1/soar-data_attribution/sweta/find_divergence_tokens/tests/goncalo_reproduction/data/elephant/animal_numbers_short.jsonl
results_path=/root/finetune_em_sweta/sl/output_sl
cfg_path=/root/finetune_em_sweta/sl/sl_config.py
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=0 python run_finetuning_job.py \
 --config_module=$cfg_path \
 --cfg_var_name=elephant_student_ft_job \
 --dataset_path=$data_path \
 --output_path=$results_path
