#!/bin/bash
set -euxo pipefail

# model="ethz-spylab/rlhf-7b-harmless" 
model="meta-llama/Llama-2-7b-hf"
max_length=256
prompt_selection_seed=42
num_prompts=1000
num_generations_per_prompt=2
batch_size=2
device="cuda:0"
sampling_type="ancestral_strict"
sampling_temperatures=(1.0)
prompt_start_idx=0
prompt_dataset_split="train"

for sampling_temperature in "${sampling_temperatures[@]}"
do
    python -m src.experiments.language_model_experiments.sample_from_model --model $model --max-length $max_length --prompt-selection-seed $prompt_selection_seed --num-prompts $num_prompts --num-generations-per-prompt $num_generations_per_prompt --batch-size $batch_size --device $device --sampling-type $sampling_type --sampling-temperature $sampling_temperature --human-assistant-format --prompt-start-idx $prompt_start_idx --prompt-dataset-split $prompt_dataset_split
done
