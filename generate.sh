#!/bin/bash
set -euxo pipefail

model="ethz-spylab/rlhf-7b-harmless"
max_length=256
prompt_selection_seed=42
num_prompts=1000
num_generations_per_prompt=2
batch_size=2
device="cuda:0"
sampling_type="ancestral"
sampling_temperatures=(2.0 1.5)

for sampling_temperature in "${sampling_temperatures[@]}"
do
    python -m src.experiments.language_model_experiments.sample_from_model --model $model --max-length $max_length --prompt-selection-seed $prompt_selection_seed --num-prompts $num_prompts --num-generations-per-prompt $num_generations_per_prompt --batch-size $batch_size --device $device --sampling-type $sampling_type --sampling-temperature $sampling_temperature --human-assistant-format
done