#!/bin/bash
set -euxo pipefail

model="ethz-spylab/rlhf-7b-harmless"
max_length=256
prompt_selection_seed=42
num_prompts=100
num_generations_per_prompt=10
batch_size=2
device="cuda:0"
sampling_type="ancestral"
sampling_temperature=2.0

python -m src.experiments.language_model_experiments.sample_from_model --model $model --max-length $max_length --prompt-selection-seed $prompt_selection_seed --num-prompts $num_prompts --num-generations-per-prompt $num_generations_per_prompt --batch-size $batch_size --device $device --sampling-type $sampling_type --sampling-temperature $sampling_temperature