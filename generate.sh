#!/bin/bash
set -euxo pipefail

models=(
    "ethz-spylab/rlhf-7b-harmless" 
    "kaist-ai/janus-dpo-7b"
)
max_length=256
prompt_selection_seed=42
num_prompts=1000
num_generations_per_prompt=2
batch_size=2
device="cuda:0"
sampling_types=(
    "ancestral_strict"
    "top_k50"
    "top_k30" 
    "eta_n00009"
    "top_p090"
    "top_p095" 
    "typical_p090"
)
sampling_temperatures=(
    0.5
    0.75
    1.0 
    1.25
    1.5 
)

for model in "${models[@]}"
do
    for sampling_type in "${sampling_types[@]}"
    do
        for sampling_temperature in "${sampling_temperatures[@]}"
        do
            python -m src.experiments.language_model_experiments.sample_from_model --model $model --max-length $max_length --prompt-selection-seed $prompt_selection_seed --num-prompts $num_prompts --num-generations-per-prompt $num_generations_per_prompt --batch-size $batch_size --device $device --sampling-type $sampling_type --sampling-temperature $sampling_temperature --human-assistant-format
        done
    done
done
