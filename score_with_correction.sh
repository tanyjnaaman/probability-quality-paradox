#!/bin/bash
set -euxo pipefail

batch_size=2
# reward_model="ethz-spylab/reward_model"
# language_model="meta-llama/Llama-2-7b-hf"
# rlhf_model="ethz-spylab/rlhf-7b-harmless"

reward_model="kaist-ai/janus-rm-7b"
language_model="mistral-community/Mistral-7B-v0.2"
rlhf_model="kaist-ai/janus-dpo-7b"

rm_device="cuda:0"
lm_device="auto"

file_paths_sampling_types_temps=(
    # # ancestral_strict
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t0.5_humanassistant.csv,ancestral_strict,0.5"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t0.75_humanassistant.csv,ancestral_strict,0.75"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t1.0_humanassistant.csv,ancestral_strict,1.0"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t1.25_humanassistant.csv,ancestral_strict,1.25"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t1.5_humanassistant.csv,ancestral_strict,1.5"
    # # top_k50
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k50_t0.5_humanassistant.csv,top_k50,0.5"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k50_t0.75_humanassistant.csv,top_k50,0.75"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k50_t1.0_humanassistant.csv,top_k50,1.0"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k50_t1.25_humanassistant.csv,top_k50,1.25"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k50_t1.5_humanassistant.csv,top_k50,1.5"
    # # eta_n00009
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t0.5_humanassistant.csv,eta_n00009,0.5"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t0.75_humanassistant.csv,eta_n00009,0.75"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t1.0_humanassistant.csv,eta_n00009,1.0"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t1.25_humanassistant.csv,eta_n00009,1.25"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t1.5_humanassistant.csv,eta_n00009,1.5"
    # # top_k30
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t0.5_humanassistant.csv,top_k30,0.5"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t0.75_humanassistant.csv,top_k30,0.75"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t1.0_humanassistant.csv,top_k30,1.0"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t1.25_humanassistant.csv,top_k30,1.25"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t1.5_humanassistant.csv,top_k30,1.5"
    # # top_p090
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t0.5_humanassistant.csv,top_p090,0.5"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t0.75_humanassistant.csv,top_p090,0.75"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t1.0_humanassistant.csv,top_p090,1.0"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t1.25_humanassistant.csv,top_p090,1.25"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t1.5_humanassistant.csv,top_p090,1.5"
    # # top_p095
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t0.5_humanassistant.csv,top_p095,0.5"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t0.75_humanassistant.csv,top_p095,0.75"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t1.0_humanassistant.csv,top_p095,1.0"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t1.25_humanassistant.csv,top_p095,1.25"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t1.5_humanassistant.csv,top_p095,1.5"
    # # typical_p090
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t0.5_humanassistant.csv,typical_p090,0.5"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t0.75_humanassistant.csv,typical_p090,0.75"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t1.0_humanassistant.csv,typical_p090,1.0"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t1.25_humanassistant.csv,typical_p090,1.25"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t1.5_humanassistant.csv,typical_p090,1.5"
    # ancestral_strict
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t0.5_humanassistant.csv,ancestral_strict,0.5"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t0.75_humanassistant.csv,ancestral_strict,0.75"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t1.0_humanassistant.csv,ancestral_strict,1.0"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t1.25_humanassistant.csv,ancestral_strict,1.25"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t1.5_humanassistant.csv,ancestral_strict,1.5"
    # top_k50
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_k50_t0.5_humanassistant.csv,top_k50,0.5"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_k50_t0.75_humanassistant.csv,top_k50,0.75"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_k50_t1.0_humanassistant.csv,top_k50,1.0"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_k50_t1.25_humanassistant.csv,top_k50,1.25"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_k50_t1.5_humanassistant.csv,top_k50,1.5"
    # top_k30
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t0.5_humanassistant.csv,top_k30,0.5"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t0.75_humanassistant.csv,top_k30,0.75"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t1.0_humanassistant.csv,top_k30,1.0"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t1.25_humanassistant.csv,top_k30,1.25"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t1.5_humanassistant.csv,top_k30,1.5"
    # top_p090
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t0.5_humanassistant.csv,top_p090,0.5"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t0.75_humanassistant.csv,top_p090,0.75"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t1.0_humanassistant.csv,top_p090,1.0"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t1.25_humanassistant.csv,top_p090,1.25"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t1.5_humanassistant.csv,top_p090,1.5"
    # top_p095
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t0.5_humanassistant.csv,top_p095,0.5"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t0.75_humanassistant.csv,top_p095,0.75"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t1.0_humanassistant.csv,top_p095,1.0"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t1.25_humanassistant.csv,top_p095,1.25"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t1.5_humanassistant.csv,top_p095,1.5"
    # typical_p090
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t0.5_humanassistant.csv,typical_p090,0.5"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t0.75_humanassistant.csv,typical_p090,0.75"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t1.0_humanassistant.csv,typical_p090,1.0"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t1.25_humanassistant.csv,typical_p090,1.25"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t1.5_humanassistant.csv,typical_p090,1.5"
    # eta_n00009
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t0.5_humanassistant.csv,eta_n00009,0.5"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t0.75_humanassistant.csv,eta_n00009,0.75"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t1.0_humanassistant.csv,eta_n00009,1.0"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t1.25_humanassistant.csv,eta_n00009,1.25"
    "kaist-ai-janus-dpo-7b_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t1.5_humanassistant.csv,eta_n00009,1.5"    
)

for line in "${file_paths_sampling_types_temps[@]}"; do
    IFS=',' read file_path sampling_type temperature <<< $line
    echo "Scoring $file_path with $sampling_type and temperature $temperature"
    # with human assistant format for reward model
    python -m src.experiments.language_model_experiments.score_sample_reward --csv-file-path $file_path --batch-size $batch_size --add-human-assistant-format --reward-model $reward_model --device $rm_device
    
    # only generated text for probability
    python -m src.experiments.language_model_experiments.score_sample_probability --csv-file-path $file_path --batch-size $batch_size --no-include-prompt --language-model $language_model --device $lm_device

    # with human assistant format for probability under generation model
    python -m src.experiments.language_model_experiments.score_sample_probability_correction --csv-file-path $file_path --batch-size $batch_size --add-human-assistant-format --sampling-type $sampling_type --language-model $rlhf_model --sampling-temperature $temperature --condition-on-prompt --device $lm_device
done