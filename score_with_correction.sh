#!/bin/bash
set -euxo pipefail

file_paths=(
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t0.5_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t0.75_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t1.0_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t1.0_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t1.5_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_strict_t2.0_humanassistant.csv"

    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_t0.5_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_t0.75_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_t1.0_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_t1.25_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_t1.5_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_t2.0_humanassistant.csv"

    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t0.5_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t0.75_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t1.0_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t1.25_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t1.5_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_eta_n00009_t2.0_humanassistant.csv"

    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t0.5_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t0.75_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t1.0_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t1.25_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t1.5_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_k30_t2.0_humanassistant.csv"

    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t0.5_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t0.75_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t1.0_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t1.25_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t1.5_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t2.0_humanassistant.csv"

    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t0.5_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t0.75_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t1.0_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t1.25_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t1.5_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t2.0_humanassistant.csv"

    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t0.5_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t0.74_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t1.0_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t1.25_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t1.5_humanassistant.csv"
    # "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t2.0_humanassistant.csv"
)
sampling_types=(
    "ancestral_strict"
    "ancestral_strict"
    "ancestral_strict"
    "ancestral_strict"
    "ancestral_strict"
    "ancestral_strict"  

    "ancestral"
    "ancestral"
    "ancestral"
    "ancestral"
    "ancestral"
    "ancestral"

    "eta_n00009"
    "eta_n00009"
    "eta_n00009"
    "eta_n00009"
    "eta_n00009"
    "eta_n00009"

    "top_k30"
    "top_k30"
    "top_k30"
    "top_k30"
    "top_k30"
    "top_k30"

    "top_p090"
    "top_p090"
    "top_p090"
    "top_p090"
    "top_p090"
    "top_p090"

    "top_p095"
    "top_p095"
    "top_p095"
    "top_p095"
    "top_p095"
    "top_p095"

    "typical_p090"
    "typical_p090"
    "typical_p090"
    "typical_p090"
    "typical_p090"
    "typical_p090"
)
temperatures=(
    "0.5"
    "0.75"
    "1.0"
    "1.25"
    "1.5"
    "2.0"

    "0.5"
    "0.75"
    "1.0"
    "1.25"
    "1.5"
    "2.0"

    "0.5"
    "0.75"
    "1.0"
    "1.25"
    "1.5"
    "2.0"

    "0.5"
    "0.75"
    "1.0"
    "1.25"
    "1.5"
    "2.0"

    "0.5"
    "0.75"
    "1.0"
    "1.25"
    "1.5"
    "2.0"

    "0.5"
    "0.75"
    "1.0"
    "1.25"
    "1.5"
    "2.0"

    "0.5"
    "0.75"
    "1.0"
    "1.25"
    "1.5"
    "2.0"
)
batch_size=2

for i in "${!file_paths[@]}"
do

    file_path=${file_paths[$i]}
    sampling_type=${sampling_types[$i]}
    temperature=${temperatures[$i]}

    echo "Scoring $file_path"
    # with human assistant format for reward model
    python -m src.experiments.language_model_experiments.score_sample_reward --csv-file-path $file_path --batch-size $batch_size --add-human-assistant-format
    
    # only generated text for probability
    python -m src.experiments.language_model_experiments.score_sample_probability --csv-file-path $file_path --batch-size $batch_size --no-include-prompt
    python -m src.experiments.language_model_experiments.score_sample_probability --csv-file-path $file_path --batch-size $batch_size --add-human-assistant-format  --condition-on-prompt


    # with human assistant format for probability under generation model (for correction)
    python -m src.experiments.language_model_experiments.score_sample_probability_correction --csv-file-path $file_path --batch-size $batch_size --add-human-assistant-format --sampling-type $sampling_type --language-model "ethz-spylab/rlhf-7b-harmless" --sampling-temperature $temperature --condition-on-prompt

done