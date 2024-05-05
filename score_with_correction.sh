#!/bin/bash
set -euxo pipefail

file_paths=(
"ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_t1.0_humanassistant.csv"
"ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_t1.5_humanassistant.csv"
"ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_ancestral_t2.0_humanassistant.csv"
"ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t1.0_humanassistant.csv"
"ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t1.5_humanassistant.csv"
"ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t2.0_humanassistant.csv"
"ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t1.0_humanassistant.csv"
"ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t1.5_humanassistant.csv"
"ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p095_t2.0_humanassistant.csv"
"ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t1.0_humanassistant.csv"
"ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t1.5_humanassistant.csv"
"ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_typical_p090_t2.0_humanassistant.csv"
)
sampling_types=(
    "ancestral"
    "ancestral"
    "ancestral"
    "top_p090"
    "top_p090"
    "top_p090"
    "top_p095"
    "top_p095"
    "top_p095"
    "typical_p090"
    "typical_p090"
    "typical_p090"
)
temperatures=(
    "1.0"
    "1.5"
    "2.0"
    "1.0"
    "1.5"
    "2.0"
    "1.0"
    "1.5"
    "2.0"
    "1.0"
    "1.5"
    "2.0"
)


for i in "${!file_paths[@]}"
do

    file_path=${file_paths[$i]}
    sampling_type=${sampling_types[$i]}
    temperature=${temperatures[$i]}

    echo "Scoring $file_path"
    # with human assistant format for reward model
    # python -m src.experiments.language_model_experiments.score_sample_reward --csv-file-path $file_path --batch-size 16 --add-human-assistant-format
    
    # only generated text for probability
    # python -m src.experiments.language_model_experiments.score_sample_probability --csv-file-path $file_path --batch-size 16 --no-include-prompt
    python -m src.experiments.language_model_experiments.score_sample_probability --csv-file-path $file_path --batch-size 16 --add-human-assistant-format  --condition-on-prompt


    # with human assistant format for probability under generation model (for correction)
    python -m src.experiments.language_model_experiments.score_sample_probability_correction --csv-file-path $file_path --batch-size 16 --add-human-assistant-format --sampling-type $sampling_type --language-model "ethz-spylab/rlhf-7b-harmless" --sampling-temperature $temperature --condition-on-prompt

done