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
)


for file_path in "${file_paths[@]}"
do
    echo "Scoring $file_path"

    # TODO
done