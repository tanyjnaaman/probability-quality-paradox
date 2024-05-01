#!/bin/bash
set -euxo pipefail

file_paths=(
"ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t1.0_humanassistant.csv"
"ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t1.5_humanassistant.csv"
"ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt1000_numgenerations2_top_p090_t2.0_humanassistant.csv"
)

for file_path in "${file_paths[@]}"
do
    echo "Scoring $file_path"

    # with human assistant format
    python -m src.experiments.language_model_experiments.score_sample_reward --csv-file-path $file_path --batch-size 16 --add-human-assistant-format
    python -m src.experiments.language_model_experiments.score_sample_probability --csv-file-path $file_path --batch-size 16 --add-human-assistant-format

    # without prompt
    python -m src.experiments.language_model_experiments.score_sample_reward --csv-file-path $file_path --batch-size 16 --no-include-prompt
    python -m src.experiments.language_model_experiments.score_sample_probability --csv-file-path $file_path --batch-size 16 --no-include-prompt

    # without human assistant format, but prompted (default)
    python -m src.experiments.language_model_experiments.score_sample_reward --csv-file-path $file_path --batch-size 16
    python -m src.experiments.language_model_experiments.score_sample_probability --csv-file-path $file_path --batch-size 16
done