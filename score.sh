#!/bin/bash
set -euxo pipefail

file_paths=("ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt100_numgenerations10_ancestral_humanassistant.csv"
)

for file_path in "${file_paths[@]}"
    echo "Scoring $file_path"
    # with human assistant format
    python -m src.experiments.language_model_experiments.score_sample_reward --csv-file-path $file_path --batch-size 2 --add-human-assistant-format
    python -m src.experiments.language_model_experiments.score_sample_probability --csv-file-path $file_path --batch-size 2 --add-human-assistant-format

    # without prompt
    python -m src.experiments.language_model_experiments.score_sample_reward --csv-file-path $file_path --batch-size 2 --no-include-prompt
    python -m src.experiments.language_model_experiments.score_sample_probability --csv-file-path $file_path --batch-size 2 --no-include-prompt

    # without human assistant format, but prompted (default)
    python -m src.experiments.language_model_experiments.score_sample_reward --csv-file-path $file_path --batch-size 2
    python -m src.experiments.language_model_experiments.score_sample_probability --csv-file-path $file_path --batch-size 2
