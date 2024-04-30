#!/bin/bash
set -euxo pipefail

python -m src.experiments.language_model_experiments.score_sample_reward --csv-file-path "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt500_numgenerations100_ancestral.csv" --batch-size 16

python -m src.experiments.language_model_experiments.score_sample_probability --csv-file-path "ethz-spylab-rlhf-7b-harmless_l256_promptseed42_numprompt500_numgenerations100_ancestral_scoredreward.csv" --batch-size 16