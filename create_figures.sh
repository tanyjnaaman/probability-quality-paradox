#!/bin/bash
set -euxo pipefail

# rlhf model
python -m src.experiments.language_model_experiments.create_figures --input-data-dir "./src/experiments/language_model_experiments/data" --cache-dir "./src/experiments/language_model_experiments/.cache" --figure-name "correlations" --overwrite-cache

# dpo model
python -m src.experiments.language_model_experiments.create_figures --input-data-dir "./src/experiments/language_model_experiments/data_dpo" --cache-dir "./src/experiments/language_model_experiments/.cache_dpo_secret" --figure-name "correlations_dpo_secret" --use-secret-reward --overwrite-cache