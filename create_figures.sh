#!/bin/bash
set -euxo pipefail

python -m src.experiments.language_model_experiments.create_figures --input-data-dir "./src/experiments/language_model_experiments/data" --cache-dir "./src/experiments/language_model_experiments/.cache" # --overwrite-cache