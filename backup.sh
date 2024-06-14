#!/bin/bash
set -euxo pipefail

zip -r backup.zip backup
zip -r data.zip src/experiments/language_model_experiments/data
zip -r .cache.zip src/experiments/language_model_experiments/.cache

zip -r data_dpo.zip src/experiments/language_model_experiments/data_dpo
zip -r .cache_dpo.zip src/experiments/language_model_experiments/.cache_dpo