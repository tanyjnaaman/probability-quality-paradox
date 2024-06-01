#!/bin/bash
set -euxo pipefail

zip -r backup.zip backup
zip -r data.zip src/experiments/language_model_experiments/data