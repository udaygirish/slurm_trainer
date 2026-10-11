#!/bin/bash
mkdir -p lib helpers
touch requirements.txt README.md train.py eval.py slurm_submit.sh
mkdir -p lib/models lib/data
touch lib/models/__init__.py lib/models/mnist_model.py
touch lib/data/__init__.py lib/data/dataset.py
touch helpers/__init__.py helpers/training_utils.py helpers/eval_utils.py 