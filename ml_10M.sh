#!/usr/bin/env bash

export PYTHONPATH="$(pwd)"

python demo_ml_10M.py 2>&1 | tee results/ml_10M_log.txt
