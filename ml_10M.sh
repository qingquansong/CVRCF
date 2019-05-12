#!/usr/bin/env bash

export PYTHONPATH="$(pwd)"

python demo_ml_10M.py 2>&1 | tee logs/0427/A6_alpha_0.1_sym_0.555.log
