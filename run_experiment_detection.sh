#!/usr/bin/env bash
LOG_DIR=logs/exp_detection_hierarchical_bandit/
mkdir -p $LOG_DIR
for i in {1..10}; do
    python run.py --environment detection --scenario hierarchical --agent bandit --iterations 5000 --log_interval 50 --log_dir $LOG_DIR --seed $i
done

LOG_DIR=logs/exp_detection_hierarchical_random/
mkdir -p $LOG_DIR
for i in {1..10}; do
    python run.py --environment detection --scenario hierarchical --agent random --iterations 5000 --log_interval 50 --log_dir $LOG_DIR --seed $i
done
