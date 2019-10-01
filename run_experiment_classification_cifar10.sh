#!/usr/bin/env bash
LOG_DIR=logs/exp_classification_hierarchical_cifar10_bandit/
mkdir -p $LOG_DIR
for i in {1..10}; do
    python run.py --environment classification --scenario hierarchical --agent bandit --iterations 10000 --log_interval 100 --log_dir $LOG_DIR --seed $i --dataset cifar10
done

LOG_DIR=logs/exp_classification_hierarchical_cifar10_random/
mkdir -p $LOG_DIR
for i in {1..10}; do
    python run.py --environment classification --scenario hierarchical --agent random --iterations 10000 --log_interval 100 --log_dir $LOG_DIR --seed $i --dataset cifar10
done

LOG_DIR=logs/exp_classification_rotation_cifar10_bandit/
mkdir -p $LOG_DIR
for i in {1..10}; do
    python run.py --environment classification --scenario rotation --agent bandit --iterations 10000 --log_interval 100 --log_dir $LOG_DIR --seed $i --dataset cifar10
done

LOG_DIR=logs/exp_classification_rotation_cifar10_random/
mkdir -p $LOG_DIR
for i in {1..10}; do
    python run.py --environment classification --scenario rotation --agent random --iterations 10000 --log_interval 100 --log_dir $LOG_DIR --seed $i --dataset cifar10
done

LOG_DIR=logs/exp_classification_shear_cifar10_bandit/
mkdir -p $LOG_DIR
for i in {1..10}; do
    python run.py --environment classification --scenario shear --agent bandit --iterations 10000 --log_interval 100 --log_dir $LOG_DIR --seed $i --dataset cifar10
done

LOG_DIR=logs/exp_classification_shear_cifar10_random/
mkdir -p $LOG_DIR
for i in {1..10}; do
    python run.py --environment classification --scenario shear --agent random --iterations 10000 --log_interval 100 --log_dir $LOG_DIR --seed $i --dataset cifar10
done
