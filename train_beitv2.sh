#!/usr/bin/env bash

GPU=$1

python train_kaggle_style.py -g $GPU -c configs/beitv2.yaml ${@:2}