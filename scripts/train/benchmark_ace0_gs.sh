#!/usr/bin/env bash

# NOTE(chenyu): manually modify the code at line252-254 in `DOGS/conerf/datasets/load_colmap.py` 
# for which we have to specify the 'factor=1' since ACE0 already downsampled the image during 
# training.

./train_nvs.sh 0 ace0_gs llff gaussian_splatting acezero
../eval/eval_nvs.sh 0 ace0_gs llff gaussian_splatting acezero

./train_nvs.sh 0 ace0_gs mipnerf360 gaussian_splatting acezero
../eval/eval_nvs.sh 0 ace0_gs mipnerf360 gaussian_splatting acezero

./train_nvs.sh 0 ace0_gs tanks_and_temples gaussian_splatting acezero
../eval/eval_nvs.sh 0 ace0_gs tanks_and_temples gaussian_splatting acezero

./train_nvs.sh 0 ace0_gs seq_tnt gaussian_splatting acezero
../eval/eval_nvs.sh 0 ace0_gs seq_tnt gaussian_splatting

./train_nvs.sh 0 ace0_gs seven_scenes gaussian_splatting acezero
../eval/eval_nvs.sh 0 ace0_gs seven_scenes gaussian_splatting
