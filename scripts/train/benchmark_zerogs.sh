#!/usr/bin/env bash

./train_nvs.sh 0 zero_gs llff gaussian_splatting zero_gs
../eval/eval_nvs.sh 0 zero_gs llff gaussian_splatting zero_gs

./train_nvs.sh 0 zero_gs mipnerf360 gaussian_splatting zero_gs
../eval/eval_nvs.sh 0 zero_gs mipnerf360 gaussian_splatting zero_gs

./train_nvs.sh 0 zero_gs tanks_and_temples gaussian_splatting zero_gs
../eval/eval_nvs.sh 0 zero_gs tanks_and_temples gaussian_splatting zero_gs

./train_nvs.sh 0 zero_gs seq_tnt gaussian_splatting zero_gs
../eval/eval_nvs.sh 0 zero_gs seq_tnt gaussian_splatting

./train_nvs.sh 0 zero_gs seven_scenes gaussian_splatting zero_gs
../eval/eval_nvs.sh 0 zero_gs seven_scenes gaussian_splatting zero_gs
