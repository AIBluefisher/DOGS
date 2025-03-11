#!/usr/bin/env bash

./train_nvs.sh 0 colmap_gs llff gaussian_splatting
../eval/eval_nvs.sh 0 colmap_gs llff gaussian_splatting

./train_nvs.sh 0 colmap_gs mipnerf360 gaussian_splatting
../eval/eval_nvs.sh 0 colmap_gs mipnerf360 gaussian_splatting

./train_nvs.sh 0 colmap_gs tanks_and_temples gaussian_splatting
../eval/eval_nvs.sh 0 colmap_gs tanks_and_temples gaussian_splatting

./train_nvs.sh 0 colmap_gs seq_tnt gaussian_splatting
../eval/eval_nvs.sh 0 colmap_gs seq_tnt gaussian_splatting

./train_nvs.sh 0 colmap_gs seven_scenes 
../eval/eval_nvs.sh 0 colmap_gs seven_scenes gaussian_splatting
