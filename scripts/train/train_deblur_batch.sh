#!/usr/bin/env bash

./train_nvs.sh 0 deblur_3dgs synthetic_motion_blur deblur_3dgs
./train_nvs.sh 0 gs synthetic_motion_blur gaussian_splatting
./train_nvs.sh 0 deblur_bags synthetic_motion_blur bags


./deblur/real_motion_deblur.sh 0 deblur_3dgs deblur_3dgs
./deblur/real_motion_deblur.sh 0 gs gaussian_splatting
./train_nvs.sh 0 deblur_bags real_motion_blur bags
