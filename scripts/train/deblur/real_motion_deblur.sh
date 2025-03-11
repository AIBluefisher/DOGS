#!/usr/bin/env bash

CUDA_IDS=$1 # {'0,1,2,...'}

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_IDS}

DATASET_DIR="/home/yuchen/datasets/deblur/real_camera_motion_blur"
# SCENES=("blurball" "blurbasket" "blurbuick" "blurcoffee" "blurdecoration" "blurgirl" "blurheron" "blurparterre" "blurpuppet" "blurstair")
SCENES=("blurbasket" "blurbuick" "blurcoffee" "blurdecoration" "blurgirl" "blurheron" "blurparterre" "blurpuppet" "blurstair")
# VALS=("7" "7" "7" "6" "6" "7" "8" "6" "6" "6")
VALS=("7" "7" "6" "6" "7" "8" "6" "6" "6")
NUM_SCENES=${#SCENES[@]}


# Default parameters.
DATASET='real_motion_blur'
MODEL_FOLDER='sparse'
INIT_PLY_TYPE='sparse'
# Parameters can be specified from command line.
ENCODING='gaussian_splatting'
SUFFIX=''

NUM_CMD_PARAMS=$#
if [ $NUM_CMD_PARAMS -ge 2 ]
then
    SUFFIX=$2
fi

if [ $NUM_CMD_PARAMS -ge 3 ]
then
    ENCODING=$3
fi

YAML=${ENCODING}/${DATASET}'.yaml'
echo "Using yaml file: ${YAML}"

HOME_DIR=$HOME
CODE_ROOT_DIR=$HOME/'Projects/DOGS'

cd $CODE_ROOT_DIR

for (( i=0; i<NUM_SCENES; i++ )); do
    scene=${SCENES[i]}
    val=${VALS[i]}

    python train.py --config 'config/'${YAML} \
                    --suffix $SUFFIX \
                    --model_folder $MODEL_FOLDER \
                    --init_ply_type $INIT_PLY_TYPE \
                    --load_specified_images \
                    --scene ${scene} \
                    --val ${val}

done
