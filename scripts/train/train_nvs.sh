#!/usr/bin/env bash

CUDA_IDS=$1 # {'0,1,2,...'}

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_IDS}

# Default parameters.
DATASET='blender'                # [blender, dtu, tat]
ENCODING='gaussian_splatting'    # [gaussian_splatting]
SUFFIX=''
MODEL_FOLDER='sparse'
INIT_PLY_TYPE='sparse'

NUM_CMD_PARAMS=$#
if [ $NUM_CMD_PARAMS -ge 2 ]
then
    SUFFIX=$2
fi

if [ $NUM_CMD_PARAMS -ge 3 ]
then
    DATASET=$3
fi

if [ $NUM_CMD_PARAMS -ge 4 ]
then
    ENCODING=$4
fi

if [ $NUM_CMD_PARAMS -ge 5 ]
then
    MODEL_FOLDER=$5
fi

if [ $NUM_CMD_PARAMS -ge 6 ]
then
    INIT_PLY_TYPE=$6
fi

YAML=${ENCODING}/${DATASET}'.yaml'
echo "Using yaml file: ${YAML}"

HOME_DIR=$HOME
CODE_ROOT_DIR=$HOME/'Projects/DOGS'

cd $CODE_ROOT_DIR

python train.py --config 'config/'${YAML} \
                --suffix $SUFFIX \
                --model_folder $MODEL_FOLDER \
                --init_ply_type $INIT_PLY_TYPE # \
                # --load_specified_images
