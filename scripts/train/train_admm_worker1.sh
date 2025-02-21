#!/usr/bin/env bash

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16
export CUDA_LAUNCH_BLOCKING=1

# Default parameters.
NUM_GPUS_PER_NODE=4
NUM_TOTAL_NODES=9 # one fat node and N thin node.
ETHERNET_INTERFACE=eno1 # enp129s0f0

DATASET='urban3d_admm' # [urban3d]
ENCODING='gaussian_splatting'
SUFFIX=''

NUM_CMD_PARAMS=$#
if [ $NUM_CMD_PARAMS -eq 1 ]
then
    SUFFIX=$1
elif [ $NUM_CMD_PARAMS -eq 2 ]
then
    SUFFIX=$1
    DATASET=$2
fi

YAML=${ENCODING}/${DATASET}'.yaml'
echo "Using yaml file: ${YAML}"

HOME_DIR=$HOME
CODE_ROOT_DIR=$HOME/'Projects/DOGS'

cd $CODE_ROOT_DIR

TP_SOCKET_IFNAME=$ETHERNET_INTERFACE GLOO_SOCKET_IFNAME=$ETHERNET_INTERFACE torchrun \
        --nnodes=$NUM_TOTAL_NODES \
        --nproc_per_node=$NUM_GPUS_PER_NODE \
        --node_rank=1 \
        --master_addr=xx.xx.xx.xx \
        --master_port=25500 \
        -m conerf.trainers.master_gaussian_trainer \
        --config 'config/'${YAML} \
        --suffix $SUFFIX
