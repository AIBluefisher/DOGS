#!/usr/bin/env bash

DATASET_NAME=$1

DATASET_DIR="/home/yuchen/datasets/"${DATASET_NAME}
scenes=("Ballroom" "Barn" "Church" "Family" "Francis" "Horse" "Ignatius" "Museum")
# scenes=("chess" "fire" "heads" "pumpkin")
num_scenes=${#scenes[@]}

COLMAP_DIR=/usr/local/bin
COLMAP_EXE=$COLMAP_DIR/colmap

for (( i=0; i<num_scenes; i++ )); do
    scene=${scenes[i]}

    $COLMAP_EXE model_orientation_aligner \
    --image_path=${DATASET_DIR}/${scene}/images \
    --input_path=${DATASET_DIR}/${scene}/sparse/0 \
    --output_path=${DATASET_DIR}/${scene}/sparse/manhattan_world \
    > ${DATASET_DIR}/${scene}/log_align_manhattan_world.txt 2>&1

done
