#!/usr/bin/env bash

export PYTHONDONTWRITEBYTECODE=1

HOME_DIR=$HOME
CODE_ROOT_DIR=$HOME/'Projects/DOGS'

cd $CODE_ROOT_DIR

DATASET_DIR="/home/yuchen/datasets/llff"
scenes=("horns" "room" "trex")
# scenes=("bicycle" "counter" "garden" "kitchen")
# scenes=("Family" "Francis" "Ignatius" "Train")
# scenes=("Ballroom" "Barn" "Church" "Family" "Francis" "Horse" "Ignatius" "Museum")
# scenes=("chess" "fire" "heads" "pumpkin")
num_scenes=${#scenes[@]}

for (( i=0; i<num_scenes; i++ )); do
    scene=${scenes[i]}
    
    # mkdir $DATASET_DIR/$scene/sparse/manhattan_world
    # colmap model_orientation_aligner \
    #     --image_path=$DATASET_DIR/$scene/images \
    #     --input_path=$DATASET_DIR/$scene/sparse/0 \
    #     --output_path=$DATASET_DIR/$scene/sparse/manhattan_world

    python -m scripts.preprocess.align_to_colmap_coord_frame \
        --data_root_dir ${DATASET_DIR} \
        --scene ${scene} \
        --method zero_gs

done
