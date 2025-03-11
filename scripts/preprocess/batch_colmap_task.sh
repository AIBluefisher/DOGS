#!/usr/bin/env bash

VOC_TREE_PATH=/home/yuchen/datasets/voc_tress/vocab_tree_flickr100K_words1M.bin
DATASET_DIR="/home/yuchen/datasets/Tanks"


scenes=("Ballroom" "Barn" "Church" "Family" "Francis" "Horse" "Ignatius" "Museum")

num_scenes=${#scenes[@]}
for (( i=0; i<num_scenes; i++ )); do
    scene=${scenes[i]}
    dataset_path=$DATASET_DIR/$scene
    output_path=$dataset_path/benchmark_time
    mkdir -p $output_path

    ./colmap_mapping.sh $dataset_path $output_path $VOC_TREE_PATH 100 0

done
