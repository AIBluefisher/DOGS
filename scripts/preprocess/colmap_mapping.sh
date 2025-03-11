#!/usr/bin/env bash

DATASET_PATH=$1
OUTPUT_PATH=$2
VOC_TREE_PATH=$3
MOST_SIMILAR_IMAGES_NUM=$4
CUDA_IDS=$5

NUM_THREADS=24
# export PYTHONDONTWRITEBYTECODE=1
# export CUDA_VISIBLE_DEVICES=${CUDA_IDS}

COLMAP_DIR=/usr/local/bin
COLMAP_EXE=$COLMAP_DIR/colmap

mkdir $OUTPUT_PATH/sparse
mkdir $OUTPUT_PATH/sparse/manhattan_world

begin_time=$(date "+%s")

$COLMAP_EXE feature_extractor \
    --database_path=$OUTPUT_PATH/database.db \
    --image_path=$DATASET_PATH/images \
    --SiftExtraction.num_threads=$NUM_THREADS \
    --SiftExtraction.use_gpu=1 \
    --SiftExtraction.gpu_index=$CUDA_IDS \
    --SiftExtraction.estimate_affine_shape=true \
    --SiftExtraction.domain_size_pooling=true \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --SiftExtraction.max_num_features 8192 \
    > $OUTPUT_PATH/log_extract_feature.txt 2>&1

end_time=$(date "+%s")
feature_extraction_time=$(($end_time - $begin_time))
echo "feature extraction time: $feature_extraction_time s" \
    > $OUTPUT_PATH/feature_extraction_time.txt

begin_time=$(date "+%s")

$COLMAP_EXE vocab_tree_matcher \
    --database_path=$OUTPUT_PATH/database.db \
    --SiftMatching.num_threads=$NUM_THREADS \
    --SiftMatching.use_gpu=1 \
    --SiftMatching.gpu_index=$CUDA_IDS \
    --SiftMatching.guided_matching=false \
    --VocabTreeMatching.num_images=$MOST_SIMILAR_IMAGES_NUM \
    --VocabTreeMatching.num_nearest_neighbors=5 \
    --VocabTreeMatching.vocab_tree_path=$VOC_TREE_PATH \
    > $OUTPUT_PATH/log_match.txt 2>&1

end_time=$(date "+%s")
feature_matching_time=$(($end_time - $begin_time))
echo "feature extraction time: $feature_matching_time s" \
    > $OUTPUT_PATH/feature_matching_time.txt

begin_time=$(date "+%s")

$COLMAP_EXE mapper $OUTPUT_PATH \
    --database_path=$OUTPUT_PATH/database.db \
    --image_path=$DATASET_PATH/images \
    --output_path=$OUTPUT_PATH/sparse \
    --Mapper.num_threads=$NUM_THREADS \
    > $OUTPUT_PATH/log_sfm.txt 2>&1

end_time=$(date "+%s")
sfm_mapping_time=$(($end_time - $begin_time))
echo "feature extraction time: $sfm_mapping_time s" \
    > $OUTPUT_PATH/sfm_mapping_time.txt

$COLMAP_EXE model_orientation_aligner \
    --image_path=$DATASET_PATH/images \
    --input_path=$OUTPUT_PATH/sparse/0 \
    --output_path=$OUTPUT_PATH/sparse/manhattan_world \
    > $OUTPUT_PATH/log_align_manhattan_world.txt 2>&1
