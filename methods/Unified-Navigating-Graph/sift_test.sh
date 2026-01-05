#!/bin/bash

BINARY="./build/apps/search_UNG_index"

BASE_BIN="/home/mintaek/hybrid_index/hybrid_benchmark/sift1m_zipf/sift_for_ung/sift_base.bin"
BASE_LABEL="/home/mintaek/hybrid_index/hybrid_benchmark/sift1m_zipf/sift_for_ung/sift_base_label.txt"
INDEX_PATH="/home/mintaek/hybrid_index/methods/Unified-Navigating-Graph/sift1m_zipf_index/"

for i in {0..9}
do
    BATCH_DIR="/home/mintaek/hybrid_index/hybrid_benchmark/sift1m_zipf/sift_for_ung/batch${i}"
    QUERY_BIN="${BATCH_DIR}/query.bin"
    QUERY_LABEL="${BATCH_DIR}/query_label.txt"
    GT_FILE="${BATCH_DIR}/gt.bin"
    RESULT_PREFIX="${BATCH_DIR}/"

    echo "Running batch${i}..."

    ${BINARY} \
        --data_type float \
        --dist_fn L2 \
        --num_threads 16 \
        --K 10 \
        --base_bin_file ${BASE_BIN} \
        --base_label_file ${BASE_LABEL} \
        --query_bin_file ${QUERY_BIN} \
        --query_label_file ${QUERY_LABEL} \
        --gt_file ${GT_FILE} \
        --index_path_prefix ${INDEX_PATH} \
        --result_path_prefix ${RESULT_PREFIX} \
        --scenario containment \
        --num_entry_points 16 \
        --Lsearch 10 50 300 500
done
