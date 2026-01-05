#!/bin/bash

# 고정된 공통 입력값
INDEX_TYPE="NHQ-NPG_nsw"
DATA_VEC="/home/mintaek/hybrid_index/hybrid_benchmark/sift1m_zipf/sift_for_NHQ/sift_base.fvecs"
DATA_LABEL="/home/mintaek/hybrid_index/hybrid_benchmark/sift1m_zipf/sift_for_NHQ/base_labels.txt"
BASE_OUTPUT_DIR="/home/mintaek/hybrid_index/methods/NHQ/sift1m_zipf"

# (MaxM0, efConstruction) 값 목록
declare -a configs=(
  "10 30"
  "20 50"
  "30 70"
  "40 100"
  "50 150"
)

# 반복 실행
for config in "${configs[@]}"; do
  read maxm0 efc <<< "$config"
  subdir="M${maxm0}_ef${efc}"
  output_dir="${BASE_OUTPUT_DIR}/${subdir}"
  mkdir -p "$output_dir"

  echo "Building index with MaxM0=$maxm0, efConstruction=$efc → $output_dir"

  python test_hybrid_query.py build <<EOF
$INDEX_TYPE
$DATA_VEC
$DATA_LABEL
${output_dir}/index.bin
${output_dir}/index.txt
$maxm0
$efc
EOF

done
