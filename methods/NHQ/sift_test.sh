#!/bin/bash

# 공통 경로 설정
BASE_BATCH_DIR="/home/mintaek/hybrid_index/hybrid_benchmark/sift1m_zipf/sift_for_NHQ"
BASE_INDEX_DIR="/home/mintaek/hybrid_index/methods/NHQ/sift1m_zipf"
OUTPUT_FILE="search_results_sift_zipf.txt"

# 결과 파일 초기화
echo "Batch | M | ef | SearchTime | Accuracy" > "$OUTPUT_FILE"

# (M, efConstruction) 조합
declare -a configs=(
  "10 30"
  "20 50"
  "30 70"
  "40 100"
  "50 150"
)

# batch0 ~ batch9 루프
for BATCH_ID in {0..9}; do
  BATCH_DIR="${BASE_BATCH_DIR}/batch${BATCH_ID}"

  for config in "${configs[@]}"; do
    read maxm0 efc <<< "$config"
    INDEX_DIR="${BASE_INDEX_DIR}/M${maxm0}_ef${efc}"
    INDEX_BIN="${INDEX_DIR}/index.bin"
    INDEX_TXT="${INDEX_DIR}/index.txt"

    echo "▶ Running: Batch $BATCH_ID | M=$maxm0 ef=$efc"

    # 실행 및 로그 저장 (stderr 포함)
    python test_hybrid_query.py search <<EOF > tmp_log.txt 2>&1
NHQ-NPG_nsw
$INDEX_BIN
$INDEX_TXT
$BATCH_DIR/sift_query.fvecs
$BATCH_DIR/sift_query_label.txt
$BATCH_DIR/gt.ivecs
EOF

    # 결과 파싱 (정규표현식 기반)
    line=$(grep "Search Time:" tmp_log.txt)

    search_time=$(echo "$line" | grep -oP 'Search Time: \K[0-9.]+')
    accuracy=$(echo "$line" | grep -oP 'accuracy: \K[0-9.]+')

    # 결과 기록
    echo "$BATCH_ID | $maxm0 | $efc | $search_time | $accuracy" >> "$OUTPUT_FILE"
  done
done

# 정리
rm tmp_log.txt
echo "✅ All results saved to $OUTPUT_FILE"
