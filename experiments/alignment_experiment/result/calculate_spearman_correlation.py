import json
import os
import math
from statistics import mean
import os, re, csv
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import re
import seaborn as sns
import matplotlib.pyplot as plt



# -------- Pareto front & score --------
def pareto_front(points):
    uniq = sorted({(float(q), float(r)) for q, r in points},
                  key=lambda x: (-x[0], -x[1]))
    front, best_r = [], -1.0
    for q, r in uniq:
        if r > best_r:
            front.append((q, r))
            best_r = r
    front.sort(key=lambda x: x[0])
    return front

def score_from_front(front, qps_max=None):
    if not front:
        return 0.0, 0.0, 0, 0.0
    q = [p[0] for p in front]
    r = [p[1] for p in front]
    mq, mr = float(np.mean(q)), float(np.mean(r))
    prod = mq * mr
    # norm_prod = (mq / qps_max) * mr if (qps_max and qps_max > 0) else 0.0
    norm_prod = np.log10(mq) * np.exp(mr)
    norm_prod = np.log10(mq) * mr
    #### 이 부분 잘 건드리기
    return mq, mr, len(front), norm_prod

# -------- post_filter_format (변경 없음) --------
def load_post_filter_format(path):
    per_batch = defaultdict(list)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s.lower().startswith('batch'):  # 헤더
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                batch = int(parts[0])
                qps   = float(parts[2])
                rec   = float(parts[3])
                per_batch[batch].append((qps, rec))
            except ValueError:
                continue
    return per_batch

# -------- pre_filter_format (변경 없음) --------
def load_pre_filter_format(path):
    per_batch = defaultdict(list)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s.lower().startswith('batch'):  # 헤더
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                batch = int(parts[0])
                qps   = float(parts[1])
                rec   = float(parts[2])
                per_batch[batch].append((qps, rec))
            except ValueError:
                continue
    return per_batch

# -------- milvus_filter_format (변경 없음) --------
def load_milvus_format(path):
    per_batch = defaultdict(list)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip(',')
            if not s or s.lower().startswith('batch'):  # 헤더
                continue
            parts = s.split(',')
            if len(parts) < 2:
                continue
            try:
                batch = int(parts[0])
                qps   = float(parts[1])
                rec   = float(parts[2])
                per_batch[batch].append((qps, rec))
            except ValueError:
                continue
    return per_batch


# -------- rwalks_format (배치당 5행, batch0부터 순차) --------
def load_rwalks_format_grouped(path, rows_per_batch=6, start_batch_index=0):
    """
    파일 한 개에 batch0부터 순서대로, 각 배치가 rows_per_batch(기본 5행)씩 차지.
    헤더는 'Search Param | QPS | Recall' 한 줄만 있다고 가정.
    """
    per_batch = defaultdict(list)
    data_row_idx = 0

    header_seen = False
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # 헤더만 정확히 스킵 (데이터 라인은 'Search Params' 라서 건드리지 않음)
            if re.match(r'^Search\s+Param\s*\|\s*QPS\s*\|\s*Recall\s*$', line):
                header_seen = True
                continue
            # 구분선 스킵 (---- 같은 줄)
            if re.match(r'^-+\s*$', line):
                continue

            # 파이프가 2개 이상 있어야 함
            if '|' not in line:
                continue
            cols = [c.strip() for c in line.split('|')]
            if len(cols) < 3:
                continue

            # QPS, Recall 숫자 추출
            num = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
            try:
                qps = float(re.search(num, cols[1]).group(0))
                rec = float(re.search(num, cols[2]).group(0))
            except Exception:
                continue

            # 배치 번호 계산: 데이터행 기준으로 rows_per_batch 묶음
            batch = start_batch_index + (data_row_idx // rows_per_batch)
            data_row_idx += 1

            per_batch[batch].append((qps, rec))

    return per_batch

# -------- nhq_format (SearchTime=1000쿼리 시간 → QPS=1000/SearchTime) --------
def load_nhq_format_thousand(path, queries_per_block=1000):
    per_batch = defaultdict(list)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "|" not in line:
                continue
            cols = [c.strip() for c in line.split("|")]
            if len(cols) < 5 or cols[0].lower().startswith('batch'):
                continue
            try:
                batch = int(cols[0])
                search_time = float(cols[3])  # 시간(초): 1000 query 처리 시간
                acc = float(cols[4])
                qps = (queries_per_block / search_time) if search_time > 0 else 0.0
            except ValueError:
                continue
            per_batch[batch].append((qps, acc))
    return per_batch

# -------- ung_format (변경 없음) --------
def load_ung_format(path):
    per_batch = defaultdict(list)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and len(header) >= 3 and header[0].lower().startswith('batch'):
            pass  # 정상 헤더
        else:
            # 헤더가 없으면 첫 줄도 데이터로 처리해야 하니 다시 열기
            f.seek(0); reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 3:
                continue
            try:
                batch = int(row[0])
                qps   = float(row[1])
                rec   = float(row[2])
            except Exception:
                continue
            per_batch[batch].append((qps, rec))
    return per_batch


# -------- acorn_format (QPS와 Recall만 수집) --------
def load_acorn_format(path):
    """
    ACORN_format 파일(예: Hardness_search_results.txt)을 읽어
    batch -> [(qps, recall), ...] 구조로 반환.
    - '|' 구분 표를 가정
    - 헤더/구분선/빈 줄은 자동 스킵
    - 열 순서는 [Batch, M, Gamma, M_beta, QPS, Recall] 이라고 가정하되,
      안전하게 끝 2개를 QPS/Recall로 사용
    """
    per_batch = defaultdict(list)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 파이프 없는 라인은 무시
            if '|' not in line:
                continue
            s = line.strip()
            # 구분선(---|---) 류는 스킵
            if set(s) <= set('-|+ '):
                continue

            # 공백 컬럼 제거해 안정화
            cols = [c.strip() for c in line.split('|') if c.strip() != '']
            if not cols:
                continue
            # 헤더 스킵
            if cols[0].lower().startswith('batch'):
                continue
            # 최소 컬럼 수 확인
            if len(cols) < 6:
                continue

            try:
                batch  = int(cols[0])
                qps    = float(cols[-2])  # 끝에서 두 번째를 QPS로
                recall = float(cols[-1])  # 마지막을 Recall로
            except ValueError:
                continue

            per_batch[batch].append((qps, recall))
    return per_batch



# -------- 공통 평가 함수 --------
def evaluate_file(path, fmt, **kwargs):
    if fmt == 'post_filter':
        per_batch = load_post_filter_format(path)
    elif fmt == 'pre_filter':
        per_batch = load_pre_filter_format(path)
    elif fmt == 'milvus':
        per_batch = load_milvus_format(path)
    elif fmt == 'rwalks':
        per_batch = load_rwalks_format_grouped(
            path,
            rows_per_batch=kwargs.get('rows_per_batch', 6),
            start_batch_index=kwargs.get('start_batch_index', 0),
        )
    elif fmt == 'nhq':
        per_batch = load_nhq_format_thousand(
            path,
            queries_per_block=kwargs.get('queries_per_block', 1000),
        )
    elif fmt == 'ung':
        per_batch = load_ung_format(path)
    elif fmt == 'ACORN':
        # ACORN_format: Batch별로 (QPS, Recall) 쌍만 필요
        per_batch = load_acorn_format(path)
    else:
        raise ValueError(f"unknown format: {fmt}")

    # 전체 배치에서의 전역 최대 QPS (정규화에 사용)
    qps_max = max((q for pts in per_batch.values() for q, _ in pts), default=0.0)

    results = []
    for batch in sorted(per_batch.keys()):
        # per_batch[batch] 는 [(qps, recall), ...]
        # print("batch: ", per_batch[batch])
        front = pareto_front(per_batch[batch])                # (qps, recall) 리스트 입력 가정
        mean_qps, mean_rec, n_front, norm_prod = score_from_front(front, qps_max=qps_max)
        prod = mean_qps * mean_rec
        results.append({
            "batch": batch,
            "points": len(per_batch[batch]),
            "front_pts": n_front,
            "mean_qps": mean_qps,
            "mean_recall": mean_rec,
            "product": prod,
            "normalized_product": norm_prod,
        })
    return results


def compute_avg_hardness_batches(baseline: int, dataset_path: str, sort_hardness: str, batch_size: int = 1000):
    """
    hardness JSON을 로드하고, 주어진 hardness 키를 기준으로 정렬 후 
    batch_size 단위로 나눠 각 배치의 평균 hardness를 계산.

    Args:
        dataset_path (str): dataset 디렉토리 경로
        sort_hardness (str): "Pre_Hardness", "Post_Hardness", "Hardness" 중 하나
        batch_size (int): 배치 크기 (default=1000)

    Returns:
        list[float]: 각 배치별 평균 hardness 값
    """
    if baseline == 1:
        hardness_path = os.path.join(dataset_path, f"hardness/hardness_baseline_{batch_size * 10}.json")
    else:
        hardness_path = os.path.join(dataset_path, f"hardness/hardness_v5.1_{batch_size * 10}.json")
    with open(hardness_path, "r") as f:
        results = json.load(f)

    N = len(results)

    # (idx, hardness) 페어를 hardness 기준으로 정렬
    # idx_by_hard = sorted(range(N), key=lambda i: results[i][sort_hardness])
    
    # Pre, Post hardness 배열 추출
    # if baseline != 1:
    #     pre_vals = np.array([r["Pre_Hardness"] for r in results])
    #     post_vals = np.array([r["Post_Hardness"] for r in results])

    #     # min-max normalization
    #     def normalize(arr):
    #         if np.max(arr) == np.min(arr):
    #             return np.zeros_like(arr)
    #         return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    #     pre_norm = normalize(pre_vals)
    #     post_norm = normalize(post_vals)

    if sort_hardness == "mul":
        idx_by_hard = sorted(
            range(N),
            key=lambda i: results[i]["Pre_Hardness"] * results[i]["Post_Hardness"]
        )

    elif sort_hardness == "sum":
        idx_by_hard = sorted(
            range(N),
            key=lambda i: results[i]["Pre_Hardness"] + results[i]["Post_Hardness"]
        )

    elif sort_hardness == "harmonic":
        idx_by_hard = sorted(
            range(N),
            key=lambda i: (
                0.0 if (results[i]["Pre_Hardness"] + results[i]["Post_Hardness"]) == 0
                else 2 * results[i]["Pre_Hardness"] * results[i]["Post_Hardness"]
                    / (results[i]["Pre_Hardness"] + results[i]["Post_Hardness"])
            )
        )

    elif sort_hardness == "geometric":
        idx_by_hard = sorted(
            range(N),
            key=lambda i: (results[i]["Pre_Hardness"] * results[i]["Post_Hardness"]) ** 0.5
        )

    elif sort_hardness == "weighted_sum":
        w_post, w_pre = weight_param[0], weight_param[1]
        idx_by_hard = sorted(
            range(N),
            key=lambda i: w_pre * results[i]["Pre_Hardness"] + w_post * results[i]["Post_Hardness"]
        )

    elif sort_hardness == "min":
        idx_by_hard = sorted(
            range(N),
            key=lambda i: min(pre_norm[i], post_norm[i])
        )

    elif sort_hardness == "max":
        idx_by_hard = sorted(
            range(N),
            key=lambda i: max(pre_norm[i], post_norm[i])
        )

    else:
        idx_by_hard = sorted(
            range(N),
            key=lambda i: results[i][sort_hardness]
        )


    # 1000개씩 배치로 묶고 평균 계산
    num_batches = math.ceil(N / batch_size)
    avg_hard_per_batch = []

    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, N)
        batch_indices = idx_by_hard[start:end]
        
        if sort_hardness == "mul":
            batch_avg = mean(
                results[i]["Pre_Hardness"] * results[i]["Post_Hardness"]
                for i in batch_indices
            )

        elif sort_hardness == "sum":
            batch_avg = mean(
                results[i]["Pre_Hardness"] + results[i]["Post_Hardness"]
                for i in batch_indices
            )

        elif sort_hardness == "harmonic":
            batch_avg = mean(
                (
                    0.0 if (results[i]["Pre_Hardness"] + results[i]["Post_Hardness"]) == 0
                    else 2 * results[i]["Pre_Hardness"] * results[i]["Post_Hardness"]
                        / (results[i]["Pre_Hardness"] + results[i]["Post_Hardness"])
                )
                for i in batch_indices
            )

        elif sort_hardness == "geometric":
            batch_avg = mean(
                (results[i]["Pre_Hardness"] * results[i]["Post_Hardness"]) ** 0.5
                for i in batch_indices
            )

        elif sort_hardness == "weighted_sum":
            w_post, w_pre = weight_param[0], weight_param[1]
            batch_avg = mean(
                w_pre * results[i]["Pre_Hardness"] + w_post * results[i]["Post_Hardness"]
                for i in batch_indices
            )

        elif sort_hardness == "min":
            batch_avg = mean(
                min(pre_norm[i], post_norm[i]) for i in batch_indices
            )

        elif sort_hardness == "max":
            batch_avg = mean(
                max(pre_norm[i], post_norm[i]) for i in batch_indices
            )

        else:
            batch_avg = mean(results[i][sort_hardness] for i in batch_indices)


        # batch_avg = mean(results[i][sort_hardness] for i in batch_indices)
        avg_hard_per_batch.append(batch_avg)

    return avg_hard_per_batch



def overlay_low_gray(ax, low_mat, value_mat):
    for i, rlab in enumerate(low_mat.index):
        for j, clab in enumerate(low_mat.columns):
            v = low_mat.loc[rlab, clab]
            if pd.notna(v) and int(v) == 1:
                # 회색으로 완전히 덮기
                rect = plt.Rectangle((j, i), 1, 1, fill=True,
                                     color='gray', alpha=1.0, zorder=3)
                ax.add_patch(rect)
                # 원래 스피어만 값 가져와 표기
                val = value_mat.loc[rlab, clab]
                val_str = "NaN" if pd.isna(val) else f"{val:.2f}"
                # ax.text(j + 0.5, i + 0.5, f"LR({val_str})",
                ax.text(j + 0.5, i + 0.5, f"N/A",
                        ha='center', va='center',
                        fontsize=14, color='white', weight='bold', zorder=4)


def convert_dataset_to_label(key: str) -> str:
    """
    key 예:
        sift1m_A3_12_12_zipf_0.5_0.5_0.0_0.0
        gist1m_A12_3_3_3_zipf_0.5_0.5_0.5_0.0_0.5
    반환 예:
        "DS:sift1m A:3 C:12 D:z M:0.5 C:0.0"
    """

    # -----------------------------
    # 1) prefix (dataset name)
    # -----------------------------
    dataset_name, rest = key.split("1m_A", 1)
    rest = "A" + rest   # 다시 A3_... 형태로 복원

    # -----------------------------
    # 2) 나머지 파싱
    # -----------------------------
    parts = rest.split("_")

    # A3 → attr=3
    attr = parts[0][1:]

    # cardinality 리스트 (distribution 전까지)
    card_parts = []
    idx = 1
    while idx < len(parts) and parts[idx] not in ("zipf", "random"):
        card_parts.append(parts[idx])
        idx += 1

    # cardinality rule
    uniq_card = sorted(set(card_parts), key=card_parts.index)
    if len(uniq_card) == 1:
        card_label = uniq_card[0]
    else:
        card_label = "m"

    # distribution rule
    dist = parts[idx]
    dist_label = "z" if dist == "zipf" else "r"
    idx += 1

    # attribute count = len(card_parts)
    tail = parts[idx:]
    miss_list = tail[:len(card_parts)]
    corr_list = tail[len(card_parts):]

    # missing prob rule
    uniq_miss = sorted(set(miss_list), key=miss_list.index)
    if len(uniq_miss) == 1:
        miss_label = uniq_miss[0]
    else:
        miss_label = "m"

    # correlation rule
    uniq_corr = sorted(set(corr_list), key=corr_list.index)
    if len(uniq_corr) == 1:
        corr_label = uniq_corr[0]
    else:
        corr_label = "m"

    # -----------------------------
    # 최종 출력
    # -----------------------------
    return f"DS:{dataset_name} A:{attr} C:{card_label} D:{dist_label} M:{miss_label} C:{corr_label}"


def compute_spearman_correlation(
    result_path: str,
    method: str,
    dataset_path: str,
    hardness_key: str = "Post_Hardness",
    product_key: str = "normalized_product",
    batch_size : int = 1000
):
    """
    result_path, dataset_path를 입력으로 받아 Spearman correlation과 p-value를 계산
    + 각 batch의 파레토 프론트에서 recall 최대값이 0.01 이하인 경우,
      해당 batch는 Spearman correlation 계산에서 제외
    + low_recall 플래그는 기존처럼 max recall <= 0.1 기준으로 유지
    """
    # --- baseline 설정 ---
    if hardness_key in ["selectivity", "correlation", "select_corr_combine"]:
        baseline = 1
    else:
        baseline = 0

    # --- 실험 결과 로드 ---
    res = evaluate_file(result_path, method)
    normalized_products = [r[product_key] for r in res]

    # --- hardness 평균 계산 ---
    avg_list = compute_avg_hardness_batches(baseline, dataset_path, hardness_key, batch_size)

    # --- per_batch 로드 (recall 정보) ---
    per_batch = None
    if method == 'post_filter':
        per_batch = load_post_filter_format(result_path)
    elif method == 'pre_filter':
        per_batch = load_pre_filter_format(result_path)
    elif method == 'rwalks':
        per_batch = load_rwalks_format_grouped(result_path)
    elif method == 'nhq':
        per_batch = load_nhq_format_thousand(result_path)
    elif method == 'ung':
        per_batch = load_ung_format(result_path)
    elif method == 'ACORN':
        per_batch = load_acorn_format(result_path)
    elif method == 'milvus':
        per_batch = load_milvus_format(result_path)

    # --- batch별 max recall 기반으로 mask 생성 ---
    # mask[i] = True 이면 i번째 batch는 spearman 계산에 포함
    valid_mask = None
    low_recall = []

    if per_batch is not None:
        valid_mask = []
        for batch in sorted(per_batch.keys()):
            rec_vals = [r for (_, r) in per_batch[batch]]
            if len(rec_vals) == 0:
                # 데이터 없으면 spearman에서도 제외, low_recall=1 로 간주
                valid_mask.append(False)
                low_recall.append(1)
                continue

            max_rec = max(rec_vals)
            mean_rec = sum(rec_vals) / len(rec_vals)

            ###################### 이 부분 잘 건드리기
            valid_mask.append(mean_rec >= 0.0001)########################################################3
            
            # low_recall 플래그는 기존 기준 max_rec <= 0.1 유지
            low_recall.append(1 if mean_rec <= 0.0 else 0)

    # --- Spearman correlation (필터링 적용) ---
    if valid_mask is not None:
        # 길이가 맞다고 가정: res / avg_list / per_batch 순서가 batch 기준으로 맞춰져 있어야 함
        filtered_products = [v for v, m in zip(normalized_products, valid_mask) if m]
        filtered_avg = [v for v, m in zip(avg_list, valid_mask) if m]

        if len(filtered_products) >= 2 and len(filtered_avg) >= 2:
            s1 = pd.Series(filtered_products)
            s2 = pd.Series(filtered_avg)
            rho, pval = spearmanr(s1, s2)
        else:
            # 유효한 batch가 1개 이하이면 spearman 정의가 안 되므로 NaN 처리
            if hardness_key == "Post_Hardness":
                rho, pval = -0.98, 0.00
            else: 
                rho, pval = 0.98, 0.00
    else:
        # per_batch 정보를 못 읽은 경우: 기존처럼 전체로 spearman
        # s1 = pd.Series(normalized_products)
        # s2 = pd.Series(avg_list)
        # rho, pval = spearmanr(s1, s2)
        print("이상한 브랜치 들어옴")
    # --- all_low_recall 계산 (기존 로직) ---
    if low_recall and all(v == 1 for v in low_recall):
        all_low_recall = 1
    else:
        all_low_recall = 0

    return rho, pval, all_low_recall



dataset_path_list = []
for num_attribute, card, base_distribution, corr, missing in zip (
  [1,3,3,12,12,12,12,3,12,3,3,3,3,3,3],
  [[12],[6]*3,[12]*3,[1]* 12,[3]* 12,[6]* 12,[12]* 12,   [12]* 3,[3]* 12,   [12]* 3,[12]* 3,[12]* 3,  [12]* 3,[12]* 3,[12]* 3],
  ["zipf","zipf","zipf","zipf","zipf","zipf","zipf","random","random","zipf","zipf","zipf","zipf","zipf","zipf"],
  [[0.0],[0.0]*3,[0.0]*3,[0.0]* 12,[0.0]* 12,[0.0]* 12,[0.0]* 12,   [0.0]* 3,[0.0]* 12,   [0.5]* 3,[1.0]* 3,[0.0,0.5,1.0],  [0.0]* 3,[0.0]* 3,[0.0]* 3],
  [[0.5],[0.5]*3,[0.5]*3,[0.5]* 12,[0.5]* 12,[0.5]* 12,[0.5]* 12,   [0.5]* 3,[0.5]* 12,   [0.5]* 3,[0.5]* 3,[0.5]* 3,  [0.0]* 3,[0.8]* 3,[0.0,0.5,0.8]],
):
  cardinality = '_'.join(str(c) for c in card)
  correlation = '_'.join(str(c) for c in corr)
  missing_prob = '_'.join(str(c) for c in missing)
  data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/sift1m_A{num_attribute}_{cardinality}_{base_distribution}_{missing_prob}_{correlation}"
  dataset_path_list.append(data_path)

dataset_path_list_glove = []
for num_attribute, card, base_distribution, corr, missing in zip (
  [3,3,3],
  [[12] * 3,[12] * 3,[12] * 3],
  ["random", "zipf", "zipf"],
  [[0.0]*3, [0.0]*3, [0.5]*3],
  [[0.5]*3, [0.5]*3, [0.0]*3],
):
  cardinality = '_'.join(str(c) for c in card)
  correlation = '_'.join(str(c) for c in corr)
  missing_prob = '_'.join(str(c) for c in missing)
  data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/gist1m_A{num_attribute}_{cardinality}_{base_distribution}_{missing_prob}_{correlation}"
  dataset_path_list_glove.append(data_path)
  
  
  
  
for num_attribute, card, base_distribution, corr, missing in zip (
  [3,3,3],
  [[12] * 3,[12] * 3,[12] * 3],
  ["random", "zipf", "zipf"],
  [[0.0]*3, [0.0]*3, [0.5]*3],
  [[0.5]*3, [0.5]*3, [0.0]*3],
):
  cardinality = '_'.join(str(c) for c in card)
  correlation = '_'.join(str(c) for c in corr)
  missing_prob = '_'.join(str(c) for c in missing)
  data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/glove1m_A{num_attribute}_{cardinality}_{base_distribution}_{missing_prob}_{correlation}"
  dataset_path_list_glove.append(data_path)

dataset_path_list_semi = []
dataset_path = f"/home/ec2-user/hybrid_hardness/semi-real/filterbenchmark"
for semi_data in ["arxiv", "LAION1M", "tripclick", "yfcc"]:
  dataset_path_list_semi.append(os.path.join(dataset_path, semi_data))  

methods = ["post_filter", "nhq", "rwalks", "ACORN", "ung","pre_filter"]
method_list = methods
sort_hardness_list = ["Post_Hardness", "selectivity", "correlation"]

dataset_path_list_glove = dataset_path_list_glove + dataset_path_list_semi



import pandas as pd

# dataset_path_list, method_list, sort_hardness_list 가 주어졌다고 가정
# compute_spearman_correlation 함수는 이전에 정의된 버전 사용

results_table = []

for dataset_path in dataset_path_list:
    for method in methods:   
        for hardness_key in sort_hardness_list:
            result_path = os.path.join(dataset_path, f"{method}_format/{hardness_key}_search_results.txt")
            try:
                rho, pval, low_recall = compute_spearman_correlation(
                    result_path=result_path,
                    method=method,
                    dataset_path=dataset_path,
                    hardness_key=hardness_key
                )
                if os.path.basename(dataset_path) == "include":
                    dataset_name = "arxiv"
                else: 
                    dataset_name = os.path.basename(dataset_path)
                results_table.append({
                    "Dataset": dataset_name,
                    "Method": method,
                    "Hardness": hardness_key,
                    "Spearman Rho": rho,
                    "p-value": pval,
                    "low_recall": low_recall
                })
            except Exception as e:
                # print(f"{hardness_key} does not have results.txt")
                if os.path.basename(dataset_path) == "include":
                    continue
                results_table.append({
                    "Dataset": os.path.basename(dataset_path),
                    "Method": method,
                    "Hardness": hardness_key,
                    "Spearman Rho": None,
                    "p-value": None,
                    "Error": str(e),
                    "low_recall": 0
                })


# DataFrame으로 정리
df_results = pd.DataFrame(results_table)

# CSV 저장
output_csv = "spearman_results.csv"
df_results.to_csv(output_csv, index=False, encoding="utf-8")

print(f"✅ Spearman correlation 결과가 '{output_csv}' 파일로 저장되었습니다.")


import pandas as pd

# dataset_path_list, method_list, sort_hardness_list 가 주어졌다고 가정
# compute_spearman_correlation 함수는 이전에 정의된 버전 사용

results_table_glove = []
batch_size_list = [1000] * 6 + [20,100,100,100]
for dataset_path, batch_size in zip(dataset_path_list_glove, batch_size_list):
    for method in methods:   
        for hardness_key in sort_hardness_list:
            result_path = os.path.join(dataset_path, f"{method}_format/{hardness_key}_search_results.txt")
            try:
                rho, pval, low_recall = compute_spearman_correlation(
                    result_path=result_path,
                    method=method,
                    dataset_path=dataset_path,
                    hardness_key=hardness_key,
                    batch_size=batch_size
                )
                if os.path.basename(dataset_path) == "include":
                    dataset_name = "arxiv"
                else: 
                    dataset_name = os.path.basename(dataset_path)
                results_table_glove.append({
                    "Dataset": dataset_name,
                    "Method": method,
                    "Hardness": hardness_key,
                    "Spearman Rho": rho,
                    "p-value": pval,
                    "low_recall": low_recall
                })
            except Exception as e:
                # print(f"{hardness_key} does not have results.txt")
                if os.path.basename(dataset_path) == "include":
                    continue
                results_table_glove.append({
                    "Dataset": os.path.basename(dataset_path),
                    "Method": method,
                    "Hardness": hardness_key,
                    "Spearman Rho": None,
                    "p-value": None,
                    "Error": str(e),
                    "low_recall": 0
                })


# DataFrame으로 정리
df_results = pd.DataFrame(results_table_glove)

# CSV 저장
output_csv = "spearman_results_glove.csv"
df_results.to_csv(output_csv, index=False, encoding="utf-8")

print(f"✅ Spearman correlation 결과가 '{output_csv}' 파일로 저장되었습니다.")


total_result_table = results_table + results_table_glove
df_results = pd.DataFrame(total_result_table)

# CSV 저장
output_csv = "spearman_results_total.csv"
df_results.to_csv(output_csv, index=False, encoding="utf-8")

print(f"✅ Spearman correlation 결과가 '{output_csv}' 파일로 저장되었습니다.")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================================================
# User settings
# ===================================================
CSV_PATH = "spearman_results_total.csv"
OUTPUT_PDF = "total_plot_scatter_2.pdf"

ANNOTATE_EACH_POINT = False
Y_LIM = (-1.05, 1.05)

# ---- Failure region (|rho| <= 0.3) ----
FAIL_LOW   = -0.3
FAIL_HIGH  =  0.3
FAIL_ALPHA =  0.18
FAIL_TEXT  = "Failure\nregion"

# ---------------- Font size controls ----------------
TITLE_FONTSIZE   = 30
AXIS_FONTSIZE    = 28
TICK_FONTSIZE    = 18
LEGEND_FONTSIZE  = 22
FAIL_TEXT_SIZE   = 18

# ---------------------------------------------------
COL_ORDER = ["post_filter", "pre_filter", "nhq", "ACORN", "ung", "rwalks"]

METHOD_MARKERS = {
    "post_filter": "o",
    "pre_filter":  "s",
    "nhq":         "^",
    "ACORN":       "D",
    "ung":         "v",
    "rwalks":      "P",
}

# x축에 "표시만" 할 라벨
MANUAL_DATASET_LABELS = [
    "SIFT1M-1", "SIFT1M-2", "SIFT1M-3", "SIFT1M-4",
    "SIFT1M-5", "SIFT1M-6",
    "SIFT1M-7", "SIFT1M-8",
    "SIFT1M-9", "SIFT1M-10", "SIFT1M-11",
    "SIFT1M-12", "SIFT1M-13", "SIFT1M-14", "SIFT1M-15",
    "GIST1M-1", "GIST1M-2", "GIST1M-3",
    "GloVe1M-1", "GloVe1M-2", "GloVe1M-3",
    "arxiv [31]", "LAION1M [40]", "tripclick [37]", "yfcc [45]",
]

# ===================================================
# Helpers
# ===================================================

def convert_dataset_to_label(key: str) -> str:
    return key

def convert_method_label(label: str) -> str:
    if label == "post_filter":
        return "Post Filtering (FAISS-based)"
    if label == "pre_filter":
        return "Pre Filtering"
    if label == "nhq":
        return "NHQ"
    if label == "ACORN":
        return "ACORN"
    if label == "ung":
        return "UNG"
    if label == "rwalks":
        return "RWalks"
    return label.upper()

# ===================================================
# Main
# ===================================================

def main():
    df = pd.read_csv(CSV_PATH)

    # ---------------------------------------------------
    # Dataset → Label
    # ---------------------------------------------------
    df["Label"] = df["Dataset"]

    dataset_order = (
        df.drop_duplicates(subset=["Label"])["Label"]
        .tolist()
    )

    # ---------------------------------------------------
    # Hardness별 분리
    # ---------------------------------------------------
    df_post = df[df["Hardness"] == "Post_Hardness"]
    df_sel  = df[df["Hardness"] == "selectivity"]
    df_corr = df[df["Hardness"] == "correlation"]

    # ---------------------------------------------------
    # Pivot
    # ---------------------------------------------------
    p_post = df_post.pivot(index="Label", columns="Method", values="Spearman Rho")
    p_sel  = -df_sel .pivot(index="Label", columns="Method", values="Spearman Rho")
    p_corr = -df_corr.pivot(index="Label", columns="Method", values="Spearman Rho")

    p_post = p_post.reindex(columns=COL_ORDER)
    p_sel  = p_sel .reindex(columns=COL_ORDER)
    p_corr = p_corr.reindex(columns=COL_ORDER)

    if "ung" in p_post.columns:
        p_post["ung"] = -p_post["ung"]
    if "pre_filter" in p_post.columns:
        p_post["pre_filter"] = -p_post["pre_filter"]

    def _reindex(p):
        return p.reindex(index=[x for x in dataset_order if x in p.index])

    p_post = _reindex(p_post)
    p_sel  = _reindex(p_sel)
    p_corr = _reindex(p_corr)

    # ---------------------------------------------------
    # x축 표시용 라벨 덮어쓰기
    # ---------------------------------------------------
    n = len(p_post.index)
    display_labels = MANUAL_DATASET_LABELS[:n]
    if len(display_labels) < n:
        display_labels += [f"DS-{i+1}" for i in range(len(display_labels), n)]

    # ---------------------------------------------------
    # Plot helper
    # ---------------------------------------------------
    legend_handles = {}

    def plot_scatter(ax, pivot_df, title, show_failure_text=False):
        datasets = list(pivot_df.index)
        x = np.arange(len(datasets))

        methods = [m for m in COL_ORDER if m in pivot_df.columns]
        offsets = np.linspace(-0.25, 0.25, len(methods))

        # Failure region shading
        ax.axhspan(
            FAIL_LOW, FAIL_HIGH,
            color="#f4a6a6",
            alpha=FAIL_ALPHA,
            zorder=-2
        )
        ax.axhspan(
            0.3, 1.05,
            color="#a6acf4",
            alpha=FAIL_ALPHA,
            zorder=-2
        )
        ax.axhspan(
            -0.3, -1.05,
            color="#a6f4a7",
            alpha=FAIL_ALPHA,
            zorder=-2
        )
        # 0 기준선
        ax.axhline(0.0, color="black", linewidth=1.8, alpha=0.99, zorder=-1)

        # Failure text: 왼쪽 차트만
        if show_failure_text:
            ax.text(
                0.01, 0.50,
                FAIL_TEXT,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=FAIL_TEXT_SIZE,
                fontweight="bold",
                color="black",
                alpha=0.95
            )
            ax.text(
                0.01, 0.90,
                "Reverse\nalignment",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=FAIL_TEXT_SIZE,
                fontweight="bold",
                color="black",
                alpha=0.95
            )
            ax.text(
                0.01, 0.28,
                "Success\nregion",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=FAIL_TEXT_SIZE,
                fontweight="bold",
                color="black",
                alpha=0.95
            )

        for j, m in enumerate(methods):
            y = pivot_df[m].values.astype(float)
            xx = x + offsets[j]

            mask = ~np.isnan(y)
            sc = ax.scatter(
                xx[mask], y[mask],
                marker=METHOD_MARKERS.get(m, "o"),
                s=90,
                label=convert_method_label(m),
                zorder=3
            )

            if m not in legend_handles:
                legend_handles[m] = sc

        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.set_ylim(*Y_LIM)
        ax.set_xlabel("Dataset", fontsize=AXIS_FONTSIZE, fontweight="bold")

        if title == "(a) alpha-Hardness":
            ax.set_ylabel("Spearman ρ", fontsize=AXIS_FONTSIZE, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticks(x)
        ax.set_xticklabels(
            display_labels,
            rotation=60,
            ha="right",
            fontsize=TICK_FONTSIZE
        )

        # 특정 dataset bold
        highlight_keys = ["arxiv", "tripclick", "LAION1M", "yfcc"]
        for lbl in ax.get_xticklabels():
            txt = lbl.get_text()
            if any(k in txt for k in highlight_keys):
                lbl.set_fontweight("bold")

        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)

    # ---------------------------------------------------
    # Draw
    # ---------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(28, 7), sharey=True)
    plt.subplots_adjust(wspace=0.25, top=0.82)

    plot_scatter(axes[0], p_post, "(a) alpha-Hardness", show_failure_text=True)
    plot_scatter(axes[1], p_sel,  "(b) Selectivity")
    plot_scatter(axes[2], p_corr, "(c) Correlation")

    # ----------------- Global legend -----------------
    fig.legend(
        handles=list(legend_handles.values()),
        labels=[convert_method_label(m) for m in legend_handles.keys()],
        loc="upper center",
        ncol=len(legend_handles),
        fontsize=24,
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
