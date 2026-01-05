import json, os
import numpy as np
import struct
from pathlib import Path
from tqdm import tqdm

# 먼저 환경변수로 전역 런타임 제한
os.environ["OMP_NUM_THREADS"] = "24"        # OpenMP (FAISS 포함)
os.environ["OPENBLAS_NUM_THREADS"] = "1"    # OpenBLAS는 1 추천
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

import faiss
faiss.omp_set_num_threads(8)

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "methods", "ACORN"))

from ACORN_build import (
    build_and_run_acorn_from_python,
    run_acorn_search_from_python,
    build_bit_map_from_python,
)

def find_recall(k, knn_path, gt_path, nv):
    counts = []

    sum_topk = 0
    with open(knn_path, "r") as f_knn, open(gt_path, "r") as f_gt:
        for i, (knn_line, gt_line) in enumerate(zip(f_knn, f_gt)):
            # 공백 라인 안전 처리
            if not knn_line.strip() or not gt_line.strip():
                counts.append(0)
                continue

            # KNN은 membership 조회가 많으니 set으로
            knn_ids = set(map(int, knn_line.strip().split()))
            # GT는 순서가 필요하니 list로 -> 앞 k개를 취함
            gt_ids  = list(map(int, gt_line.strip().split()))
            topk_gt = []
            
            for i in range(k):
                if len(gt_ids) <= i:
                    break

                if gt_ids[i] <= nv:
                    topk_gt.append(gt_ids[i])
                else:
                    break

            sum_topk += len(topk_gt)

            match_count = sum(1 for x in topk_gt if x in knn_ids)
            counts.append(match_count)

    if not counts:
        return 0.0

    avg_hits = sum(counts) / len(counts)
    avg_topk = sum_topk / len(counts)

    if avg_topk == 0:
        recall_at_k = 1
    else:
        recall_at_k = avg_hits / avg_topk  # 평균적으로 k개 중 몇 개가 맞았는지 (비율)
    
    return recall_at_k


ACORN_trade_off_semi = {}

for dataset_name in ["arxiv", "LAION1M", "tripclick", "yfcc"]: #"arxiv", "LAION1M", "tripclick", "yfcc", "sift_high","sift_low","gist_high","gist_low", "sift1m_RWalks", "sift1m_ACORN", "sift1m_NHQ", "sift1m_UNG"
    if dataset_name in ["arxiv", "LAION1M", "tripclick", "yfcc"]:
        data_path = f"/home/ec2-user/hybrid_hardness/semi-real/filterbenchmark/{dataset_name}"
    else:
        data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset_name}"


    ACORN_trade_off_semi[dataset_name] = {}

    k = 10

    params_list = [
        (32, 12, 64),
        (40, 15, 80),
        (48, 18, 96),
        (56, 21, 112),
        (64, 24, 128),
        (80, 30, 160),
        (96, 36, 192),
        (128, 48, 256)
    ]
    faiss_include_dirs = [
        "/home/ec2-user/hybrid_hardness/methods/ACORN",  # 이 경로 아래에 faiss/IndexACORN.h 가 있어야 함
    ]
    faiss_lib_dirs = [
        "/home/ec2-user/hybrid_hardness/methods/ACORN/build/faiss",
    ]

    fname = f"{data_path}/mid_format/db_vectors_ACORN.fvec"
    assert os.path.exists(fname), f"입력 파일 없음: {fname}"

    # fvec 헤더에서 dimension 읽기
    with open(fname, "rb") as f:
        d_from_file = np.fromfile(f, dtype=np.int32, count=1)[0]
    print("d_from_file =", d_from_file)
    if(dataset_name in ["sift_high","sift_low","gist_high","gist_low"]):
        npy_path = f"{data_path}/vectors.npy"
    else:
        npy_path = f"{data_path}/hardness_format/vectors.npy"

    data = np.load(npy_path)
    nv = data.shape[0]

    for M, gamma, M_beta in params_list:
        idx_path = f"{data_path}/ACORN_format/ACORN_index_{M}_{gamma}_{M_beta}.faiss"
        qps_batch = run_acorn_search_from_python(
            d=int(d_from_file), M=M, gamma=gamma, M_beta=M_beta, k=k,
            faiss_index_path=idx_path,
            db_vectors_path=f"{data_path}/mid_format/db_vectors_ACORN.fvec",
            query_vectors_path=f"{data_path}/mid_format/query_vectors_ACORN.fvec",
            db_filters_path=f"{data_path}/mid_format/db_filter_ACORN.txt",
            query_filters_path=f"{data_path}/mid_format/query_filter_ACORN.txt",
            out_I_path=f"{data_path}/ACORN_format/knn_I.txt",
            bit_map=f"{data_path}/ACORN_format/bit_map.txt",
            include_dirs=faiss_include_dirs,
            lib_dirs=faiss_lib_dirs
            #libs=["faiss"],   # GPU 빌드라면 CUDA 관련 libs 추가
        )
        recall = find_recall(k, f"{data_path}/ACORN_format/knn_I.txt", f"{data_path}/mid_format/query_gt_ACORN.txt", nv)
        ACORN_trade_off_semi[dataset_name][(M, gamma, M_beta)] = {
            "qps": qps_batch,
            "avg_recall": recall
        }

import pickle
with open(os.path.join(".", "ACORN_trade_off_semi_result.pkl"), "wb") as f:
    pickle.dump(ACORN_trade_off_semi, f)

print(ACORN_trade_off_semi)