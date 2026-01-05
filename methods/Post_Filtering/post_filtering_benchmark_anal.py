# 필요한 함수들 정의
from tqdm import tqdm
import time
import numpy as np
import random
import hnswlib
from tqdm import tqdm
import time
import json
import os 
import matplotlib.pyplot as plt
# 데이터셋 불러오는 블록
####################################################################
dataset_name ="sift1m"
# dataset_name ="HnM"
# dataset_name ="ArXiv"

# num_attribute = 3
# cardinality = [6] * num_attribute
# distribution = "random"
# distribution = "zipf"

# sort_hardness = "Hardness"
# sort_hardness = "Pre_Hardness"
# sort_hardness = "Post_Hardness"


# sort_hardness = "selectivity"
# sort_hardness = "correlation"
# sort_hardness = "select_corr_combine"
##################################################################################################

def satisfies_conditions(payload, conditions):
    if "and" in conditions:
        for cond in conditions["and"]:
            # cond는 단일 dict: {label: {match: {value: ...}}}
            if not isinstance(cond, dict):
                continue
            for key, rule in cond.items():
                if "match" in rule and "value" in rule["match"]:
                    if payload.get(key) != rule["match"]["value"]:
                        return False
                else:
                    # 지원하지 않는 조건
                    return False
        return True
    else:
        # "and" 키가 없으면 조건 없음 → 항상 True
        return True

def post_filtering(index, K, tests, payloads, K_n, space):
    """
    index: hnswlib index
    K: top-K
    tests: batch 단위 리스트 (길이: 1000)
    payloads: 전체 데이터셋 메타데이터 리스트 (len = index에 들어간 벡터 개수)
    ---
    return: 각 쿼리별 post-filtering 결과 (리스트 of id 리스트)
    """
    results = []
    # batch knn-query (각 쿼리 vector를 모아서 한 번에 처리)
    queries = [t['query'] for t in tests]
    # 후보 pool을 넉넉히 잡자 (예: K*10)
    labels, dists = index.knn_query(queries, k=K*K_n)
    for i, test in enumerate(tests):
        # print(test["conditions"])
        conditions = test['conditions']
        filtered = []
        for idx in labels[i]:
            if satisfies_conditions(payloads[idx], conditions):
                filtered.append(idx)
            if len(filtered) == K:
                break
        results.append(filtered)
    return results


def recall_at_k(retrieved, gt, k):
    if not gt:
        return 0.0
    return len(set(retrieved[:k]) & set(gt)) / min(len(gt), k)


trade_off = {}
for num_attribute in [1, 3, 12]:
    for distribution in ["zipf"]:
        for card in [1, 3, 12]:
            if num_attribute == 1 and card == 1:
                continue
            if num_attribute == 1 and card == 3:
                continue
            if num_attribute == 3 and card == 1:
                continue
            cardinality = [card] * num_attribute

            if dataset_name == "sift1m":
                cardi = '_'.join(str(c) for c in cardinality)
                dataset_path = f"/home/mintaek/hybrid_index/Benchmark/{dataset_name}_A{num_attribute}_{cardi}_{distribution}"
            elif dataset_name == "HnM":
                dataset_path = f"/home/mintaek/hybrid_index/Benchmark/{dataset_name}"

            elif dataset_name == "ArXiv":
                dataset_path = f"/home/mintaek/hybrid_index/Benchmark/ArXiv/medium/include"

            DATA_DIR = os.path.join(dataset_path, "hardness_format")


            vectors_file = f"{DATA_DIR}/vectors.npy"
            # print("vector file path", vectors_file)
            payloads_file = f"{DATA_DIR}/payloads.jsonl"
            tests_file = f"{DATA_DIR}/tests.jsonl"

            # ------------------------------------
            # 1. Load vectors.npy
            # ------------------------------------
            vectors = np.load(vectors_file)
            print("vectors.shape =", vectors.shape)

            # ------------------------------------
            # 2. Load payloads.jsonl
            # ------------------------------------
            payloads = []
            with open(payloads_file, "r") as f:
                for line in f:
                    payloads.append(json.loads(line))

            print(f"Loaded {len(payloads)} payloads")

            # ------------------------------------
            # 3. Load tests.jsonl
            # ------------------------------------
            tests = []
            with open(tests_file, "r") as f:
                for line in f:
                    tests.append(json.loads(line))

            print(f"Loaded {len(tests)} tests")

            # # ------------------------------------
            # # 예시 출력
            # print("\nSample payload:", payloads[0])
            # print("\nSample test:", tests[0])




            # 1. Load hardness and GT
            space = "l2"
            # space = "cosine"

            ## post filtering 측정하기
            index = hnswlib.Index(space=space, dim=len(vectors[0]))
            index.init_index(max_elements=len(vectors), ef_construction=50, M=8)
            index.add_items(vectors, num_threads=32)

            if dataset_name == "HnM":
                K = 25  # top-K
            else:
                K = 10
            trade_off[(num_attribute, card, distribution)] = {}
            for K_n in tqdm([5, 10, 20, 50, 100]):
                t0 = time.time()
                results = post_filtering(index, K, tests, payloads, K_n, space)
                t1 = time.time()
                elapsed = t1 - t0
                recalls = []
                for i, test in enumerate(tests):
                    gt_ids = test['closest_ids']
                    retrieved_ids = results[i]

                    # gt에서 4294967295 제거
                    valid_gt_ids = [gt for gt in gt_ids if gt != 4294967295]

                    # valid_gt가 없으면 recall 정의 불가 → 0으로 처리
                    if len(valid_gt_ids) == 0:
                        continue
                    else:
                        recalls.append(recall_at_k(retrieved_ids, valid_gt_ids, K))
                avg_recall = np.mean(recalls)
                qps = len(tests) / elapsed if elapsed > 0 else 0
                # print(f"Batch {batch_idx}: QPS={qps:.2f}, Avg Recall@{K}={avg_recall:.4f}, Time={elapsed:.2f}s")
                stats = {
                    'qps': qps,
                    'avg_recall': avg_recall,
                    'elapsed': elapsed,
                    'num_queries': len(tests)
                }
                trade_off[(num_attribute, card, distribution)][K_n] = stats





import pickle
with open(os.path.join("/home/mintaek/hybrid_index/methods/Post_Pre_Filtering", "trade_off_new.pkl"), "wb") as f:
    pickle.dump(trade_off, f)

