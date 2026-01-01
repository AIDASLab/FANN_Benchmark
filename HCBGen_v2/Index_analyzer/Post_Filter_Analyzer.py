import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np
import random
import hnswlib
import json
from Index_analyzer import pareto_comp as pc


def Analyze(mode):
    if mode == True:
        return "post_base"
    ####################################################################
    dataset_name_list = ["closer_to_post", "closer_to_pre"]
    ##################################################################################################

    def satisfies_conditions(payload, conditions):
        if "and" in conditions:
            for cond in conditions["and"]:
                if not isinstance(cond, dict):
                    continue
                for key, rule in cond.items():
                    if "match" in rule and "value" in rule["match"]:
                        if payload.get(key) != rule["match"]["value"]:
                            return False
                    else:
                        return False
            return True
        else:
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
        queries = [t['query'] for t in tests]
        labels, dists = index.knn_query(queries, k=K * K_n)
        for i, test in enumerate(tests):
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

    for dataset_name in dataset_name_list:
        print(f"\n[Dataset] Loading → {dataset_name}")

        if dataset_name == "closer_to_post":
            dataset_path = "/home/mintaek/hybrid_index/Benchmark/test_dataset/sift1m_A3_6_6_6_random"
        elif dataset_name == "closer_to_pre":
            dataset_path = "/home/mintaek/hybrid_index/Benchmark/test_dataset/sift1m_A10_6_6_6_6_6_6_6_6_6_6_zipf"

        DATA_DIR = os.path.join(dataset_path, "hardness_format")

        vectors_file = f"{DATA_DIR}/vectors.npy"
        payloads_file = f"{DATA_DIR}/payloads.jsonl"
        tests_file = f"{DATA_DIR}/tests.jsonl"

        # --- 데이터 로딩 ---
        vectors = np.load(vectors_file)
        print(f"[Vectors] Loaded vectors → shape = {vectors.shape}")

        payloads = []
        with open(payloads_file, "r") as f:
            for line in f:
                payloads.append(json.loads(line))
        print(f"[Payloads] Loaded payloads → count = {len(payloads):,}")

        tests = []
        with open(tests_file, "r") as f:
            for line in f:
                tests.append(json.loads(line))
        print(f"[Tests] Loaded test queries → count = {len(tests):,}")

        # --- HNSW 인덱스 구축 ---
        space = "l2"
        print("\n[Index] Building HNSW index …")
        index = hnswlib.Index(space=space, dim=len(vectors[0]))
        index.init_index(max_elements=len(vectors), ef_construction=50, M=8)
        index.add_items(vectors, num_threads=32)
        print("[Index] Index build complete ✅")

        # --- 쿼리 테스트 ---
        print("[Querying] Running test benchmarks …")
        K = 10
        trade_off[dataset_name] = {}

        for K_n in tqdm([5, 10, 20, 50, 100], desc=f"[{dataset_name}] Evaluating"):
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
                if len(valid_gt_ids) == 0:
                    continue
                recalls.append(recall_at_k(retrieved_ids, valid_gt_ids, K))

            avg_recall = np.mean(recalls)
            qps = len(tests) / elapsed if elapsed > 0 else 0
            trade_off[dataset_name][K_n] = {
                'qps': qps,
                'avg_recall': avg_recall,
                'elapsed': elapsed,
                'num_queries': len(tests)
            }

            print(f"  ├─ [K_n={K_n}] QPS={qps:,.2f} | Recall@{K}={avg_recall:.4f} | Time={elapsed:.2f}s")

    # --- 결과 요약 ---
    post_score = pc.final_score(trade_off["closer_to_post"])
    pre_score = pc.final_score(trade_off["closer_to_pre"])

    print("\n[Final Scores]")
    print(f"  ├─ Post-base Score : {post_score:.6f}")
    print(f"  └─ Pre-base  Score : {pre_score:.6f}")

    if post_score > pre_score:
        print("[Decision] → ✅ post_base selected")
        return "post_base"
    else:
        print("[Decision] → ✅ pre_base selected")
        return "pre_base"
