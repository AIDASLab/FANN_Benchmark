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

dataset_name_list = ["arxiv", "LAION1M", "tripclick", "yfcc"]


for dataset_name in dataset_name_list:

    dataset_path = f"/home/ec2-user/hybrid_hardness/semi-real/filterbenchmark/{dataset_name}"

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

    query_num = len(tests)
    print(f"Loaded {query_num} tests")


    # # ------------------------------------
    # # 예시 출력
    # print("\nSample payload:", payloads[0])
    # print("\nSample test:", tests[0])
    
    for sort_hardness in ["Post_Hardness","selectivity", "correlation"]:
    # for sort_hardness in ["Post_Hardness"]:
    # for sort_hardness in ["min", "max"]:
        if sort_hardness == "selectivity" or sort_hardness == "correlation" or sort_hardness == "select_corr_combine":
            baseline = 1
        else:
            baseline = 0

        # 1. Load hardness and GT
        space = "l2"
        # space = "cosine"

        if baseline == 1:
            hardness_path = os.path.join(dataset_path, f"hardness/hardness_baseline_{query_num}.json")
        else:
            hardness_path = os.path.join(dataset_path, f"hardness/hardness_v5.1_{query_num}.json")

        with open(hardness_path) as f:
            results = json.load(f)


        for i, r in enumerate(results):
                tests[i]["Hardness"] = r[sort_hardness]

        sorted_tests = sorted(tests, key=lambda x: x['Hardness'])

        # sorting 하고 난 뒤 원래 index를 results에 저장해주기
        for i, r in enumerate(results):
            r['orig_idx'] = i


        ## post filtering 측정하기
        index = hnswlib.Index(space=space, dim=len(vectors[0]))
        index.init_index(max_elements=len(vectors), ef_construction=50, M=8)
        index.add_items(vectors, num_threads=32)

        num_batches = len(tests) // (query_num / 10) + (1 if len(tests) % (query_num / 10) != 0 else 0)
        num_batches = int(num_batches)
        if dataset_name == "HnM":
            K = 25  # top-K
        else:
            K = 10
        
        trade_off = {}
        batch_size = int(query_num / 10)
        for K_n in [5, 10, 20, 50, 100]:
            batch_stats = []
            for batch_idx in tqdm(range(num_batches)):
                batch_tests = sorted_tests[batch_idx * batch_size : (batch_idx+1)*batch_size]
                # batch_tests는 Hardness 기준 정렬된 tests의 slice
                t0 = time.time()
                batch_results = post_filtering(index, K, batch_tests, payloads, K_n, space)
                t1 = time.time()
                elapsed = t1 - t0
                recalls = []
                for i, test in enumerate(batch_tests):
                    gt_ids = test['closest_ids']
                    retrieved_ids = batch_results[i]

                    # gt에서 4294967295 제거
                    valid_gt_ids = [gt for gt in gt_ids if gt != 4294967295]

                    # valid_gt가 없으면 recall 정의 불가 → 0으로 처리
                    if len(valid_gt_ids) == 0:
                        recalls.append(1.0)
                    else:
                        recalls.append(recall_at_k(retrieved_ids, valid_gt_ids, K))
                avg_recall = np.mean(recalls)
                qps = len(batch_tests) / elapsed if elapsed > 0 else 0
                # print(f"Batch {batch_idx}: QPS={qps:.2f}, Avg Recall@{K}={avg_recall:.4f}, Time={elapsed:.2f}s")
                batch_stats.append({
                    'batch': batch_idx,
                    'qps': qps,
                    'avg_recall': avg_recall,
                    'elapsed': elapsed,
                    'num_queries': len(batch_tests)
                })
            trade_off[K_n] = batch_stats



        post_filter_path =  os.path.join(dataset_path, "post_filter_format")
        os.makedirs(post_filter_path, exist_ok=True)
        output_file = os.path.join(post_filter_path, f"{sort_hardness}_search_results.txt")


        with open(output_file, "w") as f:
            # 헤더
            f.write("Batch\tK\tQPS\tAvg_Recall\n")
            num_batches = len(next(iter(trade_off.values())))  # batch 개수
            k_values = sorted(trade_off.keys())

            for batch_idx in range(num_batches):
                for K in k_values:
                    stats = trade_off[K][batch_idx]
                    qps = stats['qps']
                    recall = stats['avg_recall']
                    f.write(f"{batch_idx+1}\t{K}\t{qps}\t{recall}\n")

        print(f"[✓] trade_off 저장 완료 (Batch 기준 정렬): {output_file}")



        ## post filtering graph 그리기



        plt.figure(figsize=(8, 5))
        colors = plt.cm.tab10.colors  # 10개 batch 색상

        num_batches = len(next(iter(trade_off.values())))  # 10개 batch로 가정
        k_values = sorted(trade_off.keys())

        for batch_idx in range(num_batches):
            qps_list = []
            recall_list = []
            for K in k_values:
                batch_stats = trade_off[K][batch_idx]  # 각 K별 batch_idx번째 dict
                qps_list.append(float(batch_stats['qps']))
                recall_list.append(float(batch_stats['avg_recall']))
            plt.plot(qps_list, recall_list, marker='o', color=colors[batch_idx % 10], label=f'Batch {batch_idx+1}')
            # 점마다 K 표시 원하면
            # for i, K in enumerate(k_values):
            #     plt.text(qps_list[i], recall_list[i], f"{K}", fontsize=8, color=colors[batch_idx % 10])

        plt.xlabel("QPS (Queries per second)")
        plt.ylabel("Recall")
        plt.title(f"Recall-QPS Trade-off per Batch (K_n varies)")
        plt.legend(title="Batch")
        plt.grid(True)
        plt.tight_layout()

        pig_path = os.path.join(post_filter_path, f"{sort_hardness}.png")
        plt.savefig(pig_path, dpi=300)
        # plt.show()

        print(dataset_path)
        print(sort_hardness)