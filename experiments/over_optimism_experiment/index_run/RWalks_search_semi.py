from tqdm import tqdm
import time
import numpy as np
import random
import hnswlib
from tqdm import tqdm
import time
import json
import os 
import h5py
import subprocess
import matplotlib.pyplot as plt
import re
import ast
from typing import List, Dict, Any



def parse_conditions(cond_dict):
    """test['conditions']에서 (attr, value) 쌍을 list로 추출"""
    # 현재 구조는 반드시 'and': [ ... ] 만 지원한다고 가정
    conditions = []
    if "and" in cond_dict:
        for cond in cond_dict["and"]:
            for attr, value_dict in cond.items():
                # value_dict: {'match': {'value': XXX}}
                v = value_dict.get("match", {}).get("value")
                if v is not None:
                    conditions.append((attr, v))
    # 향후 or, not 등 확장 가능
    return conditions


ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def strip_ansi(text: str) -> str:
    """ANSI escape 제거"""
    return ANSI_ESCAPE.sub('', text)

def parse_rwalks_stats(stdout: str):
    stdout = strip_ansi(stdout)  # 색상 코드 제거
    lines = stdout.splitlines()

    results = []
    current_params = None

    for line in lines:
        s = line.strip()

        # -------- 1) Search Params 줄 감지 --------
        if s.startswith("Search Params"):
            # 줄에서 dict 부분을 검색
            match = re.search(r"\{.*\}", s)
            if match:
                try:
                    params = ast.literal_eval(match.group(0))
                    current_params = params
                except Exception:
                    current_params = None
            continue

        # -------- 2) stats dict 줄 감지 --------
        if current_params is not None and "qps_4_threads" in s:
            cleaned = s.replace("np.float64(", "").replace(")", "")
            try:
                stats = ast.literal_eval(cleaned)
                qps = float(stats.get("qps_4_threads", None))
                rec = None
                if "recalls" in stats and "top10" in stats["recalls"]:
                    rec = float(stats["recalls"]["top10"])

                results.append({
                    "params": current_params,
                    "qps_4_threads": qps,
                    "recall_top10": rec,
                })
            except Exception:
                pass

            current_params = None   # 한 세트 끝
            continue

    return results



dataset_list = ["arxiv", "LAION1M", "tripclick", "yfcc"]


trade_off = {}

for dataset in dataset_list:    
    dataset_path = f"/home/ec2-user/hybrid_hardness/semi-real/filterbenchmark/{dataset}"

    

    if dataset == "sift_high" or dataset == "sift_low" or dataset == "gist_high" or dataset == "gist_low":
        vectors_file = f"{dataset_path}/vectors.npy"
        payloads_file = f"{dataset_path}/payloads_all.jsonl"
        tests_file = f"{dataset_path}/tests.jsonl"
        mapping_path = os.path.join(dataset_path, "mapping_all.json")
    else:    
        vectors_file = f"{dataset_path}/hardness_format/vectors.npy"
        payloads_file = f"{dataset_path}/hardness_format/payloads.jsonl"
        tests_file = f"{dataset_path}/hardness_format/tests.jsonl"
        mapping_path = os.path.join(dataset_path, "mid_format/mapping.json")

    # vectors_file = f"{DATA_DIR}/base_vectors.npy"
    # # print("vector file path", vectors_file)
    # payloads_file = f"{DATA_DIR}/payloads.jsonl"
    # tests_file = f"{DATA_DIR}/tests.jsonl"

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
    query_vectors = np.array([test["query"] for test in tests])
    
    
    # ------------------------------------
    # 3. Load mapping.json
    # ------------------------------------
    with open(mapping_path, "r") as f:
        mapping = json.load(f)


    ground_truth = [test["closest_ids"] for test in tests]
    
    gt_lists = [test["closest_ids"] for test in tests]
    gt_arr = np.array(gt_lists)
    maxlen = max(len(x) for x in gt_lists)

    repeated = np.empty((len(gt_lists), maxlen), dtype=np.int32)
    for i, arr in enumerate(gt_lists):
        l = len(arr)
        # 리스트 원소를 반복해서 maxlen까지 채우기
        repeated[i, :] = np.resize(arr, maxlen)
        # 또는 (동일한 결과, 안전한 방법)
        # repeated[i, :] = [arr[j % l] for j in range(maxlen)]

    ground_truth = repeated

    
        
    cardinality = len(mapping)

    multi_hot_vectors = []
    for payload in tqdm(payloads):
        mh_vec = np.zeros(cardinality, dtype=np.int32)
        for k, v in payload.items():
            key = f"{k}:{v}"
            idx = mapping.get(key)
            if idx is not None:
                # mapping은 1-based index로 보이므로 -1
                mh_vec[int(idx) - 1] = 1
            else:
                # mapping에 없는 값은 무시 (필요시 warning)
                pass
        multi_hot_vectors.append(mh_vec)
    multi_hot_vectors = np.stack(multi_hot_vectors)

    onehot_vectors = []
    for test in tests:
        oh_vec = np.zeros(cardinality, dtype=np.int32)
        attr_value_pairs = parse_conditions(test["conditions"])
        for attr, value in attr_value_pairs:
            key = f"{attr}:{value}"
            idx = mapping.get(key)
            if idx is not None:
                # mapping이 1-based 인덱스라고 했으므로 -1
                oh_vec[int(idx) - 1] = 1
            else:
                # mapping에 없는 값은 무시
                pass
        onehot_vectors.append(oh_vec)

    onehot_vectors = np.stack(onehot_vectors).astype(np.int8)
    # print(onehot_vectors.shape)
    rwalks_path = dataset_path+"/rwalks_format"
    os.makedirs(rwalks_path, exist_ok=True)

    fname = os.path.join(rwalks_path, "rwalks.h5")
    with h5py.File(fname, "w") as f:
        f.create_dataset("train_vectors", data=vectors)
        f.create_dataset("test_vectors", data=query_vectors)
        f.create_dataset("train_attr_vectors", data=multi_hot_vectors)
        f.create_dataset("test_attr_vectors", data=onehot_vectors)
        f.create_dataset("neighbors", data=gt_arr)

    print(f"Saved: {fname}")

    cmd = ["bash", "/home/ec2-user/hybrid_hardness/methods/RWalks/sift_test_no_batch.sh", fname]
    cwd = "/home/ec2-user/hybrid_hardness/methods/RWalks"
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    # print(result.stdout)
    raw = result.stdout
    
    parsed = parse_rwalks_stats(raw)
    trade_off[dataset] = {}
    for item in parsed:
        trade_off[dataset][tuple(item["params"].values())] = {"qps": item["qps_4_threads"], "avg_recall": item["recall_top10"]}


import pickle
with open(os.path.join(".", "RWalks_trade_off_result_semi.pkl"), "wb") as f:
    pickle.dump(trade_off, f)