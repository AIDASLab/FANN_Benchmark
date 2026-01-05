from tqdm import tqdm
import time
import numpy as np
import random
import hnswlib
from tqdm import tqdm
import time
import json
import os 
import shutil
import re
import subprocess
import matplotlib.pyplot as plt


# dataset_name = "sift1m"
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
def save_gt_ivecs(filename, gt_list):
    with open(filename, 'wb') as f:
        for row in gt_list:
            arr = np.array(row, dtype=np.int64)
            arr[arr == 4294967295] = -1
            arr = arr.astype(np.int32)
            K = len(arr)
            f.write(np.array([K], dtype=np.int32).tobytes())
            f.write(arr.tobytes())

#$###################################################################
dataset_list = ["arxiv", "LAION1M", "tripclick", "yfcc"]
# dataset_list = ["sift1m_UNG_modi"]
#$###################################################################


NHQ_trade_off = {}
for dataset in dataset_list:
    print("processing:", dataset)
    # original_data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset_name}_original"
    
    data_path = f"/home/ec2-user/hybrid_hardness/semi-real/filterbenchmark/{dataset}"
    mid_path = os.path.join(data_path, "mid_format")
    NHQ_path = os.path.join(data_path, "nhq_format")
    if dataset == "sift_high" or dataset == "sift_low" or dataset == "gist_high" or dataset == "gist_low":
        hardness_path = data_path
        payloads_path = os.path.join(hardness_path, 'payloads_all.jsonl')
    else:    
        hardness_path = os.path.join(data_path, "hardness_format")
        payloads_path = os.path.join(hardness_path, 'payloads.jsonl')
        mapping_path = os.path.join(data_path, "mid_format/mapping.json")    

    base_vector = os.path.join(data_path, f"{dataset}_base.fvecs")
    
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

        
    num_attribute = len(mapping)
    print("number attribute: ", num_attribute)

    with open(payloads_path, 'r') as fin:
        lines = fin.readlines()

    output_path = os.path.join(mid_path, 'base_label_NHQ.txt')
    with open(output_path, 'w') as fout:
        ################## 이부분 동작 확임
        fout.write(f"{len(lines)} {num_attribute}\n")  # 첫 줄: 데이터 개수, 속성 개수
        for line in lines:
            payload = json.loads(line)
            row = [0] * num_attribute
            for attr, value in payload.items():
                key = f"{attr}:{value}"
                idx = mapping.get(key)
                row[int(idx) - 1] = 1
            # row = []
            # for i in range(1, num_attribute + 1):
            #     key = f"label_{i}"
            #     value = payload.get(key, 0)
            #     row.append(str(value))
                
            fout.write(" ".join(str(v) for v in row) + "\n")
    
    tests_path = os.path.join(hardness_path, 'tests.jsonl')
    output_path = os.path.join(mid_path, 'query_label_NHQ.txt')

    print("[Step] Extracting query labels...")
    all_labels = []
    with open(tests_path, 'r') as fin:
        for line in fin:
            if not line.strip():
                continue
            test = json.loads(line)
            labels = [0] * num_attribute
            # conditions = test["conditions"].get("and", [])
            attr_value_pairs = parse_conditions(test["conditions"])
            for attr, value in attr_value_pairs:
                key = f"{attr}:{value}"
                idx = mapping.get(key)
                if idx is not None:
                    # mapping이 1-based 인덱스라고 했으므로 -1
                    labels[int(idx) - 1] = 1
                else:
                    # mapping에 없는 값은 무시
                    pass
            all_labels.append(labels)

    with open(output_path, 'w') as fout:
        num_rows = len(all_labels)
        num_cols = num_attribute
        fout.write(f"{num_rows} {num_cols}\n")
        for labels in all_labels:
            fout.write(" ".join(str(v) for v in labels) + "\n")

    tests = []
    with open(tests_path, 'r') as fin:
        for line in fin:
            tests.append(json.loads(line))

    print(f"[Data] Loaded {len(tests):,} test cases")

    gt_list = [t["closest_ids"] for t in tests]
    save_gt_ivecs(os.path.join(mid_path, "gt.ivecs"), np.array(gt_list))
    print("[Data] Saved groundtruth (gt.ivecs)")

    print("[Index] Building NHQ index...")
        
    
    base_label = os.path.join(mid_path, "base_label_NHQ.txt")

    os.makedirs(os.path.join(NHQ_path, "NHQ_index"), exist_ok=True)
    maxm0_list = [10, 20, 30, 40, 50, 100, 200, 300]
    efc_list = [30, 50, 70, 100, 150, 300, 500, 700]

    for maxm0, efc in zip(maxm0_list, efc_list):
        curr_index_path = os.path.join(NHQ_path, "NHQ_index", f"M{maxm0}_ef{efc}")
        os.makedirs(curr_index_path, exist_ok=True)

        index_bin_path = os.path.join(curr_index_path, "index.bin")
        index_txt_path = os.path.join(curr_index_path, "index.txt")

        args = [
            "NHQ-NPG_nsw",
            base_vector,
            base_label,
            index_bin_path,
            index_txt_path,
            str(maxm0),
            str(efc),
        ]

        input_str = "\n".join(args) + "\n"
        cmd = ["python", "test_hybrid_query.py", "build"]
        workdir = "/home/ec2-user/hybrid_hardness/methods/NHQ"

        subprocess.run(
            cmd,
            input=input_str,
            text=True,
            capture_output=True,
            cwd=workdir
        )

        # lines = []
        # with open(index_txt_path, "r") as f:
        #     for line in f:
        #         line = line.strip()
        #         lines.append("" if not line else line + " 0")
        # with open(index_txt_path, "w") as f:
        #     f.write("\n".join(lines) + "\n")

        print(f"  ├─ [Index] M={maxm0}, efC={efc} → build complete ✅")

