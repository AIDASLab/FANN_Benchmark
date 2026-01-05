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

def save_gt_ivecs(filename, gt_list):
    with open(filename, 'wb') as f:
        for row in gt_list:
            arr = np.array(row, dtype=np.int64)
            arr[arr == 4294967295] = -1
            arr = arr.astype(np.int32)
            K = len(arr)
            f.write(np.array([K], dtype=np.int32).tobytes())
            f.write(arr.tobytes())

dataset_list = ["sift_high","sift_low","gist_high","gist_low","sift1m_ACORN", "sift1m_NHQ","sift1m_UNG","sift1m_RWalks"]
# dataset_list = ["sift1m_UNG_modi"]
dataset_name_list = ["sift1m", "sift1m","gist1m","gist1m", "sift1m", "sift1m", "sift1m", "sift1m"]

NHQ_trade_off = {}
for dataset , dataset_name in zip(dataset_list, dataset_name_list):
    print("processing:", dataset)
    original_data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset_name}_original"
    
    data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset}"
    mid_path = os.path.join(data_path, "mid_format")
    NHQ_path = os.path.join(data_path, "nhq_format")
    if dataset == "sift_high" or dataset == "sift_low" or dataset == "gist_high" or dataset == "gist_low":
        hardness_path = data_path
        payloads_path = os.path.join(hardness_path, 'payloads_all.jsonl')
        temp_base_label_path = os.path.join(mid_path, "base_label_all.txt")
    else:    
        hardness_path = os.path.join(data_path, "hardness_format")
        payloads_path = os.path.join(hardness_path, 'payloads.jsonl')
        temp_base_label_path = os.path.join(mid_path, "base_label.txt")
    

    temp_label_data = []
    with open(temp_base_label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip() # 앞뒤 공백 및 줄바꿈 제거
            if not line:        # 빈 줄이면 건너뜀
                continue
            row_items = [int(x) for x in line.split(',')]
            temp_label_data.append(row_items)

    num_attribute = len(temp_label_data[0])
    print("number attribute: ", num_attribute)

    with open(payloads_path, 'r') as fin:
        lines = fin.readlines()

    output_path = os.path.join(mid_path, 'base_label_NHQ.txt')
    with open(output_path, 'w') as fout:
        ################## 이부분 동작 확임
        fout.write(f"{len(lines)} {num_attribute}\n")  # 첫 줄: 데이터 개수, 속성 개수
        for line in lines:
            payload = json.loads(line)
            row = []
            for i in range(1, num_attribute + 1):
                key = f"label_{i}"
                value = payload.get(key, 0)
                row.append(str(value))
            fout.write(" ".join(row) + "\n")
    
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
            conditions = test["conditions"].get("and", [])
            for cond in conditions:
                for key, value in cond.items():
                    if key.startswith("label_") and "match" in value:
                        idx = int(key.split("_")[1]) - 1
                        labels[idx] = value["match"]["value"]
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
        
    base_vector = os.path.join(original_data_path, f"{dataset_name}_base.fvecs")
    base_label = os.path.join(mid_path, "base_label_NHQ.txt")

    os.makedirs(os.path.join(NHQ_path, "NHQ_index"), exist_ok=True)
    maxm0_list = [10, 20, 30, 40, 50, 100, 200, 300]
    efc_list = [30, 50, 70, 100, 150, 300, 500, 700]

    for maxm0, efc in zip(maxm0_list, efc_list):
        curr_index_path = os.path.join(NHQ_path, "NHQ_index", f"M{maxm0}_ef{efc}")
        os.makedirs(curr_index_path, exist_ok=True)

        index_bin_path = os.path.join(curr_index_path, "index.bin")
        index_txt_path = os.path.join(curr_index_path, "index.txt")

        # args = [
        #     "NHQ-NPG_nsw",
        #     base_vector,
        #     base_label,
        #     index_bin_path,
        #     index_txt_path,
        #     str(maxm0),
        #     str(efc),
        # ]

        # input_str = "\n".join(args) + "\n"
        # cmd = ["python", "test_hybrid_query.py", "build"]
        # workdir = "/home/ec2-user/hybrid_hardness/methods/NHQ"

        # subprocess.run(
        #     cmd,
        #     input=input_str,
        #     text=True,
        #     capture_output=True,
        #     cwd=workdir
        # )

        # lines = []
        # with open(index_txt_path, "r") as f:
        #     for line in f:
        #         line = line.strip()
        #         lines.append("" if not line else line + " 0")
        # with open(index_txt_path, "w") as f:
        #     f.write("\n".join(lines) + "\n")

        # print(f"  ├─ [Index] M={maxm0}, efC={efc} → build complete ✅")

    print("\n[Query] Running hybrid query tests...")
    nhq_index_root = os.path.join(NHQ_path, "NHQ_index")

    query_fvecs = os.path.join(original_data_path, f"{dataset_name}_query.fvecs")
    query_label = os.path.join(mid_path, "query_label_NHQ.txt")
    gt_ivecs = os.path.join(mid_path, "gt.ivecs")



    NHQ_trade_off[dataset] = {}
    for idx_dir in sorted(os.listdir(nhq_index_root)):
        idx_path = os.path.join(nhq_index_root, idx_dir)
        if not os.path.isdir(idx_path):
            continue

        bin_path = os.path.join(idx_path, "index.bin")
        txt_path = os.path.join(idx_path, "index.txt")

        args = [
            "NHQ-NPG_nsw",
            bin_path,
            txt_path,
            query_fvecs,
            query_label,
            gt_ivecs
        ]
        input_str = "\n".join(args) + "\n"
        cmd = ["python", "test_hybrid_query.py", "search"]
        workdir = "/home/ec2-user/hybrid_hardness/methods/NHQ"

        result = subprocess.run(
            cmd,
            input=input_str,
            text=True,
            capture_output=True,
            cwd=workdir
        )

        output = result.stderr
        # print(output)

        m_match = re.search(r'M(\d+)_ef(\d+)', idx_dir)
        if not m_match:
            continue
        M = int(m_match.group(1))
        ef = int(m_match.group(2))
        found = re.findall(r"Search Time.*?([\d.]+).*?accuracy.*?([\d.]+)", output)
        if not found:
            print(f"  ├─ [Skip] M={M}, ef={ef} → result not found ⚠️")
            continue

        for search_time, accuracy in found:
            NHQ_trade_off[dataset][(M, ef)] = {
                "qps": 10000.0 / float(search_time),
                "avg_recall": float(accuracy)
            }
            print(f"  ├─ [Result] M={M}, ef={ef} | Time={search_time}s | Acc={accuracy}")

import pickle
with open(os.path.join(".", "NHQ_trade_off_result.pkl"), "wb") as f:
    pickle.dump(NHQ_trade_off, f)