import os
import numpy as np
import json
import subprocess
import struct
import os
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd

dataset_list = ["sift_high","sift_low","gist_high","gist_low"]
# dataset_list = ["sift_NHQ","sift_UNG","sift_RWalks"]
dataset_list = ["sift_high","sift_low","gist_high","gist_low", "sift1m_UNG", "sift1m_RWalks","sift1m_NHQ","sift1m_ACORN"]
dataset_name_list = ["sift1m","sift1m","gist1m","gist1m", "sift1m", "sift1m","sift1m","sift1m"]

# dataset_list = ["sift1m_UNG_modi"]
# dataset_name_list = ["sift1m"]


trade_off_UNG = {}
for dataset , dataset_name in zip(dataset_list, dataset_name_list):
    original_data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset_name}_original"
    if dataset_name == "sift1m" or dataset_name == "glove1m" or dataset_name == "gist1m":
        data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset}"
        if dataset == "sift1m_UNG" or dataset == "sift1m_NHQ" or dataset == "sift1m_ACORN" or dataset == "sift1m_RWalks" or dataset == "sift1m_UNG_modi":
            hardness_path = os.path.join(data_path, "hardness_format")
            base_label_path = os.path.join(data_path, "mid_format/base_label_UNG.txt")
        else:
            hardness_path = data_path
            base_label_path = os.path.join(data_path, "mid_format/base_label.txt")
        query_fvecs_path = os.path.join(original_data_path, f"{dataset_name}_query.fvecs")
        query_label_path = os.path.join(data_path, "mid_format/query_label.txt")
        base_vector_path = os.path.join(data_path, "mid_format/base_vector.bin")
        
        # hardness_json_path = os.path.join(data_path, "hardness/hardness_v3.0_10000.json")

    ung_root = os.path.join(data_path, "ung_format")
    tests_file = f"{hardness_path}/tests.jsonl"
    tests = []
    with open(tests_file, "r") as f:
        for line in f:
            tests.append(json.loads(line))
    ground_truth = [test["closest_ids"] for test in tests]

    print(f"Loaded {len(tests)} tests")
    
    ########## search start     
    K = "10"

    query_bin = os.path.join(data_path, "mid_format/query_vector.bin")
    gt_file = os.path.join(data_path, "mid_format/gt.bin")
    gt_counts_file = os.path.join(data_path, "mid_format/count.txt")

    binary = "/home/ec2-user/hybrid_hardness/methods/Unified-Navigating-Graph/build/apps/search_UNG_index"

    cmd = [
        binary,
        "--data_type", "float",
        "--dist_fn", "L2",
        "--num_threads", "16",
        "--K", K,
        "--base_bin_file", base_vector_path,
        "--base_label_file", base_label_path,
        "--query_bin_file", query_bin,
        "--query_label_file", query_label_path,
        "--gt_file", gt_file,
        "--gt_counts_file", gt_counts_file,
        "--index_path_prefix",  ung_root+"/ung_index/",
        "--result_path_prefix", ung_root+"/ung_result/",
        "--scenario", "containment",
        "--num_entry_points", "16",
        "--Lsearch", '200', '100', '50' ,'30', '25', '15', '10', '300', '400', '500', '1000',
    ]

    subprocess.run(cmd, check=True)

    csv_path = os.path.join(ung_root, "ung_result/result.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, thousands=",")  # 쉼표 구분자 인식
        try:
            df["L"] = pd.to_numeric(df["L"], errors="coerce")
            df["QPS"] = pd.to_numeric(df["QPS"], errors="coerce")
            df["Recall"] = pd.to_numeric(df["Recall"], errors="coerce")

        except Exception as e:
            pass

    else:
        print("cannot find result.csv")

    trade_off_UNG[dataset] = {}
    for _, row in df.iterrows():
        L = int(row["L"]) if not pd.isna(row["L"]) else None
        if L is None:
            continue

        qps = row["QPS"]
        avg_recall = row["Recall"]

        stats = {
            "qps": float(qps),
            "avg_recall": float(avg_recall),
            "elapsed": None,          # csv에 없으니 None
            "num_queries": None       # csv에 없으니 None
        }

        trade_off_UNG[dataset][L] = stats

import pickle
with open(os.path.join(".", "UNG_trade_off_result_1.pkl"), "wb") as f:
    pickle.dump(trade_off_UNG, f)