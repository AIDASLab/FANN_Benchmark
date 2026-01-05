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

import numpy as np
import os

def save_gt_bin_from_list(gt_list, out_path):
    """
    Save ground truth from a Python list so that
    load_groundtruth_bin() can read it.

    Parameters
    ----------
    gt_list : List[List[int]]
        shape = (nq, K)
    out_path : str
        output gt.bin path
    """
    gt_indices = np.asarray(gt_list, dtype=np.uint32)

    if gt_indices.ndim != 2:
        raise ValueError("gt_list must be 2D: (nq, K)")

    nq, K = gt_indices.shape

    # dist는 의미 없으므로 0.0으로 채움
    gt_dists = np.zeros((nq, K), dtype=np.float32)

    record_dtype = np.dtype([
        ("idx",  np.uint32),
        ("dist", np.float32),
    ])

    records = np.empty(nq * K, dtype=record_dtype)
    records["idx"]  = gt_indices.reshape(-1)
    records["dist"] = gt_dists.reshape(-1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    records.tofile(out_path)

    print(f"[Saved] {out_path}")
    print(f"  ├─ nq={nq}, K={K}")
    print(f"  └─ total records={nq*K}")


dataset_list = ["sift_high","sift_low","gist_high","gist_low"]
# dataset_list = ["sift_NHQ","sift_UNG","sift_RWalks"]
dataset_list = ["sift_high","sift_low","gist_high","gist_low", "sift1m_UNG", "sift1m_RWalks","sift1m_NHQ","sift1m_ACORN"]


# dataset_list = ["sift1m_UNG_modi"]
# dataset_name_list = ["sift1m"]


trade_off_UNG = {}
dataset_list1 = ["arxiv_generated", "LAION1M_generated", "tripclick_generated", "yfcc_generated"]
dataset_name_list = ["arxiv", "LAION1M", "tripclick", "yfcc"]
dataset_list2 = ["arxiv_generated_wo_payload", "LAION1M_generated_wo_payload", "tripclick_generated_wo_payload", "yfcc_generated_wo_payload"]
dataset_list = dataset_list1+dataset_list2
for dataset, dataset_name in zip(dataset_list, dataset_name_list):
    original_data_path = f"/home/ec2-user/hybrid_hardness/fidelity_experiment/{dataset}"
    
    data_path = original_data_path
    hardness_path = os.path.join(data_path)
    query_fvecs_path = os.path.join(original_data_path, f"mid_format/query_vectors_used.fvecs")
    query_bin = os.path.join(original_data_path, f"mid_format/query_vectors_used.bin")
    subprocess.run([
            "/home/ec2-user/hybrid_hardness/methods/Unified-Navigating-Graph/build/tools/fvecs_to_bin",
            "--data_type", "float",
            "--input_file", query_fvecs_path,
            "--output_file", query_bin
        ], check=True)
    base_vector_path = os.path.join(data_path, f"mid_format/base_vector.bin")  
    base_label_path = os.path.join(data_path, "mid_format/base_label.txt")
    query_label_path = os.path.join(data_path, f"mid_format/query_label.txt")
    
    
    
    gt_file = os.path.join(data_path, f"mid_format/gt_UNG.txt")
    gt_counts_file = "/home/ec2-user/hybrid_hardness/semi-real/filterbenchmark/tripclick/count.txt"
    ung_root = os.path.join(data_path, "ung_format")
    tests_file = f"{hardness_path}/tests.jsonl"
    # index_path = f"/home/ec2-user/hybrid_hardness/semi-real/filterbenchmark/{dataset_name}/ung_format/ung_index/"


    tests = []
    with open(tests_file, "r") as f:
        for line in f:
            tests.append(json.loads(line))

    ground_truth = [test["closest_ids"] for test in tests]
    save_gt_bin_from_list(ground_truth, gt_file)
    print(f"Loaded {len(tests)} tests")
    
    
    binary = "/home/ec2-user/hybrid_hardness/methods/Unified-Navigating-Graph/build/apps/build_UNG_index"
    
    cmd = [
        binary,
        "--data_type", "float",
        "--dist_fn", "L2",
        "--num_threads", "4",
        "--max_degree", "32",
        "--Lbuild", "100",
        "--alpha", "1.2",
        "--base_bin_file", base_vector_path,
        "--base_label_file", base_label_path,
        "--index_path_prefix", ung_root+"/ung_index/",
        "--scenario", "general",
        "--num_cross_edges", "6"
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[에러] 명령어 실행 실패: {e}")
    
    
    ########## search start     
    K = "10"


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
with open(os.path.join(".", "UNG_trade_off_result_generated.pkl"), "wb") as f:
    pickle.dump(trade_off_UNG, f)