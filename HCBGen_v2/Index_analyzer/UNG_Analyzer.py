import os
import numpy as np
import json
import subprocess
import struct
import pandas as pd
from Index_analyzer import pareto_comp as pc


def Analyze(mode):
    if mode == True:
        return "pre_base"
    ##################################################################################################
    dataset_name = "sift1m"
    ##################################################################################################

    trade_off_UNG = {}

    for dataset_name_1, cardinality, num_attribute, distribution in zip(
        ["closer_to_post", "closer_to_pre"],
        [[6, 6, 6], [6] * 10],
        [3, 10],
        ["random", "zipf"]
    ):
        print(f"\n[Dataset] Start → {dataset_name_1}")
        cardi = '_'.join(str(c) for c in cardinality)
        data_path = f"/home/mintaek/hybrid_index/Benchmark/test_dataset/{dataset_name}_A{num_attribute}_{cardi}_{distribution}"
        hardness_path = os.path.join(data_path, "hardness_format")
        query_label_path = os.path.join(data_path, "mid_format/query_label.txt")
        base_vector_path = os.path.join(data_path, "mid_format/base_vector.bin")
        base_label_path = os.path.join(data_path, "mid_format/base_label_UNG.txt")

        tests_file = os.path.join(hardness_path, "tests.jsonl")

        # ─────────────────────────────────────────────────────────────
        # [Data Loading]
        # ─────────────────────────────────────────────────────────────
        print("[Data] Loading test set …")
        tests = []
        with open(tests_file, "r") as f:
            for line in f:
                tests.append(json.loads(line))
        print(f"  └─ Loaded test queries: {len(tests):,}")


        ung_root = os.path.join(data_path, "ung_format")
        batch_dir = os.path.join(data_path, "mid_format")

        query_bin = os.path.join(batch_dir, "query_vector.bin")
        gt_file = os.path.join(batch_dir, "gt.bin")
        gt_counts_file = os.path.join(batch_dir, "count.txt")

        # ─────────────────────────────────────────────────────────────
        # [Index Building]
        # ─────────────────────────────────────────────────────────────
        print("[Index] Building UNG index …")
        binary = "/home/mintaek/hybrid_index/methods/Unified-Navigating-Graph/build/apps/build_UNG_index"

        cmd = [
            binary,
            "--data_type", "float",
            "--dist_fn", "L2",
            "--num_threads", "64",
            "--max_degree", "32",
            "--Lbuild", "100",
            "--alpha", "1.2",
            "--base_bin_file", base_vector_path,
            "--base_label_file", base_label_path,
            "--index_path_prefix", ung_root + "/ung_index/",
            "--scenario", "general",
            "--num_cross_edges", "6"
        ]

        try:
            # subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(cmd, check=True)
            print("  └─ Index build complete ✅")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Index build failed: {e}")

        # ─────────────────────────────────────────────────────────────
        # [Query Execution]
        # ─────────────────────────────────────────────────────────────
        print("[Querying] Running search …")
        binary = "/home/mintaek/hybrid_index/methods/Unified-Navigating-Graph/build/apps/search_UNG_index"

        cmd = [
            binary,
            "--data_type", "float",
            "--dist_fn", "L2",
            "--num_threads", "32",
            "--K", "10",
            "--base_bin_file", base_vector_path,
            "--base_label_file", base_label_path,
            "--query_bin_file", query_bin,
            "--query_label_file", query_label_path,
            "--gt_file", gt_file,
            "--gt_counts_file", gt_counts_file,
            "--index_path_prefix", ung_root + "/ung_index/",
            "--result_path_prefix", "/home/mintaek/hybrid_index/Benchmark/utils/data_process/ung_results/",
            "--scenario", "containment",
            "--num_entry_points", "16",
            "--Lsearch", "150", "100", "50", "30", "25", "15", "10"
        ]

        # subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(cmd, check=True)
        print("  └─ Query execution complete ✅")

        # ─────────────────────────────────────────────────────────────
        # [Results Parsing]
        # ─────────────────────────────────────────────────────────────
        csv_path = "/home/mintaek/hybrid_index/Benchmark/utils/data_process/ung_results/result.csv"

        if not os.path.exists(csv_path):
            print("  ⚠️  result.csv not found, skipping …")
            continue

        df = pd.read_csv(csv_path, thousands=",")
        try:
            df["L"] = pd.to_numeric(df["L"], errors="coerce")
            df["QPS"] = pd.to_numeric(df["QPS"], errors="coerce")
            df["Recall"] = pd.to_numeric(df["Recall"], errors="coerce")
        except Exception:
            pass

        trade_off_UNG[dataset_name_1] = {}
        for _, row in df.iterrows():
            L = int(row["L"]) if not pd.isna(row["L"]) else None
            if L is None:
                continue

            qps = float(row["QPS"])
            recall = float(row["Recall"])
            trade_off_UNG[dataset_name_1][L] = {
                "qps": qps,
                "avg_recall": recall,
                "elapsed": None,
                "num_queries": None
            }
            print(f"  ├─ [L={L}] QPS={qps:,.2f} | Recall={recall:.4f}")

        print(f"  └─ Completed parsing results for {dataset_name_1}")

    # ─────────────────────────────────────────────────────────────
    # [Final Scores]
    # ─────────────────────────────────────────────────────────────
    post_score = pc.final_score(trade_off_UNG["closer_to_post"])
    pre_score = pc.final_score(trade_off_UNG["closer_to_pre"])

    print("\n[Final Scores]")
    print(f"  ├─ Post-base Score : {post_score:.6f}")
    print(f"  └─ Pre-base  Score : {pre_score:.6f}")

    # ─────────────────────────────────────────────────────────────
    # [Decision]
    # ─────────────────────────────────────────────────────────────
    if post_score > pre_score:
        print("[Decision] → ✅ post_base selected")
        return "post_base"
    else:
        print("[Decision] → ✅ pre_base selected")
        return "pre_base"
