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
import shutil
import re
import subprocess
import matplotlib.pyplot as plt
from Index_analyzer import pareto_comp as pc



##################################################################################################
def Analyze(mode):
    if mode == True:
        return "post_base"
    dataset_name = "sift1m"

    def save_gt_ivecs(filename, gt_list):
        with open(filename, 'wb') as f:
            for row in gt_list:
                arr = np.array(row, dtype=np.int64)
                arr[arr == 4294967295] = -1
                arr = arr.astype(np.int32)
                K = len(arr)
                f.write(np.array([K], dtype=np.int32).tobytes())
                f.write(arr.tobytes())

    NHQ_trade_off = {}
    for d_name, num_attribute, cardinality, distribution in zip(
        ["closer_to_post", "closer_to_pre"],
        [3, 10],
        ([6] * 3, [6] * 10),
        ["random", "zipf"]
    ):
        print(f"\n[Dataset] Loading → {d_name}")
        original_data_path = f"/home/mintaek/hybrid_index/Benchmark/{dataset_name}_original"
        cardi = '_'.join(str(c) for c in cardinality)
        data_path = f"/home/mintaek/hybrid_index/Benchmark/test_dataset/{dataset_name}_A{num_attribute}_{cardi}_{distribution}"
        hardness_path = os.path.join(data_path, "hardness_format")
        mid_path = os.path.join(data_path, "mid_format")
        NHQ_path = os.path.join(data_path, "nhq_format")

        NHQ_trade_off[d_name] = {}

        tests_path = os.path.join(hardness_path, 'tests.jsonl')
        output_path = os.path.join(mid_path, 'query_label_NHQ_anal.txt')

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
        if dataset_name == "sift1m":
            base_vector = os.path.join(original_data_path, "sift_base.fvecs")
            base_label = os.path.join(mid_path, "base_label_NHQ.txt")

        os.makedirs(os.path.join(NHQ_path, "NHQ_index"), exist_ok=True)
        maxm0_list = [10, 20, 30, 40, 50]
        efc_list = [30, 50, 70, 100, 150]

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
            workdir = "/home/mintaek/hybrid_index/methods/NHQ"

            subprocess.run(
                cmd,
                input=input_str,
                text=True,
                capture_output=True,
                cwd=workdir
            )

            lines = []
            with open(index_txt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    lines.append("" if not line else line + " 0")
            with open(index_txt_path, "w") as f:
                f.write("\n".join(lines) + "\n")

            print(f"  ├─ [Index] M={maxm0}, efC={efc} → build complete ✅")

        print("\n[Query] Running hybrid query tests...")
        nhq_index_root = os.path.join(NHQ_path, "NHQ_index")

        query_fvecs = os.path.join(original_data_path, "sift_query.fvecs")
        query_label = os.path.join(mid_path, "query_label_NHQ_anal.txt")
        gt_ivecs = os.path.join(mid_path, "gt.ivecs")

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
            workdir = "/home/mintaek/hybrid_index/methods/NHQ"

            result = subprocess.run(
                cmd,
                input=input_str,
                text=True,
                capture_output=True,
                cwd=workdir
            )

            output = result.stderr
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
                NHQ_trade_off[d_name][(M, ef)] = {
                    "qps": 10000.0 / float(search_time),
                    "avg_recall": float(accuracy)
                }
                print(f"  ├─ [Result] M={M}, ef={ef} | Time={search_time}s | Acc={accuracy}")

    # --- 결과 요약 ---
    post_score = pc.final_score(NHQ_trade_off["closer_to_post"])
    pre_score = pc.final_score(NHQ_trade_off["closer_to_pre"])

    print("\n[Final Scores]")
    print(f"  ├─ Post-base Score : {post_score:.6f}")
    print(f"  └─ Pre-base  Score : {pre_score:.6f}")

    if post_score > pre_score:
        print("[Decision] → ✅ post_base selected")
        return "post_base"
    else:
        print("[Decision] → ✅ pre_base selected")
        return "pre_base"
