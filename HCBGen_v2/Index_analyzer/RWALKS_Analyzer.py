from tqdm import tqdm
import time
import numpy as np
import random
import hnswlib
from tqdm import tqdm
import time
import json
import os
import re
import h5py
import subprocess
import matplotlib.pyplot as plt
from Index_analyzer import pareto_comp as pc


def Analyze(mode):
    NUM_RE = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    if mode == True:
        return "post_base"
    def load_rwalks_tradeoff_txt(path: str):
        """
        'Search Param | QPS | Recall' 형식의 txt를 읽어
        {(a,b): {'qps': float, 'avg_recall': float}, ...} 딕셔너리로 반환.
        """
        tradeoff = {}
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if not s or s.lower().startswith('search param') or set(s) <= set('-| '):
                    continue
                if '|' not in s:
                    continue

                cols = [c.strip() for c in s.split('|')]
                if len(cols) < 3:
                    continue

                param_str, qps_str, rec_str = cols[0], cols[1], cols[2]
                m = re.search(r'\((\d+)\s*,\s*(\d+)\)', param_str)
                if not m:
                    m = re.search(r'(\d+)\s*,\s*(\d+)', param_str)
                if not m:
                    continue
                a, b = int(m.group(1)), int(m.group(2))
                mq = re.search(NUM_RE, qps_str)
                mr = re.search(NUM_RE, rec_str)
                if not mq or not mr:
                    continue
                qps = float(mq.group(0))
                avg_recall = float(mr.group(0))
                tradeoff[(a, b)] = {'qps': qps, 'avg_recall': avg_recall}
        return tradeoff

    dataset_name = "sift1m"
    Rwalks_trade_off = {}

    for d_name, num_attribute, cardinality, distribution in zip(
        ["closer_to_post", "closer_to_pre"],
        [3, 10],
        ([6] * 3, [6] * 10),
        ["random", "zipf"]
    ):
        print(f"\n[Dataset] Start → {d_name}")
        cardi = '_'.join(str(c) for c in cardinality)
        dataset_path = f"/home/mintaek/hybrid_index/Benchmark/test_dataset/{dataset_name}_A{num_attribute}_{cardi}_{distribution}"
        DATA_DIR = os.path.join(dataset_path, "hardness_format")

        vectors_file = f"{DATA_DIR}/vectors.npy"
        payloads_file = f"{DATA_DIR}/payloads.jsonl"
        tests_file = f"{DATA_DIR}/tests.jsonl"

        # ─────────────────────────────────────────────
        # [Data Loading]
        # ─────────────────────────────────────────────
        print("[Data] Loading dataset …")
        vectors = np.load(vectors_file)
        print(f"  ├─ Vectors shape : {vectors.shape}")

        payloads = [json.loads(line) for line in open(payloads_file, "r")]
        tests = [json.loads(line) for line in open(tests_file, "r")]
        print(f"  ├─ Payloads loaded : {len(payloads):,}")
        print(f"  └─ Tests loaded    : {len(tests):,}")

        query_vectors = np.array([test["query"] for test in tests])
        gt_lists = [test["closest_ids"] for test in tests]
        maxlen = max(len(x) for x in gt_lists)
        repeated = np.empty((len(gt_lists), maxlen), dtype=np.int32)
        for i, arr in enumerate(gt_lists):
            repeated[i, :] = np.resize(arr, maxlen)
        ground_truth = repeated

        path = os.path.join(dataset_path, "mid_format/mapping.json")
        with open(path, "r") as f:
            mapping = json.load(f)
        cardinality = len(mapping)
        print(f"[Mapping] Cardinality = {cardinality}")

        # ─────────────────────────────────────────────
        # [Payload → Multi-hot Encoding]
        # ─────────────────────────────────────────────
        print("[Transform] Building multi-hot vectors …")
        multi_hot_vectors = []
        for payload in tqdm(payloads, desc="  ├─ Encoding base payloads"):
            mh_vec = np.zeros(cardinality, dtype=np.int32)
            for k, v in payload.items():
                key = f"{k}:{v}"
                idx = mapping.get(key)
                if idx is not None:
                    mh_vec[int(idx) - 1] = 1
            multi_hot_vectors.append(mh_vec)
        multi_hot_vectors = np.stack(multi_hot_vectors)

        def parse_conditions(cond_dict):
            conditions = []
            if "and" in cond_dict:
                for cond in cond_dict["and"]:
                    for attr, value_dict in cond.items():
                        v = value_dict.get("match", {}).get("value")
                        if v is not None:
                            conditions.append((attr, v))
            return conditions

        print("  └─ Building query one-hot vectors …")
        onehot_vectors = []
        for test in tests:
            oh_vec = np.zeros(cardinality, dtype=np.int32)
            attr_value_pairs = parse_conditions(test["conditions"])
            for attr, value in attr_value_pairs:
                key = f"{attr}:{value}"
                idx = mapping.get(key)
                if idx is not None:
                    oh_vec[int(idx) - 1] = 1
            onehot_vectors.append(oh_vec)
        onehot_vectors = np.stack(onehot_vectors).astype(np.int8)

        # ─────────────────────────────────────────────
        # [RWalks File Generation]
        # ─────────────────────────────────────────────
        rwalks_path = os.path.join(dataset_path, "rwalks_format")
        os.makedirs(rwalks_path, exist_ok=True)
        fname = os.path.join(rwalks_path, "all.h5")

        with h5py.File(fname, "w") as f:
            f.create_dataset("train_vectors", data=vectors)
            f.create_dataset("test_vectors", data=query_vectors)
            f.create_dataset("train_attr_vectors", data=multi_hot_vectors)
            f.create_dataset("test_attr_vectors", data=onehot_vectors)
            f.create_dataset("neighbors", data=ground_truth)

        print(f"[File] Saved RWalks data → {fname}")

        # ─────────────────────────────────────────────
        # [RWalks Execution]
        # ─────────────────────────────────────────────
        print("[Run] Launching RWalks evaluation …")
        data_dir = fname
        output_file = os.path.join(
            "/home/mintaek/hybrid_index/Generator/Index_analyzer/rwalks_dump",
            f"{d_name}_search_results.txt"
        )
        cmd = ["bash", "/home/mintaek/hybrid_index/methods/RWalks/sift_test_anal.sh", data_dir, output_file]
        cwd = "/home/mintaek/hybrid_index/methods/RWalks"

        subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("  └─ RWalks search complete ✅")

        # ─────────────────────────────────────────────
        # [Results Parsing]
        # ─────────────────────────────────────────────
        print("[Parse] Loading RWalks result summary …")
        Rwalks_trade_off[d_name] = load_rwalks_tradeoff_txt(output_file)
        for (a, b), vals in Rwalks_trade_off[d_name].items():
            print(f"  ├─ Param({a},{b}) → QPS={vals['qps']:,.2f}, Recall={vals['avg_recall']:.4f}")
        print(f"  └─ Results parsed for {d_name}")

    # ─────────────────────────────────────────────
    # [Final Scores]
    # ─────────────────────────────────────────────
    post_score = pc.final_score(Rwalks_trade_off["closer_to_post"])
    pre_score = pc.final_score(Rwalks_trade_off["closer_to_pre"])

    print("\n[Final Scores]")
    print(f"  ├─ Post-base Score : {post_score:.6f}")
    print(f"  └─ Pre-base  Score : {pre_score:.6f}")

    # ─────────────────────────────────────────────
    # [Decision]
    # ─────────────────────────────────────────────
    if post_score > pre_score:
        print("[Decision] → ✅ post_base selected")
        return "post_base"
    else:
        print("[Decision] → ✅ pre_base selected")
        return "pre_base"
