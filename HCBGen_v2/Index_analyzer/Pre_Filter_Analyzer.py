from tqdm import tqdm
import time
import numpy as np
import random
import hnswlib
import json
import os
import struct
from Index_analyzer import pareto_comp as pc


def Analyze(mode):
    if mode == True:
        return "pre_base"
    # ─────────────────────────────────────────────
    # [Helper Functions]
    # ─────────────────────────────────────────────
    def read_fvecs(filename):
        with open(filename, "rb") as f:
            data = []
            while True:
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                dim = struct.unpack('i', dim_bytes)[0]
                vec = np.frombuffer(f.read(4 * dim), dtype=np.float32)
                data.append(vec)
            return np.vstack(data)

    def load_groundtruth_bin(filename, nq):
        record_dtype = np.dtype([("idx", np.uint32), ("dist", np.float32)])
        file_size = os.path.getsize(filename)
        record_size = record_dtype.itemsize
        total_recs = file_size // record_size

        if nq == 0 or total_recs % nq != 0:
            raise ValueError(f"GT file size mismatch: total_recs={total_recs}, nq={nq}")

        K_gt = total_recs // nq
        data = np.fromfile(filename, dtype=record_dtype, count=total_recs)
        gt_indices = data["idx"].reshape(nq, K_gt)
        return gt_indices

    def pre_filtering(base_vectors, base_labels, query_vectors, query_labels, K):
        results = []
        filtered_count = []
        for query_vector, query_label in tqdm(zip(query_vectors, query_labels), total=len(query_labels), desc="  ├─ Pre-filtering queries"):
            # label-based filtering
            filtered_ids = [i for i, bl in enumerate(base_labels) if query_label.issubset(bl)]

            if not filtered_ids:
                results.append([])  # no candidates
                continue

            filtered_count.append(len(filtered_ids))

            # distance computation (L2)
            filtered_vectors = base_vectors[filtered_ids]
            dists = np.linalg.norm(filtered_vectors - query_vector, axis=1)

            # top-K selection
            topk_idx = np.argsort(dists)[:K]
            topk_global_ids = [filtered_ids[i] for i in topk_idx]
            results.append(topk_global_ids)

        avg_filtered = sum(filtered_count) / len(filtered_count) if filtered_count else 0
        return results, avg_filtered

    TRASH = 4294967295

    def recall_at_k(retrieved, gt, k):
        def _to_list(x):
            if isinstance(x, (set, tuple)):
                return list(x)
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x

        def _single_recall(r, g):
            r = _to_list(r) if r is not None else []
            g = _to_list(g) if g is not None else []
            filtered_gt = [x for x in g if x != TRASH]
            if not filtered_gt:
                return 1.0
            return len(set(r) & set(filtered_gt)) / min(len(filtered_gt), k)

        is_batch = (
            isinstance(retrieved, (list, tuple, np.ndarray)) and
            len(retrieved) > 0 and
            isinstance(retrieved[0], (list, tuple, set, np.ndarray))
        )

        if is_batch:
            recalls = [_single_recall(r_i, g_i) for r_i, g_i in zip(retrieved, gt)]
            return sum(recalls) / len(recalls) if recalls else 0.0
        else:
            return _single_recall(retrieved, gt)

    # ─────────────────────────────────────────────
    # [Dataset Setup]
    # ─────────────────────────────────────────────
    dataset_name_1 = "sift1m"
    dataset_name_list = ["closer_to_post", "closer_to_pre"]
    original_data_path = f"/home/mintaek/hybrid_index/Benchmark/{dataset_name_1}_original"

    Pre_filter_trade_off = {}

    for dataset_name in dataset_name_list:
        print(f"\n[Dataset] Start → {dataset_name}")

        if dataset_name == "closer_to_post":
            num_attribute, cardinality, distribution = 3, [6]*3, "random"
        else:
            num_attribute, cardinality, distribution = 10, [6]*10, "zipf"

        cardi = '_'.join(str(c) for c in cardinality)
        dataset_path = f"/home/mintaek/hybrid_index/Benchmark/test_dataset/{dataset_name_1}_A{num_attribute}_{cardi}_{distribution}"

        DATA_DIR = os.path.join(dataset_path, "hardness_format")
        mid_format = os.path.join(dataset_path, "mid_format")
        query_fvecs_path = os.path.join(original_data_path, "sift_query.fvecs")

        # ─────────────────────────────────────────────
        # [Data Loading]
        # ─────────────────────────────────────────────
        print("[Data] Loading dataset …")

        vectors = np.load(f"{DATA_DIR}/vectors.npy")
        print(f"  ├─ Base vectors shape : {vectors.shape}")

        base_labels = []
        with open(os.path.join(mid_format, "base_label.txt"), "r", encoding="utf-8") as f:
            for line in f:
                nums = [int(x) for x in line.strip().split(",") if x.strip()]
                if nums:
                    base_labels.append(set(nums))
        print(f"  └─ Loaded base labels : {len(base_labels):,}")

        queries = read_fvecs(query_fvecs_path)
        print(f"[Query] Loaded query vectors : {queries.shape}")

        query_labels = []
        with open(os.path.join(mid_format, "query_label.txt"), "r", encoding="utf-8") as f:
            for line in f:
                nums = [int(x) for x in line.strip().split(",") if x.strip()]
                if nums:
                    query_labels.append(set(nums))
        print(f"  └─ Loaded query labels : {len(query_labels):,}")

        gt = load_groundtruth_bin(os.path.join(mid_format, "gt.bin"), 10000)
        print(f"[Ground Truth] Loaded shape : {gt.shape}")

        # ─────────────────────────────────────────────
        # [Pre-filtering Evaluation]
        # ─────────────────────────────────────────────
        print("[Evaluation] Running pre-filtering search …")
        t1 = time.time()
        result, avg_filtered_ids = pre_filtering(vectors, base_labels, queries, query_labels, K=10)
        t2 = time.time()

        recall = recall_at_k(result, gt, 10)
        qps = 10000.0 / (t2 - t1)
        print(f"  ├─ Avg. filtered IDs : {avg_filtered_ids:,.2f}")
        print(f"  ├─ Avg. Recall@10    : {recall:.4f}")
        print(f"  └─ QPS               : {qps:,.2f}")

        Pre_filter_trade_off[dataset_name] = {
            1: {'qps': qps, 'avg_recall': recall}
        }

    # ─────────────────────────────────────────────
    # [Final Scores]
    # ─────────────────────────────────────────────
    post_score = pc.final_score(Pre_filter_trade_off["closer_to_post"])
    pre_score = pc.final_score(Pre_filter_trade_off["closer_to_pre"])

    print("\n[Final Scores]")
    print(f"  ├─ Post-base Score : {post_score:.6f}")
    print(f"  └─ Pre-base  Score : {pre_score:.6f}")

    # ─────────────────────────────────────────────
    # [Decision]
    # ─────────────────────────────────────────────
    if post_score > pre_score:
        print("[Decision] → ✅ post_base selected")
    else:
        print("[Decision] → ✅ pre_base selected")
