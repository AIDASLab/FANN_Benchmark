#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
from functools import lru_cache
from typing import List, Dict, Optional

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


# =========================================================
# Helpers: parsing
# =========================================================
def parse_int_list(s: str) -> List[int]:
    """
    "6,6,6" -> [6,6,6]
    """
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]

def parse_float_list(s: str) -> List[float]:
    """
    "0.5,0.0,0.8" -> [0.5,0.0,0.8]
    """
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


# =========================================================
# Zipf sampler (truncated without clipping spikes)
# =========================================================
@lru_cache(maxsize=None)
def _truncated_zipf_pmf(card: int, a: float) -> np.ndarray:
    k = np.arange(1, card + 1, dtype=np.float64)
    p = 1.0 / np.power(k, a)
    p /= p.sum()
    return p.astype(np.float64)

def _sample_labels(rng, distribution: str, card: int, size: int, zipf_param: float) -> np.ndarray:
    if distribution == "zipf":
        p = _truncated_zipf_pmf(card, float(zipf_param))
        return rng.choice(np.arange(1, card + 1, dtype=np.int32), size=size, p=p)
    elif distribution in ("uniform", "random"):
        return rng.integers(1, card + 1, size=size, dtype=np.int32)
    else:
        raise ValueError(f"unsupported distribution: {distribution}")

def _permute_by_correlation_keep_multiset(
    rng,
    labels: np.ndarray,        # (N,) int32 includes -1
    present_mask: np.ndarray,  # (N,) bool
    cluster_ids: np.ndarray,   # (N,) int
    corr: float                # [0,1]
) -> np.ndarray:
    if corr <= 0:
        return labels

    out = labels.copy()

    idx_present = np.flatnonzero(present_mask)
    if idx_present.size == 0:
        return out

    present_labels = labels[idx_present]

    # cluster size ranks
    cl = cluster_ids[idx_present]
    uniq_cl, inv_cl = np.unique(cl, return_inverse=True)
    counts_cl = np.bincount(inv_cl, minlength=uniq_cl.size)
    order_cl_by_size = np.argsort(-counts_cl, kind="mergesort")

    rank_table = np.empty_like(order_cl_by_size)
    rank_table[order_cl_by_size] = np.arange(order_cl_by_size.size)
    cl_size_rank = rank_table[inv_cl].astype(np.float32)
    denom = max(order_cl_by_size.size, 1)
    cl_rank_norm = cl_size_rank / denom

    # label multiset sorted by frequency desc, value asc
    uniq_lab, counts_lab = np.unique(present_labels, return_counts=True)
    order_lab = np.lexsort((uniq_lab, -counts_lab))
    lab_vals_sorted = uniq_lab[order_lab]
    lab_cnts_sorted = counts_lab[order_lab]
    labels_sorted = np.repeat(lab_vals_sorted, lab_cnts_sorted)

    # blend corr with noise for stable sorting
    noise = rng.random(idx_present.size, dtype=np.float32)
    blended = (1.0 - corr) * noise + corr * (cl_rank_norm + 1e-6 * noise)

    order_idx = np.argsort(blended, kind="mergesort")
    target_indices = idx_present[order_idx]

    out[target_indices] = labels_sorted
    return out


# =========================================================
# Payload generation
# =========================================================
def generate_attribute_payloads(
    num_vectors: int,
    num_attributes: int,
    cardinalities: List[int],
    missing_prob: List[float],
    base_vectors: Optional[np.ndarray] = None,
    correlations: Optional[List[float]] = None,   # [0,1]
    distribution: str = "zipf",
    zipf_param: float = 1.5,
    missing_value: int = -1,
    seed: int = 42,
) -> np.ndarray:
    assert len(cardinalities) == num_attributes, "cardinalities length must match num_attributes"
    assert len(missing_prob) == num_attributes, "missing_prob length must match num_attributes"

    rng_root = np.random.default_rng(seed)

    # base vectors
    if base_vectors is None:
        base_vectors = rng_root.normal(size=(num_vectors, 16)).astype(np.float32)
    X = normalize(base_vectors.astype(np.float32))

    # 1) raw labels + missing
    payloads = np.full((num_vectors, num_attributes), missing_value, dtype=np.int32)
    present_masks = np.zeros((num_vectors, num_attributes), dtype=bool)

    for a in range(num_attributes):
        rng = np.random.default_rng(seed + 1000 + a)
        card = int(cardinalities[a])

        raw = _sample_labels(rng, distribution, card, size=num_vectors, zipf_param=zipf_param)
        present = (rng.random(num_vectors) > float(missing_prob[a]))
        payloads[present, a] = raw[present]
        present_masks[:, a] = present

    # 2) fix all-missing rows
    all_missing = np.all(~present_masks, axis=1)
    if np.any(all_missing):
        rows = np.flatnonzero(all_missing)
        cols = rng_root.integers(0, num_attributes, size=rows.size)
        for r, c in zip(rows, cols):
            rng = np.random.default_rng(seed + 2000 + int(c))
            card = int(cardinalities[int(c)])
            val = _sample_labels(rng, distribution, card, size=1, zipf_param=zipf_param)[0]
            payloads[int(r), int(c)] = int(val)
            present_masks[int(r), int(c)] = True

    # 3) correlation: clustering + multiset-preserving permutation
    if correlations is None:
        correlations = [0.0] * num_attributes
    assert len(correlations) == num_attributes, "correlations length must match num_attributes"

    for a in range(num_attributes):
        corr = float(correlations[a])
        if corr <= 0:
            continue

        card = int(cardinalities[a])
        rng = np.random.default_rng(seed + 3000 + a)

        km = MiniBatchKMeans(
            n_clusters=card,
            random_state=seed + 4000 + a,
            batch_size=10000,
        )
        cluster_ids = km.fit_predict(X)

        col = payloads[:, a]
        present_mask = (col != missing_value)
        col_new = _permute_by_correlation_keep_multiset(
            rng=rng,
            labels=col,
            present_mask=present_mask,
            cluster_ids=cluster_ids,
            corr=corr,
        )
        payloads[:, a] = col_new

    return payloads


def generate_query_payloads(
    num_vectors: int,
    num_attributes: int,
    cardinalities: List[int],
    missing_prob: List[float],
    distribution: str = "zipf",
    zipf_param: float = 1.5,
    missing_value: int = -1,
    seed: int = 123,
) -> np.ndarray:
    assert len(cardinalities) == num_attributes
    assert len(missing_prob) == num_attributes

    rng = np.random.default_rng(seed)
    payloads = np.full((num_vectors, num_attributes), missing_value, dtype=np.int32)

    for i in range(num_attributes):
        card = int(cardinalities[i])
        present_mask = (rng.random(num_vectors) > float(missing_prob[i]))

        if distribution == "zipf":
            # NOTE: query는 기존 코드 유지 (zipf -> clip)
            raw = rng.zipf(zipf_param, size=num_vectors)
            raw = np.clip(raw, 1, card).astype(np.int32)
        elif distribution == "random":
            raw = rng.integers(1, card + 1, size=num_vectors, dtype=np.int32)
        else:
            raise ValueError(f"unsupported distribution: {distribution}")

        payloads[present_mask, i] = raw[present_mask]

    # fix all-missing
    all_missing = np.all(payloads == missing_value, axis=1)
    if np.any(all_missing):
        rows = np.flatnonzero(all_missing)
        cols = rng.integers(0, num_attributes, size=rows.size)
        for r, c in zip(rows, cols):
            card = int(cardinalities[int(c)])
            if distribution == "zipf":
                val = int(np.clip(rng.zipf(zipf_param), 1, card))
            else:
                val = int(rng.integers(1, card + 1))
            payloads[int(r), int(c)] = val

    return payloads


def payloads_to_dicts(payloads: np.ndarray, prefix: str = "label", include_missing: bool = True) -> List[Dict[str, int]]:
    n, a = payloads.shape
    out: List[Dict[str, int]] = []
    for i in range(n):
        d: Dict[str, int] = {}
        for j in range(a):
            v = int(payloads[i, j])
            if include_missing or v != -1:
                d[f"{prefix}_{j+1}"] = v
        out.append(d)
    return out


# =========================================================
# Mapping + label text for UNG tools
# =========================================================
def build_label_mapping(filter_dict_list: List[Dict[str, int]], mapping_json_path: str) -> str:
    mapping = {}
    next_id = 1

    for temp in filter_dict_list:
        for key, value in temp.items():
            tup = (key, str(value))
            if tup not in mapping:
                mapping[tup] = next_id
                next_id += 1

    str_mapping = {f"{name}:{val}": idx for (name, val), idx in mapping.items()}

    with open(mapping_json_path, "w", encoding="utf-8") as out:
        json.dump(str_mapping, out, ensure_ascii=False, indent=2)

    print(f"[Saved] mapping.json: {mapping_json_path} (size={len(str_mapping)}, max_id={next_id-1})")
    return mapping_json_path


def save_vector_label(vector_payload: List[Dict], mapping_json_path: str, output_txt_path: str) -> None:
    with open(mapping_json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    with open(output_txt_path, "w", encoding="utf-8") as fout:
        for i, payload in enumerate(vector_payload):
            ids = []
            for key, value in payload.items():
                map_key = f"{key}:{value}"
                if map_key not in mapping:
                    raise ValueError(f"[Line {i}] mapping.json missing key: {map_key}")
                ids.append(str(mapping[map_key]))
            fout.write(",".join(ids) + "\n")

    print(f"[Saved] label txt: {output_txt_path}")


# =========================================================
# I/O
# =========================================================
def read_fvecs(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype=np.int32)
    dim = int(a[0])
    if (a.size % (dim + 1)) != 0:
        raise RuntimeError("fvecs file corrupted or wrong dimension")
    return a.reshape(-1, dim + 1)[:, 1:].view("float32")


# =========================================================
# UNG tools wrapper
# =========================================================
def run_fvecs_to_bin(tool_path: str, input_fvecs: str, output_bin: str, data_type: str = "float") -> None:
    cmd = [tool_path, "--data_type", data_type, "--input_file", input_fvecs, "--output_file", output_bin]
    print("[CMD] " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def save_compute_groundtruth(
    compute_gt_tool: str,
    base_label_file: str,
    query_label_file: str,
    gt_file: str,
    query_bin_file: str,
    base_bin_file: str,
    data_type: str = "float",
    dist_fn: str = "L2",
    scenario: str = "containment",
    K: int = 10,
    num_threads: int = 200,
    quiet: bool = True,
) -> None:
    cmd = [
        compute_gt_tool,
        "--data_type", str(data_type),
        "--dist_fn", str(dist_fn),
        "--scenario", str(scenario),
        "--K", str(K),
        "--num_threads", str(num_threads),
        "--base_bin_file", str(base_bin_file),
        "--base_label_file", str(base_label_file),
        "--query_bin_file", str(query_bin_file),
        "--query_label_file", str(query_label_file),
        "--gt_file", str(gt_file),
    ]
    print("[CMD] " + " ".join(cmd))
    if quiet:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(cmd, check=True)

def load_groundtruth_bin(filename: str, nq: int) -> np.ndarray:
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


# =========================================================
# Main pipeline
# =========================================================
def main():
    p = argparse.ArgumentParser("Generate benchmark (payloads + tests + GT)")

    # paths
    p.add_argument("--benchmark_root", type=str, required=True,
                   help="Output root path, e.g. /home/ec2-user/hybrid_hardness/Benchmark")
    p.add_argument("--original_root", type=str, required=True,
                   help="Original data root containing <dataset_name>_original/ files")
    p.add_argument("--dataset_name", type=str, required=True, choices=["sift1m", "gist1m", "glove1m"],
                   help="Dataset name (must match *_original directory and fvecs filenames)")
    p.add_argument("--ung_build_dir", type=str, required=True,
                   help="UNG build dir that contains tools/fvecs_to_bin and tools/compute_groundtruth")

    # params
    p.add_argument("--num_attribute", type=int, required=True)
    p.add_argument("--cardinality", type=str, required=True,
                   help='Comma-separated list, len=num_attribute. e.g. "6,6,6" or "1,1,1"')
    p.add_argument("--base_distribution", type=str, default="zipf", choices=["zipf", "random", "uniform"])
    p.add_argument("--query_distribution", type=str, default="zipf", choices=["zipf", "random"])
    p.add_argument("--base_missing_prob", type=str, required=True,
                   help='Comma-separated float list len=num_attribute. e.g. "0.5,0.5,0.5"')
    p.add_argument("--query_missing_prob", type=str, required=True,
                   help='Comma-separated float list len=num_attribute. e.g. "0.9,0.9,0.9"')
    p.add_argument("--correlation", type=str, required=True,
                   help='Comma-separated float list len=num_attribute. e.g. "0.0,0.5,1.0"')

    # other
    p.add_argument("--zipf_param", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--missing_value", type=int, default=-1)
    p.add_argument("--num_queries_out", type=int, default=10000,
                   help="Number of valid query labels to output (same as your while-loop target)")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--num_threads", type=int, default=200)
    p.add_argument("--quiet_tools", action="store_true", help="Silence UNG tools stdout/stderr")

    args = p.parse_args()

    # parse lists
    cardinality = parse_int_list(args.cardinality)
    base_missing_prob = parse_float_list(args.base_missing_prob)
    query_missing_prob = parse_float_list(args.query_missing_prob)
    correlation = parse_float_list(args.correlation)

    if len(cardinality) != args.num_attribute:
        raise ValueError(f"len(cardinality)={len(cardinality)} must equal num_attribute={args.num_attribute}")
    if len(base_missing_prob) != args.num_attribute:
        raise ValueError(f"len(base_missing_prob)={len(base_missing_prob)} must equal num_attribute={args.num_attribute}")
    if len(query_missing_prob) != args.num_attribute:
        raise ValueError(f"len(query_missing_prob)={len(query_missing_prob)} must equal num_attribute={args.num_attribute}")
    if len(correlation) != args.num_attribute:
        raise ValueError(f"len(correlation)={len(correlation)} must equal num_attribute={args.num_attribute}")

    # derive paths
    original_data_path = os.path.join(args.original_root, f"{args.dataset_name}_original")
    base_fvecs = os.path.join(original_data_path, f"{args.dataset_name}_base.fvecs")
    query_fvecs = os.path.join(original_data_path, f"{args.dataset_name}_query.fvecs")

    if not os.path.exists(base_fvecs) or not os.path.exists(query_fvecs):
        raise FileNotFoundError(f"Missing fvecs: {base_fvecs} or {query_fvecs}")

    cardi = "_".join(str(c) for c in cardinality)
    missing = "_".join(str(x) for x in base_missing_prob)
    corr = "_".join(str(x) for x in correlation)

    target_path = os.path.join(
        args.benchmark_root,
        f"{args.dataset_name}_A{args.num_attribute}_{cardi}_{args.base_distribution}_{missing}_{corr}",
    )

    hardness_path = os.path.join(target_path, "hardness_format")
    mid_path = os.path.join(target_path, "mid_format")
    os.makedirs(hardness_path, exist_ok=True)
    os.makedirs(mid_path, exist_ok=True)

    # info.txt
    content = (
        f"num_attribute = {args.num_attribute}\n"
        f"cardinality = {cardinality}\n"
        f"base_distribution = {args.base_distribution}\n"
        f"query_distribution = {args.query_distribution}\n"
        f"correlation = {correlation}\n"
        f"base_missing_prob = {base_missing_prob}\n"
        f"query_missing_prob = {query_missing_prob}\n"
        f"zipf_param = {args.zipf_param}\n"
        f"seed = {args.seed}\n"
    )
    with open(os.path.join(target_path, "info.txt"), "w") as f:
        f.write(content)

    # load vectors
    base_vec = read_fvecs(base_fvecs)
    query_vec = read_fvecs(query_fvecs)

    # save vectors.npy for hardness format (your original behavior)
    np.save(os.path.join(hardness_path, "vectors.npy"), base_vec)

    print(f"[Load] base={len(base_vec)}, query={len(query_vec)}")
    print("[Base Payload] generating ...")

    base_label_arr = generate_attribute_payloads(
        num_vectors=len(base_vec),
        num_attributes=args.num_attribute,
        cardinalities=cardinality,
        missing_prob=base_missing_prob,
        base_vectors=base_vec,
        correlations=correlation,
        distribution=args.base_distribution,
        zipf_param=args.zipf_param,
        missing_value=args.missing_value,
        seed=args.seed,
    )
    print("[Base Payload] done")

    base_label_payload = payloads_to_dicts(base_label_arr, include_missing=True)
    base_label_payload_UNG = payloads_to_dicts(base_label_arr, include_missing=False)

    # save payload jsonl
    with open(os.path.join(hardness_path, "payloads.jsonl"), "w", encoding="utf-8") as f:
        for payload in base_label_payload:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    with open(os.path.join(hardness_path, "payloads_UNG.jsonl"), "w", encoding="utf-8") as f:
        for payload in base_label_payload_UNG:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"[Saved] base payloads: {hardness_path}")

    # mapping
    mapping_path = build_label_mapping(base_label_payload, os.path.join(mid_path, "mapping.json"))

    # UNG tool paths
    fvecs_to_bin_tool = os.path.join(args.ung_build_dir, "tools", "fvecs_to_bin")
    compute_gt_tool = os.path.join(args.ung_build_dir, "tools", "compute_groundtruth")
    if not os.path.exists(fvecs_to_bin_tool) or not os.path.exists(compute_gt_tool):
        raise FileNotFoundError(f"Missing UNG tools under: {args.ung_build_dir}/tools")

    # convert fvecs -> bin
    base_bin = os.path.join(mid_path, "base_vector.bin")
    query_bin = os.path.join(mid_path, "query_vector.bin")

    run_fvecs_to_bin(fvecs_to_bin_tool, base_fvecs, base_bin, data_type="float")
    run_fvecs_to_bin(fvecs_to_bin_tool, query_fvecs, query_bin, data_type="float")

    # save labels as txt
    base_label_txt = os.path.join(mid_path, "base_label.txt")
    base_label_txt_ung = os.path.join(mid_path, "base_label_UNG.txt")
    save_vector_label(base_label_payload, mapping_path, base_label_txt)
    save_vector_label(base_label_payload_UNG, mapping_path, base_label_txt_ung)

    # =====================================================
    # Query label generation with validity filtering via GT
    # =====================================================
    target_value = 4294967295
    valid_query_labels: List[Dict] = []

    it = 0
    nq_total = len(query_vec)

    while len(valid_query_labels) < args.num_queries_out:
        it += 1
        print(f"[Query Payload] iter={it} current_valid={len(valid_query_labels)}")

        # generate candidate query payloads (batch size = num_queries_out)
        query_label_arr = generate_query_payloads(
            num_vectors=args.num_queries_out,
            num_attributes=args.num_attribute,
            cardinalities=cardinality,
            missing_prob=query_missing_prob,
            distribution=args.query_distribution,
            zipf_param=args.zipf_param,
            missing_value=args.missing_value,
            seed=args.seed + 999 + it,
        )

        query_label_payload = payloads_to_dicts(query_label_arr, include_missing=False)

        # write temporary query_label.txt for GT tool
        tmp_query_label_txt = os.path.join(mid_path, "query_label.txt")
        save_vector_label(query_label_payload, mapping_path, tmp_query_label_txt)

        # compute GT
        gt_file = os.path.join(mid_path, "gt.bin")
        save_compute_groundtruth(
            compute_gt_tool=compute_gt_tool,
            base_label_file=base_label_txt,
            query_label_file=tmp_query_label_txt,
            gt_file=gt_file,
            query_bin_file=query_bin,
            base_bin_file=base_bin,
            data_type="float",
            dist_fn="L2",
            scenario="containment",
            K=args.K,
            num_threads=args.num_threads,
            quiet=args.quiet_tools,
        )

        gt_indices = load_groundtruth_bin(gt_file, nq_total).tolist()

        valid_indices = [
            i for i, sublist in enumerate(gt_indices)
            if not all(v == target_value for v in sublist)
        ]
        print(f"[Query Payload] valid_in_iter={len(valid_indices)}")

        filtered_query_labels = [query_label_payload[i] for i in valid_indices]
        valid_query_labels.extend(filtered_query_labels)

        if len(valid_query_labels) > args.num_queries_out:
            valid_query_labels = valid_query_labels[:args.num_queries_out]
            break

    print(f"[Query Payload] final_valid={len(valid_query_labels)}, iterations={it}")

    # finalize query_label.txt (valid only)
    query_label_txt = os.path.join(mid_path, "query_label.txt")
    save_vector_label(valid_query_labels, mapping_path, query_label_txt)

    # compute GT again for final query labels
    gt_file = os.path.join(mid_path, "gt.bin")
    save_compute_groundtruth(
        compute_gt_tool=compute_gt_tool,
        base_label_file=base_label_txt,
        query_label_file=query_label_txt,
        gt_file=gt_file,
        query_bin_file=query_bin,
        base_bin_file=base_bin,
        data_type="float",
        dist_fn="L2",
        scenario="containment",
        K=args.K,
        num_threads=args.num_threads,
        quiet=args.quiet_tools,
    )
    gt_indices = load_groundtruth_bin(gt_file, nq_total).tolist()

    # validity stats
    valid_indices = [
        i for i, sublist in enumerate(gt_indices)
        if not all(v == target_value for v in sublist)
    ]
    print(f"[GT] number of valid query labels: {len(valid_indices)}")

    # build conditions + tests.jsonl
    conditions = []
    for qlab in valid_query_labels:
        temp = {"and": []}
        for key, value in qlab.items():
            temp["and"].append({key: {"match": {"value": value}}})
        conditions.append({"and": temp["and"]})

    tests = []
    query_vec_list = query_vec.tolist()
    for query, cond, gt in zip(query_vec_list, conditions, gt_indices):
        tests.append({
            "query": query,
            "conditions": cond,
            "closest_ids": gt,
        })

    with open(os.path.join(hardness_path, "tests.jsonl"), "w", encoding="utf-8") as f:
        for item in tests:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[Saved] tests.jsonl: {os.path.join(hardness_path, 'tests.jsonl')}")

    # count.txt (number of valid ids per query)
    counts = []
    for t in tests:
        ids = t["closest_ids"]
        counts.append(sum(x != target_value for x in ids))

    with open(os.path.join(mid_path, "count.txt"), "w") as f:
        for c in counts:
            f.write(f"{c}\n")

    print(f"[Saved] count.txt: {os.path.join(mid_path, 'count.txt')}")
    print(f"[Done] dataset generated at: {target_path}")


if __name__ == "__main__":
    main()
