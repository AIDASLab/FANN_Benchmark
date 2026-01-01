import numpy as np
import json
from typing import List, Dict
import os
import subprocess
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from functools import lru_cache
import random

@lru_cache(maxsize=None)
def _truncated_zipf_pmf(card: int, a: float) -> np.ndarray:

    k = np.arange(1, card + 1, dtype=np.float64)
    p = 1.0 / np.power(k, a)
    p /= p.sum()
    return p.astype(np.float64)

def _sample_labels(rng, distribution, card, size, zipf_param):

    if distribution == 'zipf':
        p = _truncated_zipf_pmf(card, float(zipf_param))
        return rng.choice(np.arange(1, card + 1, dtype=np.int32), size=size, p=p)
    elif distribution in ('uniform', 'random'):
        return rng.integers(1, card + 1, size=size, dtype=np.int32)
    else:
        raise ValueError(f"unsupported distribution: {distribution}")

def _permute_by_correlation_keep_multiset(
    rng,
    labels,            # (N,) int32, includes -1 for missing; will be copied
    present_mask,      # (N,) bool
    cluster_ids,       # (N,) int
    corr               # float in [0,1]
):

    if corr <= 0:
        return labels

    out = labels.copy()

    idx_present = np.flatnonzero(present_mask)
    if idx_present.size == 0:
        return out

    # 1) present 라벨 멀티셋 확보
    present_labels = labels[idx_present]

    # 2) 클러스터 크기 기반 정렬 키 만들기
    cl = cluster_ids[idx_present]
    uniq_cl, inv_cl = np.unique(cl, return_inverse=True)

    # 각 클러스터의 present 샘플 수
    counts_cl = np.bincount(inv_cl, minlength=uniq_cl.size)
    # 큰 클러스터 먼저
    order_cl_by_size = np.argsort(-counts_cl, kind='mergesort')
    # 클러스터ID -> size-rank 매핑
    rank_table = np.empty_like(order_cl_by_size)
    rank_table[order_cl_by_size] = np.arange(order_cl_by_size.size)
    cl_size_rank = rank_table[inv_cl].astype(np.float32)
    denom = max(order_cl_by_size.size, 1)
    cl_rank_norm = cl_size_rank / denom

    # 3) 라벨 멀티셋을 "빈도 내림차순(+값 오름차순)" 블록으로 정렬
    uniq_lab, counts_lab = np.unique(present_labels, return_counts=True)
    order_lab = np.lexsort((uniq_lab, -counts_lab))  # (-count, value)
    lab_vals_sorted = uniq_lab[order_lab]
    lab_cnts_sorted = counts_lab[order_lab]
    labels_sorted = np.repeat(lab_vals_sorted, lab_cnts_sorted)

    # 4) corr로 클러스터 정렬과 랜덤 혼합
    noise = rng.random(idx_present.size, dtype=np.float32)
    blended = (1.0 - corr) * noise + corr * (cl_rank_norm + 1e-6 * noise)

    order_idx = np.argsort(blended, kind='mergesort')
    target_indices = idx_present[order_idx]

    # 5) 재배치
    out[target_indices] = labels_sorted
    return out

def generate_attribute_payloads(
    num_vectors,
    num_attributes,
    cardinalities,
    missing_prob,
    base_vectors=None,
    correlations=None,   # [0, 1] 범위
    distribution='zipf',
    zipf_param=1.5,
    missing_value=-1,
    # seed=42,
):

    assert len(cardinalities) == num_attributes, "length of cardinality should match attribute."
    rng_root = np.random.default_rng()

    # ----------- 입력 벡터 준비 -----------
    if base_vectors is None:
        base_vectors = rng_root.normal(size=(num_vectors, 16)).astype(np.float32)
    X = normalize(base_vectors.astype(np.float32))

    # ----------- 1) raw 생성(전역 분포) + missing 적용 -----------
    payloads = np.full((num_vectors, num_attributes), missing_value, dtype=np.int32)
    present_masks = np.zeros((num_vectors, num_attributes), dtype=bool)

    for a in range(num_attributes):
        rng = np.random.default_rng(1000 + a)
        card = cardinalities[a]
        # 전역 분포로 라벨 생성 (truncated Zipf)
        raw = _sample_labels(rng, distribution, card, size=num_vectors, zipf_param=zipf_param)
        # missing 적용
        present = (rng.random(num_vectors) > missing_prob[a])
        payloads[present, a] = raw[present]
        present_masks[:, a] = present

    # ----------- 2) 모든 attribute가 missing인 행 보정 -----------
    all_missing = np.all(~present_masks, axis=1)
    if np.any(all_missing):
        rows = np.flatnonzero(all_missing)
        cols = rng_root.integers(0, num_attributes, size=rows.size)
        for r, c in zip(rows, cols):
            rng = np.random.default_rng(2000 + c)
            card = cardinalities[c]
            val = _sample_labels(rng, distribution, card, size=1, zipf_param=zipf_param)[0]
            payloads[r, c] = val
            present_masks[r, c] = True

    # ----------- 3) corr 반영: 멀티셋 보존 재배치 -----------
    if correlations is None:
        correlations = np.zeros(num_attributes, dtype=np.float32)

    for a in range(num_attributes):
        corr = float(correlations[a])
        if corr <= 0:
            continue

        card = cardinalities[a]
        rng = np.random.default_rng(3000 + a)

        # (a) 이 attribute 전용 클러스터링: K=card
        km = MiniBatchKMeans(
            n_clusters=card,
            random_state=4000 + a,
            batch_size=10000
        )
        cluster_ids = km.fit_predict(X)

        # (b) 멀티셋 보존 재배치(결측 제외)
        col = payloads[:, a]
        present_mask = (col != missing_value)
        col_new = _permute_by_correlation_keep_multiset(
            rng=rng,
            labels=col,
            present_mask=present_mask,
            cluster_ids=cluster_ids,
            corr=corr
        )
        payloads[:, a] = col_new

    return payloads


# def generate_query_payloads(
#     num_vectors,
#     num_attributes,
#     cardinalities,
#     missing_prob,
#     distribution='random',
#     zipf_param=1.5,
#     missing_value=-1,
# ):
#     assert len(cardinalities) == num_attributes, "length of cardinality should match attribute."

#     payloads = np.full((num_vectors, num_attributes), missing_value, dtype=int)

#     for i in range(num_attributes):
#         card = cardinalities[i]
#         present_mask = np.random.rand(num_vectors) > missing_prob[i]

#         if distribution == 'zipf':
#             raw = np.random.zipf(zipf_param, size=num_vectors)
#             raw = np.clip(raw, 1, card)
#         elif distribution == 'random':
#             raw = np.random.randint(1, card + 1, size=num_vectors)
#         else:
#             raise ValueError(f"not supported distribution: {distribution}")

#         payloads[present_mask, i] = raw[present_mask]

#     # ✅ 모든 attribute가 missing인 벡터 보정
#     for idx in range(num_vectors):
#         if np.all(payloads[idx] == missing_value):
#             # 랜덤 attribute 하나 선택해서 채우기
#             i = np.random.randint(0, num_attributes)
#             card = cardinalities[i]
#             if distribution == 'zipf':
#                 val = np.clip(np.random.zipf(zipf_param), 1, card)
#             elif distribution == 'random':
#                 val = np.random.randint(1, card + 1)
#             payloads[idx, i] = val

#     return payloads

def generate_query_payloads(
    num_vectors,
    num_attributes,
    cardinalities,
    missing_prob,
    distribution='random',
    zipf_param=1.5,
    missing_value=-1,
    allowed_values_per_attr=None,
):
    """
    Query용 attribute payload 생성.

    - allowed_values_per_attr 가 None 이면:
        → 예전처럼 1 ~ cardinality[i] 범위에서 값 샘플링
    - allowed_values_per_attr 가 주어지면:
        → 각 attribute 별로 '실제로 base payload에서 등장했던 값들'만 사용

    Args:
        num_vectors: 생성할 query 개수
        num_attributes: attribute 개수
        cardinalities: 각 attribute의 cardinality (int 리스트)
        missing_prob: 각 attribute가 missing 될 확률 (float 리스트)
        distribution: 'random' 또는 'zipf'
        zipf_param: zipf 분포 파라미터
        missing_value: missing 값 (기본 -1)
        allowed_values_per_attr: 
            None 이거나, 길이 num_attributes 인 리스트.
            각 원소는 해당 attribute에서 허용되는 값들의 리스트/배열.
    """
    assert len(cardinalities) == num_attributes, "length of cardinality should match attribute."

    if allowed_values_per_attr is not None:
        assert len(allowed_values_per_attr) == num_attributes, \
            "allowed_values_per_attr length must match num_attributes"

    payloads = np.full((num_vectors, num_attributes), missing_value, dtype=int)

    for i in range(num_attributes):
        # 이 attribute에서 사용할 value 도메인 결정
        if allowed_values_per_attr is not None:
            values_i = np.array(allowed_values_per_attr[i], dtype=int)
            card = len(values_i)
            if card == 0:
                # 이 attribute는 실질적으로 값이 없는 경우 → 전부 missing으로 둔다.
                continue
        else:
            values_i = None
            card = cardinalities[i]

        present_mask = np.random.rand(num_vectors) > missing_prob[i]

        if distribution == 'zipf':
            raw = np.random.zipf(zipf_param, size=num_vectors)
            raw = np.clip(raw, 1, card)
            if values_i is not None:
                # 1..card → index로 보고 실제 값으로 매핑
                raw = values_i[raw - 1]
        elif distribution == 'random':
            if values_i is not None:
                idx = np.random.randint(0, card, size=num_vectors)
                raw = values_i[idx]
            else:
                raw = np.random.randint(1, card + 1, size=num_vectors)
        else:
            raise ValueError(f"not supported distribution: {distribution}")

        payloads[present_mask, i] = raw[present_mask]

    # ✅ 모든 attribute가 missing인 벡터 보정
    for idx in range(num_vectors):
        if np.all(payloads[idx] == missing_value):
            # 랜덤 attribute 하나 선택해서 채우기
            i = np.random.randint(0, num_attributes)

            if allowed_values_per_attr is not None:
                vals = np.array(allowed_values_per_attr[i], dtype=int)
                if len(vals) == 0:
                    # 이 attribute에도 값이 없으면 그냥 스킵 (전부 missing 허용)
                    continue
                val = np.random.choice(vals)
            else:
                card = cardinalities[i]
                if distribution == 'zipf':
                    val = np.clip(np.random.zipf(zipf_param), 1, card)
                elif distribution == 'random':
                    val = np.random.randint(1, card + 1)
            payloads[idx, i] = val

    return payloads


# def payloads_to_dicts(payloads, prefix="label", include_missing=True):
#     """
#     Convert attribute payloads to list of dicts.

#     Args:
#         payloads (np.ndarray): (N, A) array of attribute values
#         prefix (str): key prefix (default: "label")
#         include_missing (bool): whether to include -1 in output dicts

#     Returns:
#         List[Dict[str, int]]
#     """
#     num_vectors, num_attributes = payloads.shape
#     dict_list = []

#     for i in range(num_vectors):
#         d = {}
#         for j in range(num_attributes):
#             val = payloads[i, j]
#             if include_missing or val != -1:
#                 d[f"{prefix}_{j+1}"] = int(val)
#         dict_list.append(d)
#     return dict_list

# def payloads_to_dicts(payloads, prefix="label", include_missing=True):
    
#     num_vectors, num_attributes = payloads.shape
#     dict_list = []

#     for i in range(num_vectors):
#         d = {}
#         for j in range(num_attributes):
#             val = payloads[i, j]
#             if include_missing or val != -1:
#                 d[f"{prefix}_{j+1}"] = int(val)
#         dict_list.append(d)
#     return dict_list



def payloads_to_dicts(
    payloads,
    attr_keys=None,
    prefix="label",
    include_missing=True,
    missing_value=-1,
):
    """
    numpy payloads → list[dict] 로 변환.

    두 가지 모드 지원:
      1) attr_keys 가 None:
         - 기존처럼 key 이름을 prefix_{j+1} 으로 생성
      2) attr_keys 가 주어진 경우:
         - attr_keys[j] 를 key 로 사용 (예: 'label_7', 'color', ...)

    Args:
        payloads (np.ndarray): (N, A) array of attribute values
        attr_keys (List[str] or None): 각 attribute의 key 이름 리스트
        prefix (str): attr_keys 가 None 일 때 사용할 prefix
        include_missing (bool): missing_value 도 dict에 포함할지 여부
        missing_value (int): missing sentinel 값
    """
    num_vectors, num_attributes = payloads.shape

    if attr_keys is not None:
        assert len(attr_keys) == num_attributes, \
            "len(attr_keys) must match number of columns in payloads"

    dict_list = []

    for i in range(num_vectors):
        d = {}
        for j in range(num_attributes):
            val = int(payloads[i, j])
            if (val == missing_value) and not include_missing:
                continue

            if attr_keys is not None:
                key = attr_keys[j]
            else:
                key = f"{prefix}_{j+1}"

            d[key] = val
        dict_list.append(d)

    return dict_list

def load_target_hardness_distribution(
    json_path,
    num_bins,
    index_type,
    num_queries,
    bin_range=None,
):
    """
    target hardness JSON에서 Post_Hardness 분포를 읽어와서
    - bin_edges
    - bin_centers
    - target_counts (각 bin에 몇 개의 쿼리를 할당할지)
    를 계산한다.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vals = np.array([item["Post_Hardness"] for item in data], dtype=float)
    vals = vals[np.isfinite(vals)]

    if index_type == "pre_base":
        vals = -vals

    if vals.size == 0:
        raise RuntimeError(f"No valid Post_Hardness values in {json_path}")

    if bin_range is None:
        vmin = float(vals.min())
        vmax = float(vals.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6
        margin = 0.02 * (vmax - vmin)
        bin_min = vmin - margin
        bin_max = vmax + margin
    else:
        bin_min, bin_max = bin_range

    bin_edges = np.linspace(bin_min, bin_max, num_bins + 1)
    hist, _ = np.histogram(vals, bins=bin_edges)

    if hist.sum() == 0:
        raise RuntimeError("Target hardness histogram is empty.")

    # target_counts: 각 bin에 생성해야 하는 query 개수
    target_counts = np.round(hist / hist.sum() * num_queries).astype(int)

    # 합을 num_queries에 정확히 맞춰 조정
    diff = num_queries - int(target_counts.sum())
    if diff != 0:
        # hist가 큰 bin들부터 조정
        order = np.argsort(-hist)
        idx = 0
        while diff != 0 and idx < len(order):
            b = order[idx]
            if diff > 0:
                target_counts[b] += 1
                diff -= 1
            else:
                if target_counts[b] > 0:
                    target_counts[b] -= 1
                    diff += 1
            idx = (idx + 1) % len(order)

    # bin center
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    return bin_edges, bin_centers, target_counts




def sample_query_label_from_base(non_empty_payloads, attr_keys, query_missing_prob):
    """
    base_mode == 'load' 일 때, 실제 base payload에서
    쿼리 레이블을 하나 샘플링하는 헬퍼.

    - non_empty_payloads: 라벨이 1개 이상 있는 base payload 리스트
    - attr_keys: 전체 attribute key 리스트 (예: ['label_7', 'label_19', ...])
    - query_missing_prob: 각 attribute가 쿼리에서 missing될 확률 벡터
    """
    payload = random.choice(non_empty_payloads)

    q = {}
    for j, key in enumerate(attr_keys):
        if key not in payload:
            continue
        # 이 attribute를 쿼리에 포함시킬지 여부
        if np.random.rand() > query_missing_prob[j]:
            q[key] = payload[key]

    # 전부 빠져버리면, 이 payload의 라벨 중 하나는 강제로 넣어준다
    if not q:
        k = random.choice(list(payload.keys()))
        q[k] = payload[k]

    return q


def save_vector_label(
    vector_payload: List[Dict],
    mapping_json_path: str,
    output_txt_path: str
):

    # 매핑 로드
    with open(mapping_json_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    with open(output_txt_path, 'w', encoding='utf-8') as fout:
        for i, payload in enumerate(vector_payload):
            ids = []
            for key, value in payload.items():
                map_key = f"{key}:{value}"
                if map_key not in mapping:
                    raise ValueError(f"[Line {i}] {map_key} doesn't exist in mapping.json")
                ids.append(str(mapping[map_key]))
            fout.write(','.join(ids) + '\n')

    print(f"[✓] label txt saved: {output_txt_path}")


def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    dim = a[0]
    if (a.size % (dim + 1)) != 0:
        raise RuntimeError("fvecs file corrupted or wrong dimension")
    return a.reshape(-1, dim + 1)[:, 1:].view('float32')


def write_fvecs(path, vectors: np.ndarray):
    """
    Write vectors (float32 np.ndarray) to fvecs file.
    Each vector is stored as: [dim(int32), float32 * dim]

    Args:
        path (str): output fvecs file path
        vectors (np.ndarray): shape (N, dim)
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    N, dim = vectors.shape

    # Create output array: each row = 1 int32 + dim float32
    out = np.empty((N, dim + 1), dtype=np.float32)

    # First column must store dim as int32, but we'll cast later
    out[:, 0] = dim
    out[:, 1:] = vectors

    # Cast first column to int32 without altering memory layout
    out = out.view(np.int32)
    out[:, 0] = dim
    out = out.view(np.float32)

    # Finally write to file
    out.tofile(path)






def build_label_mapping(filter_dict: dict, mapping_json_path: str):

    
    filters = filter_dict

    mapping = {}
    next_id = 1


    for temp in filters:
        for key, value in temp.items():
            if (key, value) not in mapping:
                mapping[(key, value)] = next_id
                next_id += 1
            

    str_mapping = { f"{name}:{val}": idx for (name, val), idx in mapping.items() }


    with open(mapping_json_path, 'w', encoding='utf-8') as out:
        json.dump(str_mapping, out, ensure_ascii=False, indent=2)

    # print(f"Saved {len(str_mapping)} mappings to '{mapping_json_path}'. Max ID = {next_id - 1}")
    return mapping_json_path


def save_vector_label(
    vector_payload: List[Dict],
    mapping_json_path: str,
    output_txt_path: str,
    silently: bool = False
):

    with open(mapping_json_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    with open(output_txt_path, 'w', encoding='utf-8') as fout:
        for i, payload in enumerate(vector_payload):
            ids = []
            for key, value in payload.items():
                map_key = f"{key}:{value}"
                if map_key not in mapping:
                    raise ValueError(f"[Line {i}] mapping.json에 {map_key}이 없습니다.")
                ids.append(str(mapping[map_key]))
            fout.write(','.join(ids) + '\n')
    if silently == False:
        print(f"[✓] label.txt saved: {output_txt_path}")




def save_compute_groundtruth(
    base_label_file,
    query_label_file,
    gt_file,
    query_bin_file,
    base_bin_file,
    build_dir = "/home/ec2-user/hybrid_hardness/Generator/utils/compute_groundtruth",
    data_type = "float",
    dist_fn = "L2",
    scenario = "containment",
    K = 10,
    num_threads = 4,
):
    cmd = [
        build_dir,
        "--data_type", str(data_type),
        "--dist_fn", str(dist_fn),
        "--scenario", str(scenario),
        "--K", str(K),
        "--num_threads", str(num_threads),
        "--base_bin_file", str(base_bin_file),
        "--base_label_file", str(base_label_file),
        "--query_bin_file", str(query_bin_file),
        "--query_label_file", str(query_label_file),
        "--gt_file", str(gt_file)
    ]
    with open(os.devnull, 'w') as devnull:
        subprocess.run(
            cmd,
            check=True,
            stdout=devnull,  
            stderr=devnull,  
            stdin=devnull,  
        )



def load_groundtruth_bin(filename, nq):
    record_dtype = np.dtype([("idx", np.uint32), ("dist", np.float32)])
    file_size = os.path.getsize(filename)
    record_size = record_dtype.itemsize  # 8 bytes (4+4)
    total_recs = file_size // record_size

    if nq == 0 or total_recs % nq != 0:
        raise ValueError(f"GT file size mismatch: total_recs={total_recs}, nq={nq}")

    K_gt = total_recs // nq

    data = np.fromfile(filename, dtype=record_dtype, count=total_recs)

    gt_indices = data["idx"].reshape(nq, K_gt)
    return gt_indices