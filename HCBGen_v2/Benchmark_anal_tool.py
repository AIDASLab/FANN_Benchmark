#!/usr/bin/env python3
import os
import json
import argparse
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import hardness_estimator.calculate_hardness_v5_1 as hd

try:
    from sklearn.decomposition import PCA
except ImportError as e:
    raise ImportError("scikit-learn이 필요합니다. pip install scikit-learn") from e

try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute attribute/value statistics from payloads.jsonl"
    )
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--report_dir", required=True, type=str)
    return parser.parse_args()


def load_and_analyze(data_dir: str):
    jsonl_path = os.path.join(data_dir, "payloads.jsonl")
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"payloads.jsonl not found in {data_dir}")

    attr_value_counts = defaultdict(Counter)  # attr -> Counter(value -> count)
    attr_present_counts = defaultdict(int)    # attr -> #rows where attr exists
    total_rows = 0

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue  # or raise, if you want strictness

            total_rows += 1
            for attr, value in row.items():
                attr_value_counts[attr][value] += 1
                attr_present_counts[attr] += 1

    return total_rows, attr_value_counts, attr_present_counts


def _sorted_attr_names(attr_names):
    """label_1, label_2 ... 를 숫자 기준으로 정렬하기 위한 헬퍼."""
    def key_fn(name: str):
        parts = name.split("_")
        try:
            return (0, int(parts[-1]))
        except ValueError:
            return (1, name)
    return sorted(attr_names, key=key_fn)


def scatter_2d_by_label(
    X,                 # (N, D)
    labels,            # (N,)
    save_path,         # 저장 경로 (png)
    method="pca",      # "pca" | "umap"
    sample_max=50000,
    seed=42,
    point_size=2.0,
    alpha=0.8,
):
    """벡터 X와 정수 label을 받아 2D scatter 를 저장한다."""
    assert X.shape[0] == labels.shape[0]
    rng = np.random.default_rng(seed)

    # 1. 결측(-1) 제외
    mask = (labels != -1)
    Xp = X[mask]
    yp = labels[mask]

    if len(yp) == 0:
        print(f"[Plot] → {save_path} : 유효한 라벨이 없어 스킵")
        return

    # 2. 샘플링
    N = len(yp)
    if sample_max is not None and N > sample_max:
        idx = rng.choice(N, size=sample_max, replace=False)
        Xp = Xp[idx]
        yp = yp[idx]

    # 3. 2D 임베딩
    if method.lower() == "umap":
        if not _HAS_UMAP:
            raise RuntimeError("umap-learn이 설치되어 있지 않습니다. pip install umap-learn")
        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=30,
            min_dist=0.1,
        )
        Z = reducer.fit_transform(Xp)
        method_name = "UMAP"
    else:
        Z = PCA(n_components=2, random_state=seed).fit_transform(Xp)
        method_name = "PCA"

    # 4. 색상/플롯
    uniq = np.unique(yp)
    n_colors = len(uniq)
    cmap = plt.cm.get_cmap("tab20" if n_colors <= 20 else "hsv", n_colors)
    colors = {lab: cmap(i) for i, lab in enumerate(uniq)}

    plt.figure(figsize=(8, 7))
    for lab in uniq:
        m = (yp == lab)
        plt.scatter(
            Z[m, 0],
            Z[m, 1],
            s=point_size,
            c=[colors[lab]],
            label=str(lab),
            alpha=alpha,
            linewidths=0,
        )

    plt.title(f"2D scatter by label ({method_name})  N={len(yp):,}")
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")

    if len(uniq) <= 25:
        plt.legend(markerscale=3, frameon=False, ncol=3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] → Scatter saved to {save_path}")


def save_report(report_dir: str,
                total_rows: int,
                attr_value_counts,
                attr_present_counts):
    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "attribute_stats.json")

    stats = {}
    for attr in _sorted_attr_names(attr_value_counts.keys()):
        present = attr_present_counts[attr]
        missing = total_rows - present
        missing_ratio = missing / total_rows if total_rows > 0 else 0.0

        stats[attr] = {
            "present_count": present,
            "missing_count": missing,
            "missing_ratio": missing_ratio,
            "num_distinct_values": len(attr_value_counts[attr]),
            "value_counts": dict(attr_value_counts[attr]),
        }

    # JSON 저장
    with open(out_path, "w") as f:
        json.dump(
            {
                "total_rows": total_rows,
                "num_attributes": len(stats),
                "attributes": stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[Dataset] → total_rows={total_rows}")
    print(f"[Attribute] → num_attributes={len(stats)}")

    # 1️⃣ Missing ratio 막대 그래프
    attrs = list(stats.keys())
    missing_ratios = [stats[a]["missing_ratio"] for a in attrs]

    plt.figure(figsize=(10, 5))
    plt.bar(attrs, missing_ratios, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing Ratio")
    plt.title("Attribute Missing Ratio")
    plt.tight_layout()
    missing_plot = os.path.join(report_dir, "missing_ratio.png")
    plt.savefig(missing_plot, dpi=150)
    plt.close()
    print(f"[Plot] → Missing ratio chart saved to {missing_plot}")

    # 2️⃣ 각 attribute의 value 분포 히스토그램
    for attr in attrs:
        value_counts = stats[attr]["value_counts"]
        if not value_counts:
            continue
        keys = [str(k) if k is not None else "None" for k in value_counts.keys()]
        values = list(value_counts.values())

        plt.figure(figsize=(6, 4))
        plt.bar(keys, values, color="lightcoral")
        plt.title(f"Value Distribution: {attr}")
        plt.xlabel("Value")
        ax = plt.gca()
        ax.set_xticklabels([])       # ✅ 눈금은 유지, 텍스트 라벨만 제거
        plt.ylabel("Count")
        plt.tight_layout()
        out_png = os.path.join(report_dir, f"value_dist_{attr}.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[Plot] → Value distribution for {attr} saved to {out_png}")


    # 콘솔 요약 출력
    for attr in _sorted_attr_names(stats.keys()):
        s = stats[attr]
        print(f"\n[Attribute] {attr}")
        print(f"├─ present: {s['present_count']} / {total_rows}")
        print(f"└─ missing_ratio: {s['missing_ratio']:.4f}")

    return stats


def build_payload_matrix(data_dir: str, attr_names):
    """payloads.jsonl을 다시 읽어 (N, A) 행렬을 만든다. 결측치는 -1."""
    jsonl_path = os.path.join(data_dir, "payloads.jsonl")
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)

    N = len(rows)
    A = len(attr_names)
    payloads = np.full((N, A), -1, dtype=int)

    for i, row in enumerate(rows):
        for j, attr in enumerate(attr_names):
            if attr in row:
                payloads[i, j] = row[attr]

    return payloads


def vector_label_report(data_dir: str,
                        report_dir: str,
                        total_rows: int,
                        stats: dict):
    base_vec_path = os.path.join(data_dir, "mid_format/base_vectors.npy")
    if not os.path.isfile(base_vec_path):
        print(f"[Vector] → base_vectors.npy not found in {data_dir}, skip scatter plots.")
        return

    X = np.load(base_vec_path)  # (N, D)
    if X.shape[0] != total_rows:
        raise ValueError(
            f"base_vectors.npy row({X.shape[0]}) != payload rows({total_rows})"
        )

    attr_names = _sorted_attr_names(stats.keys())
    payloads = build_payload_matrix(data_dir, attr_names)  # (N, A)

    if payloads.shape[0] != X.shape[0]:
        raise ValueError(
            f"payload matrix row({payloads.shape[0]}) != base_vectors row({X.shape[0]})"
        )

    os.makedirs(report_dir, exist_ok=True)
    for j, attr in enumerate(attr_names):
        labels = payloads[:, j]
        out_png = os.path.join(report_dir, f"scatter_{attr}_pca.png")
        scatter_2d_by_label(
            X,
            labels,
            save_path=out_png,
            method="pca",
            sample_max=100000,
        )


# def query_hardness_report(data_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)

#     # 1) 벡터/페이로드 로드
#     base_vec_path = os.path.join(data_dir, "base_vectors.npy")
#     if not os.path.isfile(base_vec_path):
#         raise FileNotFoundError(f"base_vectors.npy not found in {data_dir}")
#     vectors = np.load(base_vec_path)

#     payloads_file = f"{data_dir}/payloads.jsonl"
#     payloads = []
#     with open(payloads_file, "r") as f:
#         for line in f:
#             payloads.append(json.loads(line))
#     print(f"Loaded {len(payloads)} payloads")

    
#     # 2) 테스트 로드
#     tests_file = os.path.join(data_dir, "tests.jsonl")
#     tests = []
#     with open(tests_file, "r") as f:
#         for line in f:
#             tests.append(json.loads(line))

#     print(f"Loaded {len(tests)} tests")

#     # 3) Hardness 계산
#     print("loading hardness estimator ...")
#     estimator = hd.HybridHardnessEstimator(vectors, payloads, distance_metric="l2")
#     print("Hardness estimator initialized!")
#     result = []
#     print("Calculating query hardness ... ")
#     for i, test in enumerate(tqdm(tests, desc="[Hardness]")):
#         h = estimator.compute_total_hardness(test)["Post_Hardness"]
#         result.append(h)

#     result = np.asarray(result, dtype=float)

#     # 4) 통계 출력
#     print("\n[Hardness] → Post_Hardness summary")
#     print(f"├─ count : {len(result):,}")
#     print(f"├─ min   : {result.min():.6f}")
#     print(f"├─ max   : {result.max():.6f}")
#     print(f"├─ mean  : {result.mean():.6f}")
#     print(f"└─ median: {np.median(result):.6f}")

#     # 5) 분포 차트 저장
#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax.hist(result, bins=50, alpha=0.7)
#     ax.set_title("Distribution of Post_Hardness")
#     ax.set_xlabel("Post_Hardness")
#     ax.set_ylabel("Frequency")
#     fig.tight_layout()

#     out_png = os.path.join(output_dir, "post_hardness_distribution.png")
#     fig.savefig(out_png, dpi=150)
#     plt.close(fig)
#     print(f"[Plot] → Hardness distribution saved to {out_png}")

    # (선택) 원시 값 저장
    # out_npy = os.path.join(output_dir, "post_hardness_values.npy")
    # np.save(out_npy, result)
    # print(f"[Report] → Raw hardness values saved to {out_npy}")
def query_hardness_report(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1) 벡터/페이로드 로드
    base_vec_path = os.path.join(data_dir, "mid_format/base_vectors.npy")
    if not os.path.isfile(base_vec_path):
        raise FileNotFoundError(f"base_vectors.npy not found in {data_dir}")
    vectors = np.load(base_vec_path)

    payloads_file = os.path.join(data_dir, "payloads.jsonl")
    payloads = []
    with open(payloads_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payloads.append(json.loads(line))
    print(f"Loaded {len(payloads)} payloads")

    # 2) 테스트 로드
    tests_file = os.path.join(data_dir, "tests.jsonl")
    tests = []
    with open(tests_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tests.append(json.loads(line))
    print(f"Loaded {len(tests)} tests")

    # 3) Hardness 계산
    print("loading hardness estimator ...")
    estimator = hd.HybridHardnessEstimator(vectors, payloads, distance_metric="l2")
    print("Hardness estimator initialized!")
    result = []
    print("Calculating query hardness ... ")
    for i, test in enumerate(tqdm(tests, desc="[Hardness]")):
        h = estimator.compute_total_hardness(test)["Post_Hardness"]
        result.append(h)

    result = np.asarray(result, dtype=float)

    # 4) 통계 계산
    hardness_stats = {
        "count": int(len(result)),
        "min": float(np.min(result)),
        "max": float(np.max(result)),
        "mean": float(np.mean(result)),
        "median": float(np.median(result)),
        "std": float(np.std(result)),
        "percentiles": {
            "p1": float(np.percentile(result, 1)),
            "p5": float(np.percentile(result, 5)),
            "p10": float(np.percentile(result, 10)),
            "p25": float(np.percentile(result, 25)),
            "p50": float(np.percentile(result, 50)),
            "p75": float(np.percentile(result, 75)),
            "p90": float(np.percentile(result, 90)),
            "p95": float(np.percentile(result, 95)),
            "p99": float(np.percentile(result, 99)),
        },
    }

    print("\n[Hardness] → Post_Hardness summary")
    for k, v in hardness_stats.items():
        if isinstance(v, dict):
            continue
        print(f"├─ {k:<7}: {v:.6f}")
    print("└─ percentiles:", ", ".join(
        f"{p}={v:.4f}" for p, v in hardness_stats["percentiles"].items())
    )

    # 5) 분포 차트 저장 (bins / range 고정)
    fig, ax = plt.subplots(figsize=(8, 4))
    bin_edges = np.linspace(0.0, 5.0, 101)  # 0~3을 0.05 간격으로
    n, bins, _ = ax.hist(
        result,
        bins=bin_edges,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )
    ax.set_title("Distribution of Post_Hardness")
    ax.set_xlabel("Post_Hardness")
    ax.set_ylabel("Frequency")

    # x축 범위 및 tick 고정
    ax.set_xlim(0.0, 5.0)
    ax.set_xticks(np.arange(0.0, 5.0 + 0.5, 0.5))

    fig.tight_layout()

    out_png = os.path.join(output_dir, "post_hardness_distribution.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[Plot] → Hardness distribution saved to {out_png}")

    # 6) JSON 리포트 업데이트
    attr_stats_path = os.path.join(output_dir, "attribute_stats.json")
    if os.path.exists(attr_stats_path):
        with open(attr_stats_path, "r") as f:
            try:
                full_report = json.load(f)
            except json.JSONDecodeError:
                full_report = {}
    else:
        full_report = {}

    full_report["hardness_report"] = {
        "summary": hardness_stats,
        "histogram": {
            "bins": [float(b) for b in bins.tolist()],
            "counts": [int(x) for x in n.tolist()],
        },
        "plot_file": os.path.basename(out_png),
    }

    out_json = os.path.join(output_dir, "attribute_stats.json")
    with open(out_json, "w") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"[Report] → Hardness stats appended to {out_json}")


def main():
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.report_dir

    total_rows, attr_value_counts, attr_present_counts = load_and_analyze(
        data_dir
    )
    # 1) attribute 통계 리포트
    stats = save_report(
        output_dir, total_rows, attr_value_counts, attr_present_counts
    )

    # 2) base_vectors + payloads 기반 2D scatter 리포트
    vector_label_report(data_dir, output_dir, total_rows, stats)

    # 3) query hardness 분포 리포트
    query_hardness_report(data_dir, output_dir)


if __name__ == "__main__":
    main()
