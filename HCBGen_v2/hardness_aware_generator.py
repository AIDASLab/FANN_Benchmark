import sys
import argparse
import os
import json
import subprocess
import random

import numpy as np
from tqdm import tqdm

import label_generator.base_label_generator as blg
import Index_analyzer.Post_Filter_Analyzer as PostFilterAnalyzer
import Index_analyzer.Pre_Filter_Analyzer as PreFilterAnalyzer
import Index_analyzer.NHQ_Analyzer as NHQAnalyzer
import Index_analyzer.UNG_Analyzer as UNGAnalyzer
import Index_analyzer.RWALKS_Analyzer as RWALKSAnalyzer
import Index_analyzer.MILVUS_Analyzer as MILVUSAnalyzer
# import Index_analyzer.ACORN_Analyzer as ACORNAnalyzer
# import hardness_estimator.calculate_hardness_v5_0 as hd
import hardness_estimator.calculate_hardness_v5_1 as hd
# import hardness_estimator.calculate_hardness_test as hd


# ---------------------------------------------------------
#  Helper: base_complexity 파서
# ---------------------------------------------------------
def parse_base_complexity(values):
    """
    Parse base complexity parameters.

    Expected input:
        [num_attribute, cardinalities, distribution, (zipf_param), missing_prob, correlations]

    Examples:
        --base_complexity 3 "4,6,8" zipf 1.2 "0.1,0.2,0.3" "0.8,0.6,0.9"
        --base_complexity 3 "4,6,8" random "0.0,0.0,0.0" "0.8,0.6,0.9"
    """
    if len(values) not in (5, 6):
        raise argparse.ArgumentTypeError(
            "base_complexity expects 5 or 6 arguments:\n"
            "zipf:   num_attr cardinals zipf <param> missing_prob correlation\n"
            "random: num_attr cardinals random missing_prob correlation"
        )

    # num_attribute
    try:
        num_attribute = int(values[0])
    except ValueError:
        raise argparse.ArgumentTypeError("num_attribute must be an integer.")

    # cardinalities
    try:
        cardinality = [int(x) for x in values[1].split(",")]
        if len(cardinality) != num_attribute:
            raise argparse.ArgumentTypeError(
                f"cardinality length ({len(cardinality)}) does not match num_attribute ({num_attribute})."
            )
    except Exception:
        raise argparse.ArgumentTypeError(
            "cardinality must be a comma-separated list of integers. Example: 4,6,8"
        )

    # distribution
    distribution = values[2].lower()
    if distribution not in ("zipf", "random"):
        raise argparse.ArgumentTypeError("distribution must be either 'zipf' or 'random'.")

    # zipf parameter (optional)
    if distribution == "zipf":
        try:
            zipf_param = float(values[3])
        except ValueError:
            raise argparse.ArgumentTypeError("Zipf distribution requires a float parameter, e.g., 1.2")
        offset = 1
    else:
        zipf_param = None
        offset = 0

    # missing_prob
    try:
        missing_prob = [float(x) for x in values[3 + offset].split(",")]
        if len(missing_prob) != num_attribute:
            raise argparse.ArgumentTypeError(
                f"missing_prob length ({len(missing_prob)}) does not match num_attribute ({num_attribute})."
            )
        if not all(0.0 <= c <= 1.0 for c in missing_prob):
            raise argparse.ArgumentTypeError("Each missing_prob value must be within [0, 1].")
    except Exception:
        raise argparse.ArgumentTypeError(
            "missing_prob must be a comma-separated list of floats. Example: 0.1,0.2,0.3"
        )

    # correlation
    try:
        correlations = [float(x) for x in values[4 + offset].split(",")]
        if len(correlations) != num_attribute:
            raise argparse.ArgumentTypeError(
                f"correlation length ({len(correlations)}) does not match num_attribute ({num_attribute})."
            )
        if not all(0.0 <= c <= 1.0 for c in correlations):
            raise argparse.ArgumentTypeError("Each correlation value must be within [0, 1].")
    except Exception:
        raise argparse.ArgumentTypeError(
            "correlation must be a comma-separated list of floats. Example: 0.8,0.6,0.9"
        )

    return {
        "num_attribute": num_attribute,
        "cardinality": cardinality,
        "distribution": distribution,
        "zipf_param": zipf_param,
        "missing_prob": missing_prob,
        "correlation": correlations,
    }


# ---------------------------------------------------------
#  main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hybrid Index Label Generator")
    parser.add_argument("--base_vector_path", required=True, help="Path to base vector .fvecs file")
    parser.add_argument(
        "--query_vector_path",
        required=False,
        help="Optional path to query vector .fvecs file. "
             "If omitted, queries are randomly sampled from base vectors.",
    )
    parser.add_argument("--index", required=True, help="Index type (e.g., HNSW, IVF, etc.)")
    parser.add_argument("--save_dir", required=True, help="Output directory to save artifacts")

    parser.add_argument(
        "--num_queries",
        type=int,
        required=True,
        help="Number of queries to generate",
    )

    # base_complexity OR base_payloads_path 중 하나만 사용
    parser.add_argument(
        "--base_complexity",
        nargs="+",
        metavar="BASE_COMPLEXITY_ARGS",
        type=str,
        required=False,
        help=(
            "[MODE 1: generate]\n"
            "Base complexity config:\n"
            "  zipf:   NUM_ATTR CARDINALITY zipf ZIPF_PARAM MISSING_PROB CORRELATION\n"
            "  random: NUM_ATTR CARDINALITY random MISSING_PROB CORRELATION\n"
        ),
    )

    parser.add_argument(
        "--base_payloads_path",
        type=str,
        required=False,
        help=(
            "[MODE 2: load]\n"
            "Path to existing base payloads JSONL file. "
            "Each line should be a JSON dict of attribute→value."
        ),
    )

    parser.add_argument(
        "--base_mapping_path",
        type=str,
        required=False,
        help=(
            "Optional path to existing base label mapping JSON file. "
            "If omitted, mapping will be built from payloads and saved into save_dir/mapping.json."
        ),
    )

    parser.add_argument(
        "--query_complexity",
        choices=["high", "low", "random", "range", "match_pdf"],
        required=True,
        help="Query complexity profile (high | low | random | range | match_pdf)",
    )

    parser.add_argument(
        "--hardness_range",
        type=str,
        required=False,
        help=(
            "When --query_complexity=range: target hardness range as 'MIN,MAX'. "
            "Example: --hardness_range 0.8,1.2"
        ),
    )

    parser.add_argument(
        "--target_hardness_json",
        type=str,
        required=False,
        help=(
            "When --query_complexity=match_pdf: JSON file that contains list[dict] "
            "with 'Post_Hardness' values (semi-real workload hardness)."
        ),
    )

    parser.add_argument(
        "--num_bins",
        type=int,
        default=40,
        help="Number of hardness bins for match_pdf mode (default: 40).",
    )

    parser.add_argument(
        "--dev_mode",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    # ---------------------------
    # hardness_range validation (range 모드 전용)
    # ---------------------------
    hardness_range = None
    if args.query_complexity == "range":
        if args.hardness_range is None:
            parser.error("--hardness_range is required when --query_complexity=range")
        try:
            h_min_str, h_max_str = args.hardness_range.split(",")
            h_min = float(h_min_str)
            h_max = float(h_max_str)
        except Exception:
            parser.error("--hardness_range must be 'MIN,MAX' (e.g., 0.5,1.0)")
        if h_min > h_max:
            parser.error("In --hardness_range MIN must be <= MAX")
        hardness_range = (h_min, h_max)
    else:
        if args.hardness_range is not None:
            parser.error("--hardness_range can be used only when --query_complexity=range")

    # match_pdf 모드일 때 target JSON 필수
    if args.query_complexity == "match_pdf" and not args.target_hardness_json:
        parser.error("--target_hardness_json is required when --query_complexity=match_pdf")

    # ------------------------------------------------------------------
    # MODE 결정: base label generate vs load
    # ------------------------------------------------------------------
    use_complexity = args.base_complexity is not None
    use_payloads = args.base_payloads_path is not None

    if use_complexity == use_payloads:
        parser.error(
            "You must provide exactly ONE of:\n"
            "  - --base_complexity ...      (generate base labels)\n"
            "  - --base_payloads_path FILE  (load precomputed base labels)\n"
        )

    base_mode = "generate" if use_complexity else "load"

    # generate 모드일 때만 base_complexity 파싱
    if base_mode == "generate":
        base_complexity = parse_base_complexity(args.base_complexity)
    else:
        base_complexity = None

    # Ensure save_dir exists
    os.makedirs(args.save_dir, exist_ok=True)

    base_vector_path = args.base_vector_path
    query_vector_path = args.query_vector_path  # 있을 수도, 없을 수도 있음
    index_method = args.index
    save_dir = args.save_dir
    hardness_target = args.query_complexity
    dev_mode = args.dev_mode
    num_query = args.num_queries

    # mid_dir 및 base 벡터 npy 저장
    mid_dir = os.path.join(save_dir, "mid_format")
    os.makedirs(mid_dir, exist_ok=True)

    base_vector_npy = blg.read_fvecs(base_vector_path)
    np.save(os.path.join(mid_dir, "base_vectors.npy"), base_vector_npy)
    np.save(os.path.join(save_dir, "vectors.npy"), base_vector_npy)

    # allowed_values_per_attr 기본값
    allowed_values_per_attr = None

    # ------------------------------------------------------------------
    # base_mode 별로 num_attribute / cardinality / attr_keys / allowed_values 준비
    # ------------------------------------------------------------------
    if base_mode == "generate":
        num_attribute = base_complexity["num_attribute"]
        cardinality = base_complexity["cardinality"]
        distribution = base_complexity["distribution"]
        zipf_param = base_complexity.get("zipf_param")
        base_missing_prob = base_complexity["missing_prob"]
        correlation = base_complexity["correlation"]
        base_label_payload = None  # 나중에 generate 후 채움

        # generate 모드에서는 attribute 이름을 단순 label_1, label_2, ... 로 사용
        attr_keys = [f"label_{i+1}" for i in range(num_attribute)]
        allowed_values_per_attr = None  # synthetic 이므로 제한 없음

    else:
        # base_payloads_path에서 payload 읽어서:
        #  - 전체 attr_keys (union of keys)
        #  - 각 attr 별로 실제 등장한 값 집합
        base_payloads_path = args.base_payloads_path
        base_label_payload = []

        value_sets = {}  # key -> set of observed int values (excluding -1)

        print("[Base Label Loading] → Scanning payloads to infer schema...")
        with open(base_payloads_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading base payloads"):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                base_label_payload.append(payload)

                for k, v in payload.items():
                    if v is None:
                        continue
                    if not isinstance(v, int):
                        raise ValueError(f"Non-integer value found for key '{k}': {v}")
                    if v == -1:
                        continue
                    if k not in value_sets:
                        value_sets[k] = set()
                    value_sets[k].add(v)

        # 전체 attr_keys = union of keys
        attr_keys = sorted(value_sets.keys())
        num_attribute = len(attr_keys)

        # 각 attr 별로 실제 등장한 values 목록과 cardinality
        allowed_values_per_attr = [sorted(value_sets[k]) for k in attr_keys]
        cardinality = [len(vals) for vals in allowed_values_per_attr]

        non_empty_payloads = [p for p in base_label_payload if len(p) > 0]
        if not non_empty_payloads:
            raise RuntimeError("All base payloads are empty; cannot generate queries.")

        distribution = "predefined"
        zipf_param = "predefined"
        base_missing_prob = "predefined"
        correlation = "predefined"

    # ------------------------------------------------------------------
    # query_vector_npy 준비:
    #  - query_vector_path 있으면 → 그 fvecs 사용(앞에서 num_queries 개까지만)
    #  - 없으면 → base_vector_npy에서 num_queries개 랜덤 샘플링
    # ------------------------------------------------------------------
    if query_vector_path is not None:
        query_vector_npy = blg.read_fvecs(query_vector_path)
        total_q = len(query_vector_npy)
        if total_q < num_query:
            raise ValueError(
                f"query_vector_path has only {total_q} vectors, but --num_queries={num_query} requested."
            )
        if total_q > num_query:
            query_vector_npy = query_vector_npy[:num_query]
        query_source_str = query_vector_path
    else:
        total_base = len(base_vector_npy)
        if total_base < num_query:
            raise ValueError(
                f"Base has only {total_base} vectors, but --num_queries={num_query} requested for queries."
            )
        sampled_idx = np.random.choice(total_base, size=num_query, replace=False)
        query_vector_npy = base_vector_npy[sampled_idx]
        query_source_str = "SAMPLED_FROM_BASE"

    np.save(os.path.join(mid_dir, "query_vector.npy"), query_vector_npy)

    if query_vector_npy.shape[0] != num_query:
        raise RuntimeError(
            f"Internal error: query_vector_npy.shape[0]={query_vector_npy.shape[0]} "
            f"!= num_query={num_query}"
        )

    print("=" * 180)
    print()
    print(f"Hardness Aware Hybrid Search (FANN) Benchmark Generator v.2.5")
    print()
    print("=" * 100)
    print("[Config]")
    print(f"  ├─ [Mode]    → base_mode={base_mode}")
    print(f"  ├─ [Dataset] → base={base_vector_path} | query_source={query_source_str}")
    print(f"  ├─ [Index]   → {index_method}")
    print(f"  ├─ [Output]  → {save_dir}")
    print(f"  ├─ [Querying]→ complexity={hardness_target}, num_queries={num_query}")
    if hardness_target == "range" and hardness_range is not None:
        print(f"  │                 hardness_range=[{hardness_range[0]:.6f}, {hardness_range[1]:.6f}]")
    if hardness_target == "match_pdf":
        print(f"  │                 target_hardness_json={args.target_hardness_json}, "
              f"num_bins={args.num_bins}")
    print(f"  └─ [Base]    → attr={num_attribute}, card={cardinality}, "
          f"dist={distribution}, zipf_param={zipf_param}, "
          f"missing={base_missing_prob}, corr={correlation}")

    ###########################################################
    #### base label generation / loading step
    ###########################################################
    print("=" * 100)
    print("[Base Label Gerneration Step]" if base_mode == "generate" else "[Base Label Loading Step]")

    if base_mode == "generate":
        print(f"  ├─ 🧩 Base vector loaded: shape = {base_vector_npy.shape}")
        print("  ├─ ⚙️  Now generating base labels...")

        payloads = blg.generate_attribute_payloads(
            num_vectors=len(base_vector_npy),
            num_attributes=num_attribute,
            cardinalities=cardinality,
            base_vectors=base_vector_npy,
            correlations=correlation,
            distribution=distribution,
            zipf_param=zipf_param if distribution == "zipf" else None,
            missing_prob=base_missing_prob,
            missing_value=-1
        )

        if dev_mode:
            print("  ├─ 🔧 Developer mode enabled: generating payloads with missing values (-1)...")
            base_label_payload_all = blg.payloads_to_dicts(
                payloads,
                attr_keys=attr_keys,
                include_missing=True,
                missing_value=-1
            )
            with open(os.path.join(save_dir, "payloads_all.jsonl"), "w", encoding="utf-8") as f:
                for payload in base_label_payload_all:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(f"  ├─ 💾 Saved full payloads (including missing) → {save_dir}")
            mapping_path_all = blg.build_label_mapping(
                base_label_payload_all,
                os.path.join(save_dir, "mapping_all.json")
            )
            print(f"  ├─ 🧭 Label mapping (full) saved → {mapping_path_all}")

        print("  ├─ ✅ Base label generation completed!")
        base_label_payload = blg.payloads_to_dicts(
            payloads,
            attr_keys=attr_keys,
            include_missing=False,
            missing_value=-1
        )

        with open(os.path.join(save_dir, "payloads.jsonl"), "w", encoding="utf-8") as f:
            for payload in base_label_payload:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        print(f"  ├─ 💾 Saved filtered base data payloads → {save_dir}")
        mapping_path = blg.build_label_mapping(
            base_label_payload,
            os.path.join(save_dir, "mapping.json")
        )
        print(f"  ├─ 🧭 Label mapping saved → {mapping_path}")
        print("  ├─ ✅ All base label generation tasks completed.")

    else:
        # base_mode == "load"
        print(f"  ├─ 🧩 Loaded base payloads: count = {len(base_label_payload):,}")
        payloads_out = os.path.join(save_dir, "payloads.jsonl")
        with open(payloads_out, "w", encoding="utf-8") as f:
            for payload in base_label_payload:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(f"  ├─ 💾 Re-saved base payloads → {payloads_out}")

        if args.base_mapping_path:
            mapping_path = args.base_mapping_path
            print(f"  ├─ 🧭 Using existing mapping file → {mapping_path}")
        else:
            mapping_path = blg.build_label_mapping(
                base_label_payload,
                os.path.join(save_dir, "mapping.json")
            )
            print(f"  ├─ 🧭 Built new mapping from payloads → {mapping_path}")

    # 공통: base_label.txt 생성 + .bin 생성
    base_vector_bin = os.path.join(mid_dir, "base_vector.bin")
    query_vector_bin = os.path.join(mid_dir, "query_vector.bin")

    converter = "./utils/fvecs_to_bin"
    subprocess.run(
        [
            converter,
            "--data_type", "float",
            "--input_file", base_vector_path,
            "--output_file", base_vector_bin
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # query fvecs 입력 파일 결정
    query_fvecs_src = os.path.join(mid_dir, "query_vectors_used.fvecs")
    blg.write_fvecs(query_fvecs_src, query_vector_npy)

    subprocess.run(
        [
            "./utils/fvecs_to_bin",
            "--data_type", "float",
            "--input_file", query_fvecs_src,
            "--output_file", query_vector_bin
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    if dev_mode and base_mode == "generate":
        print("  ├─ 🔧 Developer mode enabled: saveing base_label_all.txt")
        blg.save_vector_label(
            base_label_payload_all,
            mapping_path_all,
            os.path.join(mid_dir, "base_label_all.txt"),
            silently=True
        )
        print("  └─ Saveing base_label.txt")

    blg.save_vector_label(
        base_label_payload,
        mapping_path,
        os.path.join(mid_dir, "base_label.txt"),
        silently=True
    )

    ###########################################################
    #### query generation step (1차: GT>0 확보)
    ###########################################################
    target_value = 4294967295
    valid_query_labels = []
    iteraion_count = 0

    if hardness_target == "random":
        print(f"[Query Generation Step] → Target number of queries: {num_query:,}, Selected query complexity: {hardness_target.upper()}")
        print("  ├─ No need to initialize index analyzer")
        print("  └─ No need to initialize hardness esimator")
    else:
        ###########################################################
        #### index analyzer step
        ###########################################################
        print("=" * 100)
        print("[Index Type Anlyzing Step]")

        if index_method == "Post_Filtering":
            index_type = PostFilterAnalyzer.Analyze(dev_mode)
        elif index_method == "Pre_Filtering":
            index_type = PreFilterAnalyzer.Analyze(dev_mode)
        elif index_method == "NHQ":
            index_type = NHQAnalyzer.Analyze(dev_mode)
        elif index_method == "ACORN":
            index_type = ACORNAnalyzer.Analyze(dev_mode)
        elif index_method == "RWALKS":
            index_type = RWALKSAnalyzer.Analyze(dev_mode)
        elif index_method == "UNG":
            index_type = UNGAnalyzer.Analyze(dev_mode)
        elif index_method == "MILVUS":
            index_type = MILVUSAnalyzer.Analyze(dev_mode)
        else:
            print(f"⚠️  '{index_method}' is not currently supported.")
            print("   → Please test it manually and enter the result (post_base / pre_base): ", end="")
            index_type = input().strip().lower()

        if index_type not in ["post_base", "pre_base"]:
            print("  ├─ Invalid input. Defaulting to 'post_base'.")
            index_type = "post_base"

        print(f"  └─ ✅ This index type is : {index_type}")

        print("=" * 100)
        print(f"[Query Generation Step] → Target number of queries: {num_query:,}, Selected query complexity: {hardness_target.upper()}")
        print("  ├─ ⚙️ Prepareing hardness estimator ...")
        estimator = hd.HybridHardnessEstimator(base_vector_npy, base_label_payload, distance_metric="l2")
        print("  └─ ✅ hardness estimator initialized")
        print("=" * 50)
        print(f"[Generating {num_query} queries to check hardness threshold]")

    # -------------------------------
    # GT>0 query 확보
    # -------------------------------
    if base_mode == "load":
        print("=" * 100)
        print("[Query Generation Step] load-mode: sampling queries from base payloads")

        query_missing_prob = np.array([0.5] * num_attribute)
        print(f"  ├─ Using missing probability : {query_missing_prob}")

        while len(valid_query_labels) < num_query:
            iteraion_count += 1
            print(f"  ├─ Current valid query labels : {len(valid_query_labels):,}")

            remaining = num_query - len(valid_query_labels)
            sample_cnt = min(remaining, len(non_empty_payloads))

            sampled_payloads = random.sample(non_empty_payloads, sample_cnt)
            query_label_payload = []

            for payload in sampled_payloads:
                q = {}
                for j, key in enumerate(attr_keys):
                    if key not in payload:
                        continue
                    if np.random.rand() > query_missing_prob[j]:
                        q[key] = payload[key]

                if not q:
                    k = random.choice(list(payload.keys()))
                    q[k] = payload[k]
                query_label_payload.append(q)

            tmp_query_label_path = os.path.join(mid_dir, "query_label_tmp.txt")
            blg.save_vector_label(
                query_label_payload,
                mapping_path,
                tmp_query_label_path,
                silently=True
            )

            blg.save_compute_groundtruth(
                os.path.join(mid_dir, "base_label.txt"),
                tmp_query_label_path,
                os.path.join(mid_dir, "gt_tmp.bin"),
                query_vector_bin,
                base_vector_bin,
                build_dir="./utils/compute_groundtruth",
                num_threads=28
            )

            gt_indices = blg.load_groundtruth_bin(os.path.join(mid_dir, "gt_tmp.bin"), sample_cnt)
            gt_indices = gt_indices.tolist()

            valid_indices = [
                i for i, sublist in enumerate(gt_indices)
                if not all(v == target_value for v in sublist)
            ]

            print(f"  ├─ Valid queries found in this iteration : {len(valid_indices):,}")

            for idx in valid_indices:
                valid_query_labels.append(query_label_payload[idx])
                if len(valid_query_labels) >= num_query:
                    break

        print(f"  ✅ Reached target count → {len(valid_query_labels):,} valid queries total")

    else:
        # generate 모드
        while len(valid_query_labels) < num_query:
            iteraion_count += 1
            print(f"  ├─ Current valid query labels : {len(valid_query_labels):,}")
            query_missing_prob = np.array([0.5] * num_attribute)
            print(f"  ├─ Generated missing probability : {query_missing_prob}")

            query_label_arr = blg.generate_query_payloads(
                num_query,
                num_attribute,
                cardinality,
                query_missing_prob,
                "random",
                allowed_values_per_attr=allowed_values_per_attr,
            )

            query_label_payload = blg.payloads_to_dicts(
                query_label_arr,
                attr_keys=None,   # synthetic 모드는 prefix label_1, ... 사용
                include_missing=False,
            )

            blg.save_vector_label(
                query_label_payload,
                mapping_path,
                os.path.join(mid_dir, "query_label.txt"),
                silently=True
            )

            blg.save_compute_groundtruth(
                os.path.join(mid_dir, "base_label.txt"),
                os.path.join(mid_dir, "query_label.txt"),
                os.path.join(mid_dir, "gt.bin"),
                query_vector_bin,
                base_vector_bin,
                build_dir="./utils/compute_groundtruth",
                num_threads=28
            )

            gt_indices = blg.load_groundtruth_bin(os.path.join(mid_dir, "gt.bin"), num_query)
            gt_indices = gt_indices.tolist()

            valid_indices = [
                i for i, sublist in enumerate(gt_indices)
                if not all(v == target_value for v in sublist)
            ]

            print(f"  ├─ Valid queries found in this iteration : {len(valid_indices):,}")

            filtered_query_labels = [query_label_payload[i] for i in valid_indices]
            valid_query_labels += filtered_query_labels

            if len(valid_query_labels) >= num_query:
                valid_query_labels = valid_query_labels[:num_query]
                print(f"  ✅ Reached target count → {len(valid_query_labels):,} valid queries total")
                break

    print(f"\n[Summary]")
    print(f"  ├─ Total iterations     : {iteraion_count}")
    print(f"  └─ Final valid queries  : {len(valid_query_labels):,}")

    print(f"\n[Ground Truth Check]")
    blg.save_vector_label(
        valid_query_labels,
        mapping_path,
        os.path.join(mid_dir, "query_label.txt"),
        silently=True
    )
    blg.save_compute_groundtruth(
        os.path.join(mid_dir, "base_label.txt"),
        os.path.join(mid_dir, "query_label.txt"),
        os.path.join(mid_dir, "gt.bin"),
        query_vector_bin,
        base_vector_bin,
        build_dir="./utils/compute_groundtruth",
        num_threads=28
    )

    gt_indices = blg.load_groundtruth_bin(os.path.join(mid_dir, "gt.bin"), num_query)
    gt_indices = gt_indices.tolist()

    valid_indices = [
        i for i, sublist in enumerate(gt_indices)
        if not all(v == target_value for v in sublist)
    ]

    print(f"  └─ Check complete (valid GT count: {len(valid_indices):,})")

    conditions = []
    for query_label in valid_query_labels:
        temp = {"and": []}
        for key, value in query_label.items():
            temp["and"].append({key: {"match": {"value": value}}})
        conditions.append(temp)

    tests = []
    query_vector_array = query_vector_npy.tolist()
    for query, condition, gt in zip(query_vector_array, conditions, gt_indices):
        temp = {}
        temp["query"] = query
        temp["conditions"] = condition
        temp["closest_ids"] = gt
        tests.append(temp)

    # -------------------------------------------------
    # hardness_target == random 이면 여기서 종료
    # -------------------------------------------------
    if hardness_target == "random":
        with open(os.path.join(save_dir, "tests.jsonl"), "w", encoding="utf-8") as f:
            for item in tests:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"\n[Output]")
        print(f"  └─ Saved test file     → {os.path.join(save_dir, 'tests.jsonl')}")
        print("✅ Query-label generation completed successfully.")
        return

    # -------------------------------------------------
    # non-random 모드: hardness 계산
    # -------------------------------------------------
    print("=" * 50)

    result = []
    if hardness_target in ["high", "low"]:
        desc_str = f"[Checking hardness threshold for {hardness_target.upper()}]"
    elif hardness_target == "range":
        desc_str = f"[Calculating initial queries' hardness {hardness_target.upper()}]"
    else:  # match_pdf
        desc_str = "[Calculating initial queries' hardness MATCH_PDF]"

    for test in tqdm(tests, desc=desc_str):
        result.append(estimator.compute_total_hardness(test))

    print(f"  ├─ Threshold/base hardness calculation done ✅")
    print(f"  ├─ Mode              : {hardness_target.upper()}")

    # 공통: index_type에 맞게 hardness 스칼라 배열로 정규화
    if index_type == "post_base":
        post_values = np.array([r["Post_Hardness"] for r in result])
    elif index_type == "pre_base":
        post_values = np.array([-r["Post_Hardness"] for r in result])
    else:
        raise RuntimeError(f"Unknown index_type: {index_type}")

    # -------------------------------------------------
    # high / low 모드
    # -------------------------------------------------
    if hardness_target in ["high", "low"]:
        if hardness_target == "high":
            hardness_threshold = np.percentile(post_values, 80)
            pass_idx = [i for i, v in enumerate(post_values) if v >= hardness_threshold]
        elif hardness_target == "low":
            hardness_threshold = np.percentile(post_values, 20)
            pass_idx = [i for i, v in enumerate(post_values) if v <= hardness_threshold]

        print(f"  └─ Threshold value   : {hardness_threshold:.6f}")
        pass_idx_set = set(pass_idx)

        print("=" * 50)
        pbar = tqdm(
            enumerate(query_vector_npy),
            total=len(query_vector_npy),
            desc="[Hardness aware query re-generation]"
        )
        give_up = 0
        max_attempt = 10

        for idx, query_vector in pbar:
            if idx in pass_idx_set:
                continue
            attempt = 1
            min_hardness = 20
            min_query_label = []
            max_hardness = 0
            max_query_label = []
            query_missing_prob = np.random.rand(num_attribute)

            while True:
                if index_type == "post_base" and hardness_target == "high":
                    query_missing_prob = query_missing_prob / attempt
                else:
                    query_missing_prob = 1 - (1 - query_missing_prob) ** (attempt * 0.5)

                if base_mode == "load":
                    query_label = blg.sample_query_label_from_base(
                        non_empty_payloads,
                        attr_keys,
                        query_missing_prob,
                    )
                else:
                    query_label_arr = blg.generate_query_payloads(
                        1,
                        num_attribute,
                        cardinality,
                        query_missing_prob,
                        "random",
                        allowed_values_per_attr=allowed_values_per_attr,
                    )
                    query_label_payload = blg.payloads_to_dicts(
                        query_label_arr,
                        include_missing=False,
                    )
                    query_label = query_label_payload[0]

                conditions = {"and": []}
                for key, value in query_label.items():
                    conditions["and"].append({key: {"match": {"value": value}}})

                test = {"query": query_vector, "conditions": conditions}
                estimator._filter_ids_by_condition(test)
                if len(estimator.filtered_ids) < 1:
                    continue
                attempt += 1
                estimator.compute_H_cover()
                curr_hardness = estimator.compute_post_hardness(test)
                if index_type == "post_base":
                    c_h = curr_hardness["Post_Hardness"]
                    if max_hardness < c_h:
                        max_hardness = c_h
                        max_query_label = query_label
                elif index_type == "pre_base":
                    c_h = -curr_hardness["Post_Hardness"]
                    if min_hardness > c_h:
                        min_hardness = c_h
                        min_query_label = query_label

                if hardness_target == "high" and c_h >= hardness_threshold:
                    valid_query_labels[idx] = query_label
                    post_values[idx] = c_h
                    pbar.set_postfix({"give_up": give_up, "passed hardness": c_h})
                    break
                elif hardness_target == "low" and c_h <= hardness_threshold:
                    valid_query_labels[idx] = query_label
                    post_values[idx] = c_h
                    pbar.set_postfix({"give_up": give_up, "passed hardness": c_h})
                    break

                if attempt > max_attempt:
                    give_up += 1
                    if index_type == "post_base":
                        valid_query_labels[idx] = max_query_label
                        pbar.set_postfix({"give_up": give_up, "give_up_at": max_hardness})
                        post_values[idx] = max_hardness
                    elif index_type == "pre_base":
                        valid_query_labels[idx] = min_query_label
                        pbar.set_postfix({"give_up": give_up, "give_up_at": min_hardness})
                        post_values[idx] = min_hardness
                    break
                else:
                    pbar.set_postfix({"give_up": give_up, "failed hardness": c_h})
                    continue

        print(f"  ├─ Query generation done ✅")

    # -------------------------------------------------
    # range 모드
    # -------------------------------------------------
    elif hardness_target == "range":
        h_min, h_max = hardness_range
        print(f"  ├─ Target hardness range : [{h_min:.6f}, {h_max:.6f}]")

        pass_idx = [
            i for i, v in enumerate(post_values)
            if (h_min <= v <= h_max)
        ]
        pass_idx_set = set(pass_idx)
        print(f"  ├─ Initially in-range queries : {len(pass_idx):,}/{num_query:,}")

        print("=" * 50)
        pbar = tqdm(
            enumerate(query_vector_npy),
            total=len(query_vector_npy),
            desc="[Hardness aware query re-generation (range)]"
        )
        give_up = 0
        max_attempt = 10

        for idx, query_vector in pbar:
            if idx in pass_idx_set:
                continue

            attempt = 1
            best_label = None
            best_h = None
            best_dist = float("inf")
            query_missing_prob = np.random.rand(num_attribute)

            while True:
                prev_h = post_values[idx]
                if prev_h < h_min:
                    query_missing_prob = query_missing_prob / attempt
                elif prev_h > h_max:
                    query_missing_prob = 1 - (1 - query_missing_prob) ** (attempt * 0.5)
                else:
                    query_missing_prob = 1 - (1 - query_missing_prob) ** (attempt * 0.5)

                if base_mode == "load":
                    query_label = blg.sample_query_label_from_base(
                        non_empty_payloads,
                        attr_keys,
                        query_missing_prob,
                    )
                else:
                    query_label_arr = blg.generate_query_payloads(
                        1,
                        num_attribute,
                        cardinality,
                        query_missing_prob,
                        "random",
                        allowed_values_per_attr=allowed_values_per_attr,
                    )
                    query_label_payload = blg.payloads_to_dicts(
                        query_label_arr,
                        include_missing=False,
                    )
                    query_label = query_label_payload[0]

                conditions = {"and": []}
                for key, value in query_label.items():
                    conditions["and"].append({key: {"match": {"value": value}}})

                test = {"query": query_vector, "conditions": conditions}
                estimator._filter_ids_by_condition(test)
                if len(estimator.filtered_ids) < 1:
                    continue

                attempt += 1
                estimator.compute_H_cover()
                curr_hardness = estimator.compute_post_hardness(test)

                if index_type == "post_base":
                    c_h = curr_hardness["Post_Hardness"]
                else:
                    c_h = -curr_hardness["Post_Hardness"]

                if h_min <= c_h <= h_max:
                    dist = 0.0
                elif c_h < h_min:
                    dist = h_min - c_h
                else:
                    dist = c_h - h_max

                if dist < best_dist:
                    best_dist = dist
                    best_label = query_label
                    best_h = c_h

                if h_min <= c_h <= h_max:
                    valid_query_labels[idx] = query_label
                    post_values[idx] = c_h
                    pbar.set_postfix({"give_up": give_up, "hit": c_h})
                    break

                if attempt > max_attempt:
                    give_up += 1
                    if best_label is not None:
                        valid_query_labels[idx] = best_label
                        post_values[idx] = best_h
                        pbar.set_postfix({"give_up": give_up, "closest": best_h})
                    else:
                        valid_query_labels[idx] = query_label
                        post_values[idx] = c_h
                        pbar.set_postfix({"give_up": give_up, "fallback": c_h})
                    break

                pbar.set_postfix({"give_up": give_up, "last": c_h})
                continue

        print(f"  ├─ Range-based query generation done ✅")

    # -------------------------------------------------
    # match_pdf 모드 (target hardness 분포와 매칭)
    # -------------------------------------------------
    elif hardness_target == "match_pdf":
        print("  ├─ Loading target hardness distribution for MATCH_PDF mode...")
        bin_edges, bin_centers, target_counts = blg.load_target_hardness_distribution(
            args.target_hardness_json,
            args.num_bins,
            index_type,
            num_query,
            bin_range=None,
        )

        # 남은 슬롯
        remaining = target_counts.astype(int).copy()
        num_bins = args.num_bins

        print(f"  ├─ Target bin counts loaded. sum={int(remaining.sum())} (should be {num_query})")

        # -------------------------------
        # helpers
        # -------------------------------
        def _bin_index_of_value(x: float) -> int:
            # x가 어느 bin에 속하는지 (0..num_bins-1)
            # np.digitize는 edge 기준. right=False면 edge[i] <= x < edge[i+1]로 잡힘.
            b = int(np.digitize([x], bin_edges, right=False)[0]) - 1
            if b < 0:
                b = 0
            if b >= num_bins:
                b = num_bins - 1
            return b

        def _dist_to_interval(x: float, lo: float, hi: float) -> float:
            if lo <= x <= hi:
                return 0.0
            if x < lo:
                return lo - x
            return x - hi

        def _nearest_available_bin(x: float) -> tuple[int, float]:
            """
            남은 슬롯이 있는 bin 중에서, x가 그 bin interval에 얼마나 가까운지 dist 최소인 bin 반환
            return: (best_bin, best_dist)
            """
            best_b = None
            best_d = float("inf")

            # 남은 bin이 아주 적을 때도 안전하게
            for b in range(num_bins):
                if remaining[b] <= 0:
                    continue
                lo = float(bin_edges[b])
                hi = float(bin_edges[b + 1])
                d = _dist_to_interval(x, lo, hi)

                if d < best_d:
                    best_d = d
                    best_b = b
                elif d == best_d and best_b is not None:
                    # tie-break: remaining이 더 많은 bin 우선 (조금이라도 안정화)
                    if remaining[b] > remaining[best_b]:
                        best_b = b

            # 이론적으로 remaining.sum()==0이면 여기 오면 안됨
            if best_b is None:
                # fallback: 그냥 x의 bin
                best_b = _bin_index_of_value(x)
                best_d = 0.0
            return best_b, best_d

        # -------------------------------
        # main loop
        # -------------------------------
        give_up = 0
        max_attempt = 20

        pbar = tqdm(
            enumerate(query_vector_npy),
            total=len(query_vector_npy),
            desc="[Hardness PDF matching (slot-driven)]"
        )

        for idx, query_vector in pbar:
            # 이미 어떤 bin이든 채울 필요는 있지만,
            # 만약 remaining이 다 찼으면(=0) 바로 종료 가능.
            if int(remaining.sum()) <= 0:
                break

            # 현재 hardness로 이미 "빈 슬롯 있는 bin"에 들어가면 그대로 채택 (무료 패스)
            cur_h = float(post_values[idx])
            cur_bin = _bin_index_of_value(cur_h)

            if remaining[cur_bin] > 0:
                remaining[cur_bin] -= 1
                pbar.set_postfix({"give_up": give_up, "fill_bin": cur_bin, "h": f"{cur_h:.3f}", "rem": int(remaining.sum())})
                continue

            # 아니면 생성하면서 slot이 있는 bin을 "맞추면" 채택
            best_label = valid_query_labels[idx]
            best_h = cur_h

            # best_h 기준 "남은 bin들 중 가장 가까운 bin" 거리
            best_bin, best_dist = _nearest_available_bin(best_h)

            # attempt용 missing base: 현재 remaining 분포를 이용해 “희소한 bin”쪽으로 너무 고집하지 않도록
            # (너무 aggressive하면 특정 bin에 고정되면서 실패율 증가)
            # 여기서는 일단 단순하게 0.2~0.8 사이 랜덤 기반 + attempt 조정
            base_missing0 = float(np.clip(0.2 + 0.6 * np.random.rand(), 0.05, 0.95))

            accepted = False
            for attempt in range(1, max_attempt + 1):
                # target을 미리 정하지 말고,
                # 현재 best_h가 가까워야 하는 방향(가까운 available bin)으로 missing을 약하게 조정
                # - 더 hard(큰 hardness) 쪽으로 가고싶으면 missing↑
                # - 더 easy(작은 hardness) 쪽으로 가고싶으면 missing↓
                target_b, _ = _nearest_available_bin(best_h)
                target_center = float(bin_centers[target_b])

                base_missing = base_missing0
                if best_h < target_center:
                    # 아직 easy → 더 hard하게
                    base_missing = min(0.95, base_missing - 0.05 * attempt)
                else:
                    # 아직 hard → 더 easy하게
                    base_missing = max(0.05, base_missing + 0.05 * attempt)

                query_missing_prob = np.full(num_attribute, float(np.clip(base_missing, 0.05, 0.95)), dtype=float)

                # label 생성
                if base_mode == "load":
                    query_label = blg.sample_query_label_from_base(
                        non_empty_payloads,
                        attr_keys,
                        query_missing_prob,
                    )
                else:
                    query_label_arr = blg.generate_query_payloads(
                        1,
                        num_attribute,
                        cardinality,
                        query_missing_prob,
                        "random",
                        allowed_values_per_attr=allowed_values_per_attr,
                    )
                    query_label_payload = blg.payloads_to_dicts(
                        query_label_arr,
                        include_missing=False,
                    )
                    query_label = query_label_payload[0]

                conditions = {"and": []}
                for key, value in query_label.items():
                    conditions["and"].append({key: {"match": {"value": value}}})

                test = {"query": query_vector, "conditions": conditions}

                estimator._filter_ids_by_condition(test)
                if len(estimator.filtered_ids) < 1:
                    continue

                estimator.compute_H_cover()
                curr_hardness = estimator.compute_post_hardness(test)
                if index_type == "post_base":
                    c_h = float(curr_hardness["Post_Hardness"])
                else:
                    c_h = float(-curr_hardness["Post_Hardness"])

                b = _bin_index_of_value(c_h)

                # 1) 이 bin에 빈자리가 있으면 즉시 채택
                if remaining[b] > 0:
                    valid_query_labels[idx] = query_label
                    post_values[idx] = c_h
                    remaining[b] -= 1
                    accepted = True
                    pbar.set_postfix({"give_up": give_up, "fill_bin": b, "h": f"{c_h:.3f}", "rem": int(remaining.sum())})
                    break

                # 2) 없으면, 남은 bin 중 가장 가까운 bin 기준 dist로 best 업데이트
                nb, dist = _nearest_available_bin(c_h)
                if dist < best_dist:
                    best_dist = dist
                    best_label = query_label
                    best_h = c_h
                    best_bin = nb

                pbar.set_postfix({"give_up": give_up, "trial_h": f"{c_h:.3f}", "best_dist": f"{best_dist:.3f}", "rem": int(remaining.sum())})

            # 3) max_attempt 내에 빈 bin을 직접 못 맞췄으면 best를 "가장 가까운 남은 bin"에 할당
            if not accepted:
                give_up += 1
                # best_bin은 remaining>0인 bin 중에서 잡히도록 설계됨
                if remaining[best_bin] <= 0:
                    # 혹시나 race/tie로 0이 됐으면 다시 구함
                    best_bin, _ = _nearest_available_bin(best_h)

                valid_query_labels[idx] = best_label
                post_values[idx] = float(best_h)

                if remaining[best_bin] > 0:
                    remaining[best_bin] -= 1

                pbar.set_postfix({"give_up": give_up, "forced_bin": best_bin, "h": f"{best_h:.3f}", "rem": int(remaining.sum())})

        print(f"  ├─ match_pdf (slot-driven) done ✅")
        print(f"  ├─ give_up count: {give_up:,}")
        print(f"  └─ remaining slots (should be 0): {int(remaining.sum())}")

    # -------------------------------------------------
    # hardness 값 저장 + query_label / tests 재검증
    # -------------------------------------------------
    hardness_txt_path = os.path.join(mid_dir, "post_hardness.txt")
    np.savetxt(hardness_txt_path, post_values)
    print(f"  ├─ Saved hardness values → {hardness_txt_path}")

    blg.save_vector_label(
        valid_query_labels,
        mapping_path,
        os.path.join(mid_dir, "query_label.txt"),
        silently=True
    )
    save_path = os.path.join(mid_dir, "query_label.txt")
    print(f"  ✅ Query labels save: {save_path}")

    print(f"\n[Validation Check]")
    query_labels = []
    with open(os.path.join(mid_dir, "query_label.txt"), 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts and parts[0] != '':
                query_labels.append([int(x) for x in parts])

    blg.save_compute_groundtruth(
        os.path.join(mid_dir, "base_label.txt"),
        os.path.join(mid_dir, "query_label.txt"),
        os.path.join(mid_dir, "gt.bin"),
        query_vector_bin,
        base_vector_bin,
        build_dir="./utils/compute_groundtruth",
        num_threads=28
    )

    gt_indices = blg.load_groundtruth_bin(os.path.join(mid_dir, "gt.bin"), num_query)
    gt_indices = gt_indices.tolist()

    valid_indices = [
        i for i, sublist in enumerate(gt_indices)
        if not all(v == target_value for v in sublist)
    ]

    print(f"  └─ Number of valid GT entries after regeneration : {len(valid_indices):,}")

    conditions = []
    for query_label in valid_query_labels:
        temp = {"and": []}
        for key, value in query_label.items():
            temp["and"].append({key: {"match": {"value": value}}})
        conditions.append(temp)

    tests = []
    query_vector_array = query_vector_npy.tolist()
    for query, condition, gt in zip(query_vector_array, conditions, gt_indices):
        temp = {
            "query": query,
            "conditions": condition,
            "closest_ids": gt
        }
        tests.append(temp)

    with open(os.path.join(save_dir, "tests.jsonl"), "w", encoding="utf-8") as f:
        for item in tests:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[Output]")
    print(f"  └─ Saved test file     → {os.path.join(save_dir, 'tests.jsonl')}")
    print("✅ Hardness aware query generation completed successfully.")


if __name__ == "__main__":
    main()
