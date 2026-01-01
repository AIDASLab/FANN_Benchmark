import sys
import argparse
import os
import label_generator.base_label_generator as blg
import Index_analyzer.Post_Filter_Analyzer as PostFilterAnalyzer
import Index_analyzer.Pre_Filter_Analyzer as PreFilterAnalyzer
import Index_analyzer.NHQ_Analyzer as NHQAnalyzer
import Index_analyzer.UNG_Analyzer as UNGAnalyzer
import Index_analyzer.RWALKS_Analyzer as RWALKSAnalyzer
import Index_analyzer.MILVUS_Analyzer as MILVUSAnalyzer
# import Index_analyzer.ACORN_Analyzer as ACORNAnalyzer
import hardness_estimator.calculate_hardness_v5_0 as hd
# import hardness_estimator.calculate_hardness_test as hd
import numpy as np
import json
import subprocess
import random
from tqdm import tqdm


import argparse

def parse_base_complexity(values):
    """
    Parse base complexity parameters.

    Expected input:
        [num_attribute, cardinalities, distribution, (zipf_param), missing_prob, correlations]

    Examples:
        --base_complexity 3 "4,6,8" zipf 1.2 "0.1,0.2,0.3" "0.8,0.6,0.9"
        --base_complexity 3 "4,6,8" random "0.0,0.0,0.0" "0.8,0.6,0.9"

    Returns:
        dict with keys:
            num_attribute, cardinality, distribution, zipf_param, missing_prob, correlation
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



def main():
    parser = argparse.ArgumentParser(description="Hybrid Index Label Generator")
    parser.add_argument("--base_vector_path", required=True, help="Path to base vector .npy file")
    parser.add_argument("--query_vector_path", required=True, help="Path to query vector .npy file")
    parser.add_argument("--index", required=True, help="Index type (e.g., HNSW, IVF, etc.)")
    parser.add_argument("--save_dir", required=True, help="Output directory to save artifacts")

    # Allow 5 (random) or 6 (zipf + param) tokens
    parser.add_argument(
        "--base_complexity",
        nargs="+",
        metavar="BASE_COMPLEXITY_ARGS",
        type=str,
        help=(
            "Base complexity config:\n"
            "  zipf:   NUM_ATTR CARDINALITY zipf ZIPF_PARAM MISSING_PROB CORRELATION\n"
            "  random: NUM_ATTR CARDINALITY random MISSING_PROB CORRELATION\n"
            "Where:\n"
            "  - NUM_ATTR: integer\n"
            "  - CARDINALITY: comma-separated ints (e.g., 4,6,8)\n"
            "  - ZIPF_PARAM: float (e.g., 1.2) [zipf only]\n"
            "  - MISSING_PROB: comma-separated floats in [0,1]\n"
            "  - CORRELATION: comma-separated floats in [0,1]\n"
        ),
        required=True,
    )

    parser.add_argument(
        "--query_complexity",
        choices=["high", "low", "flat", "random"],
        required=True,
        help="Query complexity profile (high | low | flat | random)"
    )

    parser.add_argument(
        "--dev_mode",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS
    )

    args = parser.parse_args()

    # Parse base complexity (supports 5 or 6 args depending on distribution)
    base_complexity = parse_base_complexity(args.base_complexity)

    # Ensure save_dir exists
    os.makedirs(args.save_dir, exist_ok=True)

    base_vector_path = args.base_vector_path
    query_vector_path = args.query_vector_path
    index_method = args.index
    save_dir = args.save_dir
    hardness_target = args.query_complexity
    dev_mode = args.dev_mode

    # Unpack parsed complexity parameters
    num_attribute = base_complexity["num_attribute"]
    cardinality = base_complexity["cardinality"]
    distribution = base_complexity["distribution"]
    zipf_param = base_complexity.get("zipf_param")
    base_missing_prob = base_complexity["missing_prob"]
    correlation = base_complexity["correlation"]


    mid_dir = os.path.join(save_dir, "mid_format")
    os.makedirs(mid_dir, exist_ok=True)
    base_vector_npy = blg.read_fvecs(base_vector_path)
    np.save(os.path.join(mid_dir, "base_vectors.npy"), base_vector_npy)
    query_vector_npy = blg.read_fvecs(query_vector_path)
    np.save(os.path.join(mid_dir, "query_vector.npy"), query_vector_npy)
    
    print("=" * 180)
    print()
    print(f"Hardness Aware Hybrid Search (FANN) Benchmark Generator v.1.1")    
    print()
    
    print("=" * 100)
    # Structured output for configuration check
    print("[Config]")
    print(f"  ├─ [Dataset] → base={base_vector_path} | query={query_vector_path}")
    print(f"  ├─ [Index]   → {index_method}")
    print(f"  ├─ [Output]  → {save_dir}")
    print(f"  ├─ [Querying]→ complexity={hardness_target}")
    print(f"  └─ [Base]    → attr={num_attribute}, card={cardinality}, dist={distribution}, "
          f"zipf_param={zipf_param}, missing={base_missing_prob}, corr={correlation}")


    ########################################################### 
    #### base label generation step
    ########################################################### 
    print("=" * 100)
    print("[Base Label Gerneration Step]")
    print(f"  ├─ 🧩 Base vector loaded: shape = {base_vector_npy.shape}")
    print("  ├─ ⚙️  Now generating base labels...")

    payloads = blg.generate_attribute_payloads(
        num_vectors=len(base_vector_npy),
        num_attributes=num_attribute,
        cardinalities=cardinality,
        base_vectors=base_vector_npy,
        correlations=correlation,
        distribution=distribution,
        zipf_param=1.5,
        missing_prob=base_missing_prob,
        missing_value=-1
    )



    if dev_mode:
        print("  ├─ 🔧 Developer mode enabled: generating payloads with missing values (-1)...")
        base_label_payload_all = blg.payloads_to_dicts(payloads, include_missing=True)
        with open(os.path.join(save_dir, "payloads_all.jsonl"), "w", encoding="utf-8") as f:
            for payload in base_label_payload_all:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(f"  ├─ 💾 Saved full payloads (including missing) → {save_dir}")
        mapping_path_all = blg.build_label_mapping(base_label_payload_all, os.path.join(save_dir, "mapping_all.json"))
        print(f"  ├─ 🧭 Label mapping (full) saved → {mapping_path_all}")

    print("  ├─ ✅ Base label generation completed!")
    base_label_payload = blg.payloads_to_dicts(payloads, include_missing=False)

    with open(os.path.join(save_dir, "payloads.jsonl"), "w", encoding="utf-8") as f:
        for payload in base_label_payload:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"  ├─ 💾 Saved filtered base data payloads → {save_dir}")
    mapping_path = blg.build_label_mapping(base_label_payload, os.path.join(save_dir, "mapping.json"))
    print(f"  ├─ 🧭 Label mapping saved → {mapping_path}")
    print("  ├─ ✅ All base label generation tasks completed.")
    
    
    base_vector_bin = os.path.join(mid_dir, "base_vector.bin")
    query_vector_bin = os.path.join(mid_dir, "query_vector.bin")

    converter = "./utils/fvecs_to_bin"
    subprocess.run([
        converter,
        "--data_type", "float",
        "--input_file", base_vector_path,
        "--output_file", base_vector_bin
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run([
        converter,
        "--data_type", "float",
        "--input_file", query_vector_path,
        "--output_file", query_vector_bin
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if dev_mode :
        print("  ├─ 🔧 Developer mode enabled: saveing base_label_all.txt")
        blg.save_vector_label(base_label_payload_all, mapping_path_all, os.path.join(mid_dir, "base_label_all.txt"),silently=True)
        print("  └─ Saveing base_label.txt")
    blg.save_vector_label(base_label_payload, mapping_path, os.path.join(mid_dir, "base_label.txt"), silently=True)

    
    
    ##########  1) 일단 query를 생성하고, 2) gt > 0 인걸로 1000개씩 만들고, 3) tests.json으로 만들어야함 4) hardness 체크 5) 반복
    ########## hardness 분포를 좀 분석해서 어떤게 어려운지 봐야함
    # for i, test in enumerate(tqdm(tests)):
    #     result.append(estimator.compute_total_hardness(test))
    
    ########################################################### 
    #### query generation step
    ########################################################### 
    target_value = 4294967295
    valid_query_labels = []
    num_query = len(query_vector_npy)
    # print(f"[Query Generation Step] → Target number of queries: {num_query:,}, Selected query complexity: {hardness_target.upper()}")

    iteraion_count = 0

    # print(f"Selected query complexity: {hardness_target.upper()}")

    if hardness_target == "random":
        print(f"[Query Generation Step] → Target number of queries: {num_query:,}, Selected query complexity: {hardness_target.upper()}")
        print("  ├─ No need to initialize index analyzer")
        print("  └─ No need to initialize hardness esimator")
    else:
        
            ########################################################### 
        #### index analyzer step
        ########################################################### 
        # print("  └─ Need to analyze index type")
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
        print(f"[Generating {len(query_vector_npy)} queries to check hardness threshold]")
##################### 일단 1만개 gt가 0보다 크게 만들기
    while len(valid_query_labels) < num_query:
        # print(f"\n[Iteration {iteraion_count}]")
        iteraion_count += 1
        print(f"  ├─ Current valid query labels : {len(valid_query_labels):,}")
        # query_missing_prob = np.random.rand(num_attribute)
        # query_missing_prob = 
        query_missing_prob = np.array([0.5] * num_attribute)
        print(f"  ├─ Generated missing probability : {query_missing_prob}")

        query_label_arr = blg.generate_query_payloads(
            num_query,
            num_attribute,
            cardinality,
            query_missing_prob,
            "random",
        )

        query_label_payload = blg.payloads_to_dicts(query_label_arr, include_missing=False)
        blg.save_vector_label(query_label_payload, mapping_path, os.path.join(mid_dir, "query_label.txt"), silently=True)

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
            build_dir = "./utils/compute_groundtruth",
            num_threads = 4
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

    ####################################################### 여기까지 gt>0인거 1만개 생성하는거 

    print(f"\n[Summary]")
    print(f"  ├─ Total iterations     : {iteraion_count}")
    print(f"  └─ Final valid queries  : {len(valid_query_labels):,}")

    print(f"\n[Ground Truth Check]")
    blg.save_vector_label(valid_query_labels, mapping_path, os.path.join(mid_dir, "query_label.txt"), silently=True)
    # print(f"  ✅ label.txt saved: {save_dir}/query_label.txt")
    blg.save_compute_groundtruth(
        os.path.join(mid_dir, "base_label.txt"),
        os.path.join(mid_dir, "query_label.txt"),
        os.path.join(mid_dir, "gt.bin"),
        query_vector_bin,
        base_vector_bin,
        build_dir = "./utils/compute_groundtruth",
        num_threads = 4
    )
    
    gt_indices = blg.load_groundtruth_bin(os.path.join(mid_dir, "gt.bin"), num_query)
    gt_indices = gt_indices.tolist()

    valid_indices = [
        i for i, sublist in enumerate(gt_indices)
        if not all(v == target_value for v in sublist)
    ]

    
    print(f"  └─ Check complete")

    #######################################################  1만개 gt 다시 확인하기

    conditions = []

    for query_label in valid_query_labels:
        temp = {}
        temp["and"] = []
        for key, value in query_label.items():
            temp["and"].append({key: {"match": {"value": value}}})
        conditions.append(temp)

    ####################################################### save tests

    tests = []
    query_vector_array = query_vector_npy.tolist()
    for query, condition, gt in zip(query_vector_array, conditions, gt_indices):
        temp = {}
        temp["query"] = query
        temp["conditions"] = condition
        temp["closest_ids"] = gt
        tests.append(temp)


    if hardness_target == "random":
        with open(os.path.join(save_dir, "tests.jsonl"), "w", encoding="utf-8") as f:
            for item in tests:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"\n[Output]")
        print(f"  └─ Saved test file     → {os.path.join(save_dir, 'tests.jsonl')}")
        print("✅ Query-label generation completed successfully.")


    else: 
        print("=" * 50)
        
        result = []
        for test in tqdm(tests, desc = f"[Checking hardness threshold for {hardness_target.upper()}]"):
            result.append(estimator.compute_total_hardness(test))

        print(f"  ├─ Threshold calculation done ✅")
        print(f"  ├─ Mode              : {hardness_target.upper()}")
        if hardness_target in ["high", "low"]:
            if index_type == "post_base":
                post_values = np.array([r["Post_Hardness"] for r in result])
            elif index_type == "pre_base":
                post_values = np.array([-r["Post_Hardness"] for r in result])
            if hardness_target == "high":
                hardness_threshold = np.percentile(post_values, 80) 
                pass_idx = [i for i, v in enumerate(post_values) if v >= hardness_threshold]

                

            elif hardness_target == "low":
                hardness_threshold = np.percentile(post_values, 20)  
                pass_idx = [i for i, v in enumerate(post_values) if v <= hardness_threshold]
            print(f"  └─ Threshold value   : {hardness_threshold:.6f}")
            pass_idx_set = set(pass_idx)
            # pass_idx_set = set()
            # hardness_threshold = 0.963

            ##################################################################################################
            # pass_hardness = post_values[pass_idx]
            # pass_hardness = pass_hardness.tolist()
            ###

            ### 남은 것들 생성하기 valid_query_labels
            print("=" * 50)
            # print("\n)
            
            pbar = tqdm(enumerate(query_vector_npy), total=len(query_vector_npy), desc="[Hardness aware query re-generation]")
            give_up = 0
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
                    # print(f"\r[Iteration] Trying #{attempt}", end="", flush=True) #############
                    if index_type =="post_base" and hardness_target == "high":
                        query_missing_prob = query_missing_prob / attempt
                    else:
                        query_missing_prob = 1 - (1 - query_missing_prob) ** (attempt * 0.5)
                    query_label_arr = blg.generate_query_payloads(
                        1,
                        num_attribute,
                        cardinality,
                        query_missing_prob,
                        "random",
                    )
                    query_label_payload = blg.payloads_to_dicts(query_label_arr, include_missing=False)
                    # blg.save_vector_label(query_label_payload, mapping_path, os.path.join(save_dir, "query_label.txt"), silently=True)
                    query_label = query_label_payload[0]
                    conditions = {}
                    conditions["and"] = []
                    for key, value in query_label.items():
                        conditions["and"].append({key: {"match": {"value": value}}})

                    test = {}
                    test["query"] = query_vector 
                    test["conditions"] = conditions
                    # print("test:", test) ##########################################
                    estimator._filter_ids_by_condition(test)
                    if len(estimator.filtered_ids) < 1:
                        continue
                    attempt += 1 ############################################################3
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
                        ##################################################################################################
                        # pass_hardness.append(curr_hardness["Post_Hardness"])
                        pbar.set_postfix({"give up count": give_up, "passed hardness": c_h})
                        ###
                        
                        break
                    elif hardness_target == "low" and c_h <= hardness_threshold:
                        valid_query_labels[idx] = query_label
                        ##################################################################################################
                        # pass_hardness.append(curr_hardness["Post_Hardness"])
                        post_values[idx] = c_h
                        pbar.set_postfix({"give up count": give_up, "passed hardness": c_h})
                        ###
                        break
                    if attempt > 10:
                        give_up += 1
                        if index_type == "post_base":
                            valid_query_labels[idx] = max_query_label
                            pbar.set_postfix({"give up count": give_up, "give up at": max_hardness})
                            post_values[idx] = max_hardness
                        elif index_type == "pre_base":
                            valid_query_labels[idx] = min_query_label
                            pbar.set_postfix({"give up count": give_up, "give up at": min_hardness})
                            post_values[idx] = min_hardness
                        break
                    
                    else:
                        pbar.set_postfix({"give up count": give_up, "failed hardness": c_h})
                        continue
            print(f"  ├─ Query generation done ✅")
                    

        elif hardness_target == "flat":
            if index_type == "post_base":
                post_values = np.array([r["Post_Hardness"] for r in result])
            elif index_type == "pre_base":
                post_values = np.array([-r["Post_Hardness"] for r in result])

            top10_threshold = np.percentile(post_values, 90)
            hardness_threshold = top10_threshold
            print(f"  └─ Threshold value   : {hardness_threshold:.6f}")

            # post_values는 모두 양수이거나 모두 음수라고 가정
            if top10_threshold >= 0:
                # 양수 케이스: 0 ~ threshold 를 5등분
                bin_edges = np.linspace(0.0, top10_threshold, 6)
                print(f"  └─ Binning range: [0.0, {top10_threshold:.6f}]")

                def get_bin_id(h):
                    # 0 미만은 이론상 없지만, 들어오면 첫 bin에 넣음
                    if h < 0:
                        return 0
                    # threshold 이상은 마지막 bin
                    if h >= top10_threshold:
                        return 4
                    # 0 ~ threshold 사이를 5등분한 bin
                    bid = np.searchsorted(bin_edges, h, side="right") - 1
                    return min(max(bid, 0), 4)

            else:
                # 음수 케이스: threshold ~ 0 을 5등분
                bin_edges = np.linspace(top10_threshold, 0.0, 6)
                print(f"  └─ Binning range: [{top10_threshold:.6f}, 0.0]")

                def get_bin_id(h):
                    # threshold 보다 작은 값 → 첫 bin
                    if h <= top10_threshold:
                        return 0
                    # 0 초과는 이론상 없지만, 들어오면 마지막 bin
                    if h > 0:
                        return 4
                    # threshold ~ 0 사이를 5등분한 bin
                    bid = np.searchsorted(bin_edges, h, side="right") - 1
                    return min(max(bid, 0), 4)

            # 나머지 로직은 그대로 사용
            bin_cap = len(query_vector_npy) / 5
            bin_indices = {i: [] for i in range(5)}  # bin_0 ~ bin_4

            # [1] 기존 hardness로 먼저 bin 분류
            print("\n[Hardness aware query generation]")
            for idx, h in enumerate(post_values):
                bid = get_bin_id(h)
                if len(bin_indices[bid]) < bin_cap:
                    bin_indices[bid].append(idx)

            assigned_idx = set().union(*bin_indices.values())

            # [2] 부족한 bin을 조건 만족할 때까지 계속 생성
            for idx, query_vector in tqdm(enumerate(query_vector_npy), total=len(query_vector_npy)):
                if idx in assigned_idx:
                    continue
                if all(len(v) >= bin_cap for v in bin_indices.values()):
                    break

                while True:
                    query_missing_prob = np.random.rand(num_attribute)
                    query_label_arr = blg.generate_query_payloads(
                        1, num_attribute, cardinality, query_missing_prob, "random"
                    )
                    query_label_payload = blg.payloads_to_dicts(query_label_arr, include_missing=False)
                    query_label = query_label_payload[0]

                    conditions = {"and": [{k: {"match": {"value": v}}} for k, v in query_label.items()]}
                    test = {"query": query_vector, "conditions": conditions}

                    estimator._filter_ids_by_condition(test)
                    if len(estimator.filtered_ids) < 1:
                        continue
                    estimator.compute_H_cover()
                    curr_hardness = estimator.compute_post_hardness(test)

                    if index_type == "post_base":
                        c_h = curr_hardness["Post_Hardness"]
                    elif index_type == "pre_base":
                        c_h = -curr_hardness["Post_Hardness"]

                    bid = get_bin_id(c_h)
                    if len(bin_indices[bid]) < bin_cap:
                        bin_indices[bid].append(idx)
                        valid_query_labels[idx] = query_label
                        assigned_idx.add(idx)
                        break

            print(f"  └─ Query generation done ✅")



        if dev_mode == True:
            np.savetxt(os.path.join(mid_dir, "hardness.txt"), post_values)
        blg.save_vector_label(valid_query_labels, mapping_path, os.path.join(mid_dir, "query_label.txt"), silently=True)
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
            build_dir = "./utils/compute_groundtruth",
            num_threads = 4
        )

        gt_indices = blg.load_groundtruth_bin(os.path.join(mid_dir, "gt.bin"), num_query)
        gt_indices = gt_indices.tolist()

        valid_indices = [
            i for i, sublist in enumerate(gt_indices)
            if not all(v == target_value for v in sublist)
        ]

        
        print(f"  └─ Number of valid GT entries after regeneration : {len(valid_indices):,}")

        #######################################################  1만개 gt 다시 확인하기

        conditions = []

        for query_label in valid_query_labels:
            temp = {}
            temp["and"] = []
            for key, value in query_label.items():
                temp["and"].append({key: {"match": {"value": value}}})
            conditions.append(temp)

        ####################################################### save tests

        tests = []
        query_vector_array = query_vector_npy.tolist()
        for query, condition, gt in zip(query_vector_array, conditions, gt_indices):
            temp = {}
            temp["query"] = query
            temp["conditions"] = condition
            temp["closest_ids"] = gt
            tests.append(temp)

        with open(os.path.join(save_dir, "tests.jsonl"), "w", encoding="utf-8") as f:
            for item in tests:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"\n[Output]")
        print(f"  └─ Saved test file     → {os.path.join(save_dir, 'tests.jsonl')}")
        print("✅ Hardness aware query generation completed successfully.")
        ##################################################################################################
        # print(pass_hardness)
        ###






if __name__ == "__main__":
    main()
