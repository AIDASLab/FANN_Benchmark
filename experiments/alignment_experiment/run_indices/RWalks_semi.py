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
import h5py
import subprocess
import matplotlib.pyplot as plt


dataset_name_list = ["arxiv", "LAION1M", "tripclick", "yfcc"]


for dataset_name in dataset_name_list:

    dataset_path = f"/home/ec2-user/hybrid_hardness/semi-real/filterbenchmark/{dataset_name}"
    DATA_DIR = os.path.join(dataset_path, "hardness_format")


    vectors_file = f"{DATA_DIR}/vectors.npy"
    payloads_file = f"{DATA_DIR}/payloads.jsonl"
    tests_file = f"{DATA_DIR}/tests.jsonl"

    # ------------------------------------
    # 1. Load vectors.npy
    # ------------------------------------
    vectors = np.load(vectors_file)
    print("vectors.shape =", vectors.shape)

    # ------------------------------------
    # 2. Load payloads.jsonl
    # ------------------------------------
    payloads = []
    with open(payloads_file, "r") as f:
        for line in f:
            payloads.append(json.loads(line))

    print(f"Loaded {len(payloads)} payloads")

    # ------------------------------------
    # 3. Load tests.jsonl
    # ------------------------------------
    tests = []
    with open(tests_file, "r") as f:
        for line in f:
            tests.append(json.loads(line))

    num_query = len(tests)
    batch_size = int(len(tests) / 10)
    print(f"Loaded {len(tests)} tests")

    # # ------------------------------------
    # # 예시 출력
    # print("\nSample payload:", payloads[0])
    # print("\nSample test:", tests[0])


    query_vectors = np.array([test["query"] for test in tests])
    ground_truth = [test["closest_ids"] for test in tests]  # 그냥 list로 유지
    # numpy object array로 강제 변환하려면

    gt_lists = [test["closest_ids"] for test in tests]
    maxlen = max(len(x) for x in gt_lists)

    repeated = np.empty((len(gt_lists), maxlen), dtype=np.int32)
    for i, arr in enumerate(gt_lists):
        l = len(arr)
        # 리스트 원소를 반복해서 maxlen까지 채우기
        repeated[i, :] = np.resize(arr, maxlen)
        # 또는 (동일한 결과, 안전한 방법)
        # repeated[i, :] = [arr[j % l] for j in range(maxlen)]

    ground_truth = repeated



    path = os.path.join(dataset_path, "mid_format/mapping.json")

    with open(path, "r") as f:
        mapping = json.load(f)
        
    cardinality = len(mapping)

    multi_hot_vectors = []
    for payload in tqdm(payloads):
        mh_vec = np.zeros(cardinality, dtype=np.int32)
        for k, v in payload.items():
            key = f"{k}:{v}"
            idx = mapping.get(key)
            if idx is not None:
                # mapping은 1-based index로 보이므로 -1
                mh_vec[int(idx) - 1] = 1
            else:
                # mapping에 없는 값은 무시 (필요시 warning)
                pass
        multi_hot_vectors.append(mh_vec)
    multi_hot_vectors = np.stack(multi_hot_vectors)


    def parse_conditions(cond_dict):
        """test['conditions']에서 (attr, value) 쌍을 list로 추출"""
        # 현재 구조는 반드시 'and': [ ... ] 만 지원한다고 가정
        conditions = []
        if "and" in cond_dict:
            for cond in cond_dict["and"]:
                for attr, value_dict in cond.items():
                    # value_dict: {'match': {'value': XXX}}
                    v = value_dict.get("match", {}).get("value")
                    if v is not None:
                        conditions.append((attr, v))
        # 향후 or, not 등 확장 가능
        return conditions

    # onehot_dim = max(label_mapping.values()) + 1

    onehot_vectors = []
    for test in tests:
        oh_vec = np.zeros(cardinality, dtype=np.int32)
        attr_value_pairs = parse_conditions(test["conditions"])
        for attr, value in attr_value_pairs:
            key = f"{attr}:{value}"
            idx = mapping.get(key)
            if idx is not None:
                # mapping이 1-based 인덱스라고 했으므로 -1
                oh_vec[int(idx) - 1] = 1
            else:
                # mapping에 없는 값은 무시
                pass
        onehot_vectors.append(oh_vec)

    onehot_vectors = np.stack(onehot_vectors).astype(np.int8)




    rwalks_path = dataset_path+"/rwalks_format"
    os.makedirs(rwalks_path, exist_ok=True)
    
    for sort_hardness in ["Post_Hardness", "selectivity", "correlation"]:
        if sort_hardness == "selectivity" or sort_hardness == "correlation" or sort_hardness == "select_corr_combine":
            baseline = 1
        else:
            baseline = 0

        # 1. Hardness 값 불러오기
        if baseline == 1:
            hardness_path = os.path.join(dataset_path, f"hardness/hardness_baseline_{num_query}.json")
        else: 
            hardness_path = os.path.join(dataset_path, f"hardness/hardness_v5.1_{num_query}.json")
        with open(hardness_path, "r") as f:
            results = json.load(f)  # 또는 pickle, np.load 등 실제 포맷에 맞게 읽을 것

        # 2. Hardness 값 추출 (tests 순서대로)
        # Pre, Post hardness 배열 추출
        # if baseline == 0:
        #     pre_vals = np.array([item["Pre_Hardness"] for item in results])
        #     post_vals = np.array([item["Post_Hardness"] for item in results])

        #     # min-max normalization 함수
        #     def normalize(arr):
        #         if np.max(arr) == np.min(arr):
        #             return np.zeros_like(arr)
        #         return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        #     pre_norm = normalize(pre_vals)
        #     post_norm = normalize(post_vals)

        if sort_hardness == "mul":
            hardness = np.array([item["Pre_Hardness"] * item["Post_Hardness"] for item in results])

        elif sort_hardness == "sum":
            hardness = np.array([item["Pre_Hardness"] + item["Post_Hardness"] for item in results])

        elif sort_hardness == "harmonic":
            hardness = np.array([
                (2 * item["Pre_Hardness"] * item["Post_Hardness"]) / (item["Pre_Hardness"] + item["Post_Hardness"])
                if (item["Pre_Hardness"] + item["Post_Hardness"]) != 0 else 0
                for item in results
            ])

        elif sort_hardness == "geometric":
            hardness = np.array([
                (item["Pre_Hardness"] * item["Post_Hardness"]) ** 0.5
                for item in results
            ])

        elif sort_hardness == "weighted_sum":
            w_post, w_pre = weight_param[0], weight_param[1]
            hardness = np.array([
                w_pre * item["Pre_Hardness"] + w_post * item["Post_Hardness"]
                for item in results
            ])

        elif sort_hardness == "min":
            hardness = np.minimum(pre_norm, post_norm)

        elif sort_hardness == "max":
            hardness = np.maximum(pre_norm, post_norm)

        else:
            hardness = np.array([item[sort_hardness] for item in results])


        # 3. Hardness 기준으로 정렬된 인덱스
        sorted_idx = np.argsort(hardness)  # 오름차순(쉬운 순서)

        # 4. batching
        batch_size = int(batch_size)
        num_batches = 10

        for batch_num in range(num_batches):
            start = batch_num * batch_size
            end = start + batch_size
            batch_idx = sorted_idx[start:end]
            
            # 배치 추출 (각각 인덱싱)
            batch_test_vectors = query_vectors[batch_idx]
            batch_test_attr_vectors = onehot_vectors[batch_idx]
            batch_neighbors = ground_truth[batch_idx]
            # (필요하면 hardness 자체도 저장 가능: batch_hardness = hardness[batch_idx])

            # 파일명 지정 (ex: hnm_for_RWalks_batch0.h5 ~ hnm_for_RWalks_batch9.h5)
            fname = os.path.join(rwalks_path, f"batch{batch_num}.h5")
            
            with h5py.File(fname, "w") as f:
                f.create_dataset("train_vectors", data=vectors)
                f.create_dataset("test_vectors", data=batch_test_vectors)
                f.create_dataset("train_attr_vectors", data=multi_hot_vectors)
                f.create_dataset("test_attr_vectors", data=batch_test_attr_vectors)
                f.create_dataset("neighbors", data=batch_neighbors)
                # f.create_dataset("hardness", data=hardness[batch_idx])  # 필요하면

            print(f"Saved: {fname}")

        data_dir = rwalks_path
        output_file = f"{data_dir}/" + f"{sort_hardness}_search_results.txt"

        # 명령어 구성
        cmd = ["bash", "/home/ec2-user/hybrid_hardness/methods/RWalks/test_script.sh", data_dir, output_file]
        cwd = "/home/ec2-user/hybrid_hardness/methods/RWalks"
        # 실행
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        print(result.stdout)

        num = 10
        # 파일 경로
        # file_path = os.path.join(rwalks_path, f"{sort_hardness}_search_results.txt")
        file_path = os.path.join(rwalks_path, f"{sort_hardness}_search_results.txt")

        # QPS, Recall 데이터를 저장할 리스트
        qps = [[] for _ in range(num)]
        recall = [[] for _ in range(num)]

        with open(file_path, 'r') as f:
            lines = f.readlines()[2:]  # 헤더 2줄 건너뜀

            for i, line in enumerate(lines):
                parts = line.strip().split('|')
                if len(parts) < 3:
                    continue
                q = int(parts[1].strip())
                r = float(parts[2].strip())
                batch_idx = i // 6
                qps[batch_idx].append(q)
                recall[batch_idx].append(r)

        # 시각화
        batch_labels = [f"batch {i + 1}" for i in range(10)]
        colors = plt.cm.get_cmap('tab10', 10)

        plt.figure(figsize=(10, 6))
        for i in range(num):
            plt.plot(qps[i], recall[i], '-o', label=batch_labels[i], color=colors(i))

        plt.xlabel("QPS")
        plt.ylabel("Recall")
        plt.title("Recall vs QPS by Batch")
        plt.legend(title="Batch", loc="upper right", fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        pig_path = os.path.join(rwalks_path, f"{sort_hardness}.png")
        plt.savefig(pig_path, dpi=300)
        # plt.show()

        print(dataset_path)
        print(sort_hardness)