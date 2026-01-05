import os
import numpy as np
import json
import subprocess
import struct
import os
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd

# 경로 정의
##################################################################################################
dataset_name = "sift1m"
dataset_name = "glove1m"
dataset_name = "gist1m"
# dataset_name = "HnM"
# dataset_name = "Arxiv"
# dataset_name ="mtg-40K"

original_data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset_name}_original"
# num_attribute = 3
# cardinality = [6] * num_attribute
# # distribution = "zipf"
# distribution = "random"
# sort_hardness = "Hardness"
# # sort_hardness = "Pre_Hardness"
# # sort_hardness = "Post_Hardness"

# sort_hardness = "selectivity"
# sort_hardness = "correlation"
# sort_hardness = "select_corr_combine"

##################################################################################################
# for num_attribute in [10]:
#     for distribution in ["zipf","random"]:
#         for card in [6]:
#             if num_attribute == 1 and card == 1:
#                 continue
#             if num_attribute == 1 and card == 3:
#                 continue
#             if num_attribute == 3 and card == 1:
#                 continue
#             cardinality = [card] * num_attribute

# for num_attribute, card, base_distribution, corr, missing in zip (
#   [1,3,3,12,12,12,12,3,12,3,3,3,3,3,3],
#   [[12],[6]*3,[12]*3,[1]* 12,[3]* 12,[6]* 12,[12]* 12,   [12]* 3,[3]* 12,   [12]* 3,[12]* 3,[12]* 3,  [12]* 3,[12]* 3,[12]* 3],
#   ["zipf","zipf","zipf","zipf","zipf","zipf","zipf","random","random","zipf","zipf","zipf","zipf","zipf","zipf"],
#   [[0.0],[0.0]*3,[0.0]*3,[0.0]* 12,[0.0]* 12,[0.0]* 12,[0.0]* 12,   [0.0]* 3,[0.0]* 12,   [0.5]* 3,[1.0]* 3,[0.0,0.5,1.0],  [0.0]* 3,[0.0]* 3,[0.0]* 3],
#   [[0.5],[0.5]*3,[0.5]*3,[0.5]* 12,[0.5]* 12,[0.5]* 12,[0.5]* 12,   [0.5]* 3,[0.5]* 12,   [0.5]* 3,[0.5]* 3,[0.5]* 3,  [0.0]* 3,[0.8]* 3,[0.0,0.5,0.8]],
# ):

for num_attribute, card, base_distribution, corr, missing in zip (
  [3,3,3],
  [[12] * 3,[12] * 3,[12] * 3],
  ["random", "zipf", "zipf"],
  [[0.0]*3, [0.5]*3, [0.0]*3],
  [[0.5]*3, [0.0]*3, [0.5]*3],
):

    if dataset_name == "sift1m" or dataset_name == "glove1m" or dataset_name == "gist1m":
        cardinality = '_'.join(str(c) for c in card)
        correlation = '_'.join(str(c) for c in corr)
        missing_prob = '_'.join(str(c) for c in missing)
        data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset_name}_A{num_attribute}_{cardinality}_{base_distribution}_{missing_prob}_{correlation}"
        hardness_path = os.path.join(data_path, "hardness_format")
        query_fvecs_path = os.path.join(original_data_path, f"{dataset_name}_query.fvecs")
        query_label_path = os.path.join(data_path, "mid_format/query_label.txt")
        base_vector_path = os.path.join(data_path, "mid_format/base_vector.bin")
        base_label_path = os.path.join(data_path, "mid_format/base_label_UNG.txt")
        # hardness_json_path = os.path.join(data_path, "hardness/hardness_v3.0_10000.json")

    elif dataset_name == "HnM" or dataset_name =="mtg-40K": 
        data_path=f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset_name}"
        hardness_path = f"/ec2-user/hybrid_hardness/Benchmark/{dataset_name}/hardness_format"
        mid_format = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset_name}/mid_format"
        query_fvecs_path = os.path.join(mid_format, "query_vector.fvecs")
        query_label_path = os.path.join(mid_format, "query_label.txt")
        
        base_vector_fvecs = os.path.join(mid_format, "base_vector.fvecs")
        base_vector_path = os.path.join(mid_format, "base_vector.bin")
        subprocess.run([
            "/home/ec2-user/hybrid_hardness/methods/Unified-Navigating-Graph/build/tools/fvecs_to_bin",
            "--data_type", "float",
            "--input_file", base_vector_fvecs,
            "--output_file", base_vector_path
        ], check=True)
        
        base_label_path = os.path.join(mid_format, "base_label.txt")
        # hardness_json_path = os.path.join(data_path, "hardness/hardness_v3.0_10000.json")

    elif dataset_name == "Arxiv":
        data_path=f"/home/ec2-user/hybrid_hardness/Benchmark/ArXiv/medium/include"
        hardness_path = f"/home/ec2-user/hybrid_hardness/Benchmark/ArXiv/medium/include/hardness_format"
        mid_format = f"/home/ec2-user/hybrid_hardness/Benchmark/ArXiv/medium/include/mid_format"
        query_fvecs_path = os.path.join(mid_format, "query_vector.fvecs")
        query_label_path = os.path.join(mid_format, "query_label.txt")
        
        base_vector_fvecs = os.path.join(mid_format, "base_vector.fvecs")
        base_vector_path = os.path.join(mid_format, "base_vector.bin")
        subprocess.run([
            "/home/ec2-user/hybrid_hardness/methods/Unified-Navigating-Graph/build/tools/fvecs_to_bin",
            "--data_type", "float",
            "--input_file", base_vector_fvecs,
            "--output_file", base_vector_path
        ], check=True)
        
        base_label_path = os.path.join(mid_format, "base_label.txt")
        # hardness_json_path = os.path.join(data_path, "hardness/hardness_v3.0_10000.json")
    ung_root = os.path.join(data_path, "ung_format")
    # ################################## index build
    # binary = "/home/ec2-user/hybrid_hardness/methods/Unified-Navigating-Graph/build/apps/build_UNG_index"
    
    # cmd = [
    #     binary,
    #     "--data_type", "float",
    #     "--dist_fn", "L2",
    #     "--num_threads", "4",
    #     "--max_degree", "32",
    #     "--Lbuild", "100",
    #     "--alpha", "1.2",
    #     "--base_bin_file", base_vector_path,
    #     "--base_label_file", base_label_path,
    #     "--index_path_prefix", ung_root+"/ung_index/",
    #     "--scenario", "general",
    #     "--num_cross_edges", "6"
    # ]

    # try:
    #     subprocess.run(cmd, check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"[에러] 명령어 실행 실패: {e}")




    for sort_hardness in ["Post_Hardness","selectivity", "correlation"]: ##############################################################
    # for sort_hardness in ["Post_Hardness"]:

        if sort_hardness == "selectivity" or sort_hardness == "correlation" or sort_hardness == "select_corr_combine":
            baseline = 1
        else:
            baseline = 0

        if baseline == 1:
            hardness_json_path = os.path.join(data_path, "hardness/hardness_baseline_10000.json")

        else:
            hardness_json_path = os.path.join(data_path, "hardness/hardness_v5.1_10000.json")


        tests_file = f"{hardness_path}/tests.jsonl"
        tests = []
        with open(tests_file, "r") as f:
            for line in f:
                tests.append(json.loads(line))
        ground_truth = [test["closest_ids"] for test in tests]

        print(f"Loaded {len(tests)} tests")

        
        ung_path = "/home/ec2-user/hybrid_hardness/methods/Unified-Navigating-Graph"


        # query.fvecs 로딩 함수
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
            
        def save_groundtruth_bin(filename, batch_gt):
            """
            batch_gt: 파이썬 리스트
            각 원소는 neighbor 인덱스 리스트 (dist 정보 없음)
            """
            records = []
            for neighs in batch_gt:            # batch_gt[i] = [n1, n2, n3, ...]
                for idx in neighs:
                    records.append((np.uint32(idx), np.float32(0.0)))  # dist=0.0 더미값
            
            # (idx:uint32, dist:float32) 구조 정의
            record_dtype = np.dtype([("idx", np.uint32), ("dist", np.float32)])
            data = np.array(records, dtype=record_dtype)
            data.tofile(filename)


        def save_gt_counts(filename, batch_gt):
            """
            batch_gt: 파이썬 리스트
            각 원소는 neighbor 인덱스 리스트 (패딩 포함 가능)
            4294967295 (= 0xFFFFFFFF) 값은 더미로 취급하여 count에서 제외
            """
            DUMMY = 4294967295
            with open(filename, "w") as f:
                for neighs in batch_gt:
                    # 더미값 제외한 개수만 카운트
                    real_count = sum(1 for n in neighs if n != DUMMY)
                    f.write(f"{real_count}\n")


        # step 1: 데이터 로딩 및 정렬
        queries = read_fvecs(query_fvecs_path)                      # (10000, 128)

        labels = []

        with open(query_label_path, "r") as f:
            for line in f:
                numbers = [int(x) for x in line.strip().split(",") if x.strip()]
                labels.append(numbers)

        with open(hardness_json_path, "r") as f:
            hardness_data = json.load(f)

        # 미리 Pre와 Post hardness 값을 추출
        # if baseline == 0:
        #     pre_vals = np.array([item["Pre_Hardness"] for item in hardness_data])
        #     post_vals = np.array([item["Post_Hardness"] for item in hardness_data])

        # min-max normalize (0~1 스케일)
        # def normalize(arr):
        #     if np.max(arr) == np.min(arr):  # 상수 배열인 경우
        #         return np.zeros_like(arr)
        #     return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        # pre_norm = normalize(pre_vals)
        # post_norm = normalize(post_vals)

        if sort_hardness == "mul":
            hardness = np.array([p * q for p, q in zip(pre_vals, post_vals)])

        elif sort_hardness == "sum":
            hardness = np.array([p + q for p, q in zip(pre_vals, post_vals)])

        elif sort_hardness == "harmonic":
            hardness = np.array([
                (2 * p * q) / (p + q) if (p + q) != 0 else 0
                for p, q in zip(pre_vals, post_vals)
            ])

        elif sort_hardness == "geometric":
            hardness = np.array([(p * q) ** 0.5 for p, q in zip(pre_vals, post_vals)])

        elif sort_hardness == "weighted_sum":
            w_post, w_pre = weight_param[0], weight_param[1]
            hardness = np.array([w_pre * p + w_post * q for p, q in zip(pre_vals, post_vals)])

        elif sort_hardness == "min":
            hardness = np.minimum(pre_norm, post_norm)

        elif sort_hardness == "max":
            hardness = np.maximum(pre_norm, post_norm)

        else:
            hardness = np.array([item[sort_hardness] for item in hardness_data])


        sorted_idx = np.argsort(hardness)

        # step 2: batching
        batch_size = 1000
        for i in range(10):
            batch_dir = os.path.join(ung_root, f"batch{i}")
            os.makedirs(batch_dir, exist_ok=True)

            idx = sorted_idx[i * batch_size: (i + 1) * batch_size]
            batch_queries = queries[idx]
            batch_labels = [labels[j] for j in idx]
            # print(batch_labels)

            # save query.fvecs
            with open(os.path.join(batch_dir, "query.fvecs"), "wb") as f:
                for vec in batch_queries:
                    f.write(struct.pack('i', len(vec)))  # dimension
                    f.write(vec.astype(np.float32).tobytes())

            # save query_label.txt
            with open(os.path.join(batch_dir, "query_label.txt"), "w") as f:
                for label_list in batch_labels:
                    line = ",".join(str(x) for x in label_list)
                    f.write(line + "\n")
            
            if dataset_name == "sift1m" or dataset_name == "glove1m" or dataset_name == "gist1m":
                batch_gt = [ground_truth[i] for i in idx]
                # print(batch_gt)
                save_groundtruth_bin(os.path.join(batch_dir, "gt.bin"), batch_gt)
                save_gt_counts(os.path.join(batch_dir, "gt_counts.txt"), batch_gt)

        # step 3: 각 디렉터리 순회하며 fvecs_to_bin → compute_groundtruth
        for i in range(10):
            batch_dir = os.path.join(ung_root, f"batch{i}")
            fvecs_file = os.path.join(batch_dir, "query.fvecs")
            bin_file = os.path.join(batch_dir, "query.bin")
            label_file = os.path.join(batch_dir, "query_label.txt")
            gt_file = os.path.join(batch_dir, "gt.bin")
            gt_counts_file = os.path.join(batch_dir, "gt_counts.txt")

            # (1) fvecs_to_bin
            subprocess.run([
                "/home/ec2-user/hybrid_hardness/methods/Unified-Navigating-Graph/build/tools/fvecs_to_bin",
                "--data_type", "float",
                "--input_file", fvecs_file,
                "--output_file", bin_file
            ], check=True)
            # if dataset_name != "sift1m":
            #     # (2) compute_groundtruth
            #     subprocess.run([
            #         "/home/ec2-user/hybrid_hardness/methods/Unified-Navigating-Graph/build/tools/compute_groundtruth",
            #         "--data_type", "float",
            #         "--dist_fn", "L2",
            #         "--scenario", "containment",
            #         "--K", "10",
            #         "--num_threads", "16",
            #         "--base_bin_file", base_vector_path,
            #         "--base_label_file", base_label_path,
            #         "--query_bin_file", bin_file,
            #         "--query_label_file", label_file,
            #         "--gt_file", gt_file,
            #         "--gt_counts_file", gt_counts_file
            #     ], check=True)

                # print(idx)


            print(f"[✓] batch{i} 완료")







        # search_UNG_index binary 위치
        binary = "/home/ec2-user/hybrid_hardness/methods/Unified-Navigating-Graph/build/apps/search_UNG_index"

        if dataset_name == "HnM":
            K = "10"
        else:
            K = "10"


        # batch0 ~ batch9 순회
        # for i in range(9, -1, -1):
        for i in range(10):
            batch_dir = os.path.join(ung_root, f"batch{i}")
            query_bin = os.path.join(batch_dir, "query.bin")
            query_label = os.path.join(batch_dir, "query_label.txt")
            gt_file = os.path.join(batch_dir, "gt.bin")
            gt_counts_file = os.path.join(batch_dir, "gt_counts.txt")
            result_prefix = batch_dir + "/"  # 결과도 같은 디렉터리

            print(f"[▶] Running batch{i}...")

            cmd = [
                binary,
                "--data_type", "float",
                "--dist_fn", "L2",
                "--num_threads", "16",
                "--K", K,
                "--base_bin_file", base_vector_path,
                "--base_label_file", base_label_path,
                "--query_bin_file", query_bin,
                "--query_label_file", query_label,
                "--gt_file", gt_file,
                "--gt_counts_file", gt_counts_file,
                "--index_path_prefix",  ung_root+"/ung_index/",
                "--result_path_prefix", result_prefix,
                "--scenario", "containment",
                "--num_entry_points", "16",
                "--Lsearch",'150', '100', '50' ,'30', '25'
            ]

            subprocess.run(cmd, check=True)
            print(f"[✓] Done batch{i}")



        num_batches = 10
        save_txt = os.path.join(ung_root, f"{sort_hardness}_search_results.txt")

        with open(save_txt, "w") as f_out:
            f_out.write("batch,QPS,Recall\n")  # 헤더

            for i in range(num_batches):
                csv_path = os.path.join(ung_root, f"batch{i}", "result.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, thousands=",")  # 쉼표 구분자 인식
                    try:
                        df["QPS"] = pd.to_numeric(df["QPS"], errors="coerce")
                        df["Recall"] = pd.to_numeric(df["Recall"], errors="coerce")

                        # txt 파일에 저장
                        for qps, recall in zip(df["QPS"], df["Recall"]):
                            f_out.write(f"{i},{qps},{recall}\n")

                    except Exception as e:
                        print(f"Error parsing batch{i}: {e}")

        print(f"[저장 완료] {save_txt}")




        save_txt = os.path.join(ung_root, f"{sort_hardness}_search_results.txt")

        # search_result.txt 읽기
        df = pd.read_csv(save_txt)

        plt.figure(figsize=(10, 6))

        # batch 별로 플롯
        for batch_id, batch_df in df.groupby("batch"):
            plt.plot(
                batch_df["QPS"],
                batch_df["Recall"],   # 0~1 범위로 스케일
                marker="o",
                label=f"batch{batch_id}"
            )

        plt.xlabel("QPS")
        plt.ylabel("Recall")
        plt.title("Recall vs QPS for Each Batch (from search_result.txt)")
        plt.legend(title="Batch", fontsize="small", ncol=2)
        plt.grid(True)
        plt.tight_layout()

        pig_path = os.path.join(ung_root, f"{sort_hardness}.png")
        plt.savefig(pig_path, dpi=300)
        # plt.show()
        print(data_path)
        print(sort_hardness)