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

# 경로 정의
##################################################################################################

trade_off_UNG = {}
dataset_list = ["arxiv","LAION1M", "tripclick", "yfcc" ]
for dataset in dataset_list:
    original_data_path = f"/home/ec2-user/hybrid_hardness/semi-real/filterbenchmark/{dataset}"
    
    data_path = original_data_path
    hardness_path = os.path.join(data_path, "hardness_format")
    query_fvecs_path = os.path.join(original_data_path, f"{dataset}_query_equal.fvecs")
    base_label_path = os.path.join(data_path, "label_base.txt")
    query_label_path = os.path.join(data_path, f"{dataset}_query_equal.txt")
    base_vector_path = os.path.join(data_path, f"{dataset}_base.bin")
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

    tests_file = f"{hardness_path}/tests.jsonl"
    tests = []
    with open(tests_file, "r") as f:
        for line in f:
            tests.append(json.loads(line))
    ground_truth = [test["closest_ids"] for test in tests]

    print(f"Loaded {len(tests)} tests")

    num_query = len(tests)
    batch_size = int(num_query / 10)


    for sort_hardness in ["Post_Hardness","selectivity", "correlation"]: ##############################################################
    # for sort_hardness in ["Post_Hardness"]:

        if sort_hardness == "selectivity" or sort_hardness == "correlation" or sort_hardness == "select_corr_combine":
            baseline = 1
        else:
            baseline = 0

        if baseline == 1:
            hardness_json_path = os.path.join(data_path, f"hardness/hardness_baseline_{num_query}.json")

        else:
            hardness_json_path = os.path.join(data_path, f"hardness/hardness_v5.1_{num_query}.json")
        
        # step 1: 데이터 로딩 및 정렬
        queries = read_fvecs(query_fvecs_path)                      # (10000, 128)

        labels = []

        with open(query_label_path, "r") as f:
            for line in f:
                numbers = [int(x) for x in line.strip().split(",") if x.strip()]
                labels.append(numbers)

        with open(hardness_json_path, "r") as f:
            hardness_data = json.load(f)


        hardness = np.array([item[sort_hardness] for item in hardness_data])


        sorted_idx = np.argsort(hardness)

        # step 2: batching
        batch_size = batch_size
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
            
            if dataset != "sift1m":
                # (2) compute_groundtruth
                subprocess.run([
                    "/home/ec2-user/hybrid_hardness/methods/Unified-Navigating-Graph/build/tools/compute_groundtruth",
                    "--data_type", "float",
                    "--dist_fn", "L2",
                    "--scenario", "containment",
                    "--K", "10",
                    "--num_threads", "16",
                    "--base_bin_file", base_vector_path,
                    "--base_label_file", base_label_path,
                    "--query_bin_file", bin_file,
                    "--query_label_file", label_file,
                    "--gt_file", gt_file,
                    "--gt_counts_file", gt_counts_file
                ], check=True)
                print("gt calculate done")
                # print(idx)


            print(f"[✓] batch{i} 완료")







        # search_UNG_index binary 위치
        binary = "/home/ec2-user/hybrid_hardness/methods/Unified-Navigating-Graph/build/apps/search_UNG_index"

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