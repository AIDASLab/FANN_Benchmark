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
import matplotlib.pyplot as plt
import os
import json
import numpy as np

import subprocess


# 경로 정의
##################################################################################################
dataset_name = "sift1m"
dataset_name = "glove1m"
dataset_name = "gist1m"
# dataset_name = "HnM"
# dataset_name = "ArXiv"

# num_attribute = 10
# cardinality = [12] * num_attribute
# distribution = "zipf"
# distribution = "random"

# sort_hardness = "Hardness"
# sort_hardness = "Pre_Hardness"
# sort_hardness = "Post_Hardness"

# sort_hardness = "selectivity"
# sort_hardness = "correlation"
# sort_hardness = "select_corr_combine"


##################################################################################################
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
        original_data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset_name}_original"
        cardinality = '_'.join(str(c) for c in card)
        correlation = '_'.join(str(c) for c in corr)
        missing_prob = '_'.join(str(c) for c in missing)
        data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset_name}_A{num_attribute}_{cardinality}_{base_distribution}_{missing_prob}_{correlation}"
        hardness_path = os.path.join(data_path, "hardness_format")
        hardness_json_path = os.path.join(data_path, "hardness")
        mid_path = os.path.join(data_path, "mid_format")
        NHQ_path = os.path.join(data_path, "nhq_format")
    elif dataset_name == "HnM":
        original_data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/HnM/mid_format"
        data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/HnM"
        hardness_path = f"/home/ec2-user/hybrid_hardness/Benchmark/HnM/hardness_format"
        hardness_json_path = f"/home/ec2-user/hybrid_hardness/Benchmark/HnM/hardness"
        mid_path = f"/home/ec2-user/hybrid_hardness/Benchmark/HnM/mid_format"
        NHQ_path = f"/home/ec2-user/hybrid_hardness/Benchmark/HnM/nhq_format"
    elif dataset_name == "ArXiv":
        original_data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/ArXiv/medium/include/mid_format"
        data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/ArXiv/medium/include"
        hardness_path = f"/home/ec2-user/hybrid_hardness/Benchmark/ArXiv/medium/include/hardness_format"
        hardness_json_path = f"/home/ec2-user/hybrid_hardness/Benchmark/ArXiv/medium/include/hardness"
        mid_path = original_data_path
        NHQ_path = f"/home/ec2-user/hybrid_hardness/Benchmark/ArXiv/medium/include/nhq_format"


    ## index를 build 하기 위한 전체 base vector, label 만드는 블록
    ## 이 코드는 sift1m 만을 처리하기 위한 코드임
    ## 다른 데이터셋을 처리하기 위해서는 filter mapping 하는 파이프라인을 짜야함


    if dataset_name == "sift1m" or dataset_name == "glove1m" or dataset_name == "gist1m":
        payloads_path = os.path.join(hardness_path, 'payloads.jsonl')
        output_path = os.path.join(mid_path, 'base_label_NHQ.txt')

        with open(payloads_path, 'r') as fin:
            lines = fin.readlines()

        with open(output_path, 'w') as fout:
            fout.write(f"{len(lines)} {num_attribute}\n")  # 첫 줄: 데이터 개수, 속성 개수
            for line in lines:
                payload = json.loads(line)
                row = []
                for i in range(1, num_attribute + 1):
                    key = f"label_{i}"
                    value = payload.get(key, 0)
                    row.append(str(value))
                fout.write(" ".join(row) + "\n")


        tests_path = os.path.join(hardness_path, 'tests.jsonl')
        output_path = os.path.join(mid_path, 'query_label_NHQ.txt')

        with open(tests_path, 'r') as fin, open(output_path, 'w') as fout:
            for line in fin:
                if not line.strip():
                    continue
                test = json.loads(line)
                labels = [0] * num_attribute

                conditions = test["conditions"].get("and", [])
                for cond in conditions:
                    for key, value in cond.items():
                        if key.startswith("label_") and "match" in value:
                            idx = int(key.split("_")[1]) - 1
                            labels[idx] = value["match"]["value"]

                fout.write(" ".join(str(v) for v in labels) + "\n")

    elif dataset_name == "HnM" or dataset_name == "ArXiv":
        payloads_path = os.path.join(mid_path, 'base_label.txt')
        output_path = os.path.join(mid_path, 'base_label_NHQ.txt')

        with open(payloads_path, 'r', encoding='utf-8') as fin:
            lines = [line.strip() for line in fin if line.strip()]

        num_attribute = max(len(line.split(',')) for line in lines)

        with open(output_path, 'w', encoding='utf-8') as fout:
            fout.write(f"{len(lines)} {num_attribute}\n")  # 첫 줄 추가
            for line in lines:
                fout.write(line + "\n")


        groups = []
        value_to_group = {}
        with open(payloads_path, "r", encoding="utf-8") as f:
            for line in f:
                nums = [int(x) for x in line.strip().split(",") if x]
                groups.append(nums)
                for idx, val in enumerate(nums):
                    value_to_group[val] = (len(groups)-1, idx)  # (행번호, 열번호)

        num_groups = len(groups)
        group_sizes = [len(g) for g in groups]



        payloads_path = os.path.join(mid_path, 'base_label.txt')
        output_path = os.path.join(mid_path, 'base_label_NHQ.txt')

        with open(payloads_path, 'r', encoding='utf-8') as fin:
            lines = [line.strip() for line in fin if line.strip()]

        num_attribute = max(len(line.split(',')) for line in lines)

        with open(output_path, 'w', encoding='utf-8') as fout:
            fout.write(f"{len(lines)} {num_attribute}\n")  # 첫 줄 추가
            for line in lines:
                fout.write(line + "\n")

        base_labels = [[int(x) for x in line.strip().split(",")] for line in lines]

        mapping = {}
        for line in base_labels:
            for i, label in enumerate(line):
                if i not in mapping.keys():
                    mapping[i] = set()
                mapping[i].add(label)


        query_path = os.path.join(mid_path, 'query_label.txt')
        output_path = os.path.join(mid_path, 'query_label_NHQ.txt')

        # 1) 역매핑: 라벨 -> 컬럼 인덱스
        label2col = {}
        for c, s in mapping.items():
            for lab in s:
                label2col[lab] = c

        num_cols = len(mapping)  # 원하는 최종 열 개수

        # 2) query_label.txt 읽어서 "열 수 = num_cols" 벡터로 변환
        with open(query_path, 'r', encoding='utf-8') as fin:
            q_lines = [line.strip() for line in fin if line.strip()]

        with open(output_path, 'w', encoding='utf-8') as fout:
            # # 첫 줄: (쿼리 수, 컬럼 수)
            # fout.write(f"{len(q_lines)} {num_cols}\n")

            for line in q_lines:
                nums = [int(x) for x in line.split(',') if x]
                vec = [0] * num_cols  # 기본 0 패딩
                for n in nums:
                    c = label2col.get(n)
                    if c is not None:
                        vec[c] = n        # 해당 컬럼에 그 숫자 기록
                        # 만약 한 컬럼에 여러 숫자가 들어올 수 있으면, 규칙에 따라 처리(첫 번째만, 최대값, 등)
                fout.write(','.join(map(str, vec)) + '\n')

        print(f"[✓] 변환 완료 → {output_path}  (queries={len(q_lines)}, cols={num_cols})")

        tests_path = os.path.join(hardness_path, 'tests.jsonl')



    tests = []
    with open(tests_path, 'r') as fin:
        for line in fin:
            tests.append(json.loads(line))

    print(f"Loaded {len(tests)} tests")

    gt_list = []
    for test in tests:
        gt_list.append(test["closest_ids"])






    # 쿼리 벡터 로드 (.fvecs 포맷)
    def load_fvecs(filename):
        with open(filename, 'rb') as f:
            dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
            f.seek(0)
            all_data = []
            while True:
                header = f.read(4)
                if not header:
                    break
                d = int(np.frombuffer(header, dtype=np.int32)[0])
                vec = np.frombuffer(f.read(4 * d), dtype=np.float32)
                all_data.append(vec)
        return np.stack(all_data)

    if dataset_name == "sift1m" or dataset_name == "glove1m" or dataset_name == "gist1m":
        query_vectors = load_fvecs(os.path.join(original_data_path, f"{dataset_name}_query.fvecs"))
    elif dataset_name == "HnM" or dataset_name == "ArXiv":
        query_vectors = load_fvecs(os.path.join(original_data_path, "query_vector.fvecs"))


    print(data_path)
# ################### index build
#     if dataset_name == "sift1m" or dataset_name == "glove1m" or dataset_name == "gist1m":
#         base_vector = os.path.join(original_data_path, f"{dataset_name}_base.fvecs")
#         base_label = os.path.join(mid_path, "base_label_NHQ.txt")
#     elif dataset_name == "HnM" or dataset_name == "ArXiv" or dataset_name == "mtg-40K":
#         base_vector = os.path.join(original_data_path, "base_vector.fvecs")
#         base_label = os.path.join(mid_path, "base_label_NHQ.txt")

#     os.makedirs(os.path.join(NHQ_path, "NHQ_index"), exist_ok=True)

#     maxm0_list = [10,20,30,40,50]
#     efc_list = [30,50,70,100,150]
#     for maxm0, efc in zip(maxm0_list, efc_list):
#         curr_index_path = os.path.join(NHQ_path, "NHQ_index", f"M{maxm0}_ef{efc}")
#         os.makedirs(os.path.join(curr_index_path), exist_ok=True)

#         index_bin_path = os.path.join(curr_index_path, "index.bin")
#         index_txt_path = os.path.join(curr_index_path, "index.txt")

#         args = [
#             "NHQ-NPG_nsw",
#             base_vector,
#             base_label,
#             index_bin_path,
#             index_txt_path,
#             str(maxm0),
#             str(efc),
#         ]

#         # 인자를 개행으로 합침 (heredoc 효과)
#         input_str = "\n".join(args) + "\n"

#         # 명령어
#         cmd = ["python", "test_hybrid_query.py", "build"]

#         workdir = "/home/ec2-user/hybrid_hardness/methods/NHQ"


#         # subprocess로 실행 (입력은 stdin으로 전달)
#         result = subprocess.run(
#             cmd,
#             input=input_str,
#             text=True,
#             capture_output=True,  # 필요시 출력 저장
#             cwd=workdir
#         )

#         lines = []
#         with open(index_txt_path, "r") as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:   # 빈 줄은 그대로 유지
#                     lines.append("")
#                 else:
#                     lines.append(line + " 0")

#         with open(index_txt_path, "w") as f:
#             f.write("\n".join(lines) + "\n")


#         print(f"{maxm0}_{efc} index save done")







                # 1. Hardness 로드
    for sort_hardness in ["selectivity","Post_Hardness", "correlation"]:


        if sort_hardness == "selectivity" or sort_hardness == "correlation" or sort_hardness == "select_corr_combine":
            baseline = 1
        else:
            baseline = 0

        if baseline == 1:
            with open(os.path.join(hardness_json_path, "hardness_baseline_10000.json"), "r") as f:
                results = json.load(f)    
        else:
            with open(os.path.join(hardness_json_path, "hardness_v5.1_10000.json"), "r") as f:
                results = json.load(f)
        # hardness = np.array([item[sort_hardness] for item in results])

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

        sorted_idx = np.argsort(hardness)  # 쉬운 순서

        # 2. Attribute 로드 및 정제
        with open(os.path.join(mid_path, 'query_label_NHQ.txt'), 'r') as fin:
            lines = [line.strip().replace(',', ' ') for line in fin if line.strip()]
        attr_dim = len(lines[0].split())
        attr_array = np.array([line.split() for line in lines])  # shape (10000, attr_dim), dtype=object

        # 3. Ground Truth 로드
        gt_list = []
        with open(os.path.join(hardness_path, "tests.jsonl"), 'r') as fin:
            for line in fin:
                test = json.loads(line)
                gt_list.append(test["closest_ids"])

        if dataset_name == "sift1m" or dataset_name == "glove1m" or dataset_name == "gist1m":
            gt_array = np.array(gt_list)
        elif dataset_name == "HnM" or dataset_name == "ArXiv":
            # 쿼리별 neighbor 개수의 최대값 (K_max)
            K_max = 10  # 앞의 10개만 사용

            # -1로 패딩된 배열 생성
            gt_array = np.full((len(gt_list), K_max), -1, dtype=np.int32)
            for i, ids in enumerate(gt_list):
                trimmed = ids[:10]  # 앞의 10개만 선택
                gt_array[i, :len(trimmed)] = trimmed

        # 4. 저장 함수
        def save_fvecs(filename, vectors):
            with open(filename, 'wb') as f:
                for vec in vectors:
                    d = np.array([vec.shape[0]], dtype=np.int32)
                    f.write(d.tobytes())
                    f.write(vec.astype(np.float32).tobytes())

        def save_txt(filename, lines):
            with open(filename, 'w') as fout:
                fout.write(f"{len(lines)} {len(lines[0].split())}\n")
                for line in lines:
                    fout.write(line + '\n')

        # def save_ivecs(filename, ivecs):
        #     with open(filename, 'wb') as f:
        #         for row in ivecs:
        #             arr = np.array(row, dtype=np.int32)
        #             K = len(arr)
        #             f.write(np.array([K], dtype=np.int32).tobytes())
        #             f.write(arr.tobytes())

        def save_gt_ivecs(filename, gt_list):
            with open(filename, 'wb') as f:
                for row in gt_list:
                    # 4294967295를 -1로 치환
                    arr = np.array(row, dtype=np.int64)  # int64로 먼저 변환
                    arr[arr == 4294967295] = -1
                    arr = arr.astype(np.int32)           # 최종적으로 int32로 캐스팅
                    K = len(arr)
                    f.write(np.array([K], dtype=np.int32).tobytes())  # 4 bytes: K
                    f.write(arr.tobytes())  

        # 5. Batch로 저장 (각 batch 마다 개별 디렉터리 생성)
        batch_size = 1000
        for batch_num in range(10):
            start = batch_num * batch_size
            end = start + batch_size
            batch_idx = sorted_idx[start:end]

            batch_vecs = query_vectors[batch_idx]
            batch_attrs = attr_array[batch_idx]
            batch_gt = gt_array[batch_idx]
            batch_lines = [' '.join(row) for row in batch_attrs]

            batch_dir = os.path.join(NHQ_path, f"batch{batch_num}")
            os.makedirs(batch_dir, exist_ok=True)

            save_fvecs(os.path.join(batch_dir, "query_vectors.fvecs"), batch_vecs)
            save_txt(os.path.join(batch_dir, "query_labels.txt"), batch_lines)
            save_gt_ivecs(os.path.join(batch_dir, "gt.ivecs"), batch_gt)

            print(f"✅ Saved batch{batch_num} to {batch_dir}")



        import re
        import subprocess

        nhq_format_dir = os.path.join(data_path, "nhq_format")
        nhq_index_root = os.path.join(nhq_format_dir, "NHQ_index")
        output_summary = os.path.join(nhq_format_dir, f"{sort_hardness}_search_results.txt")

        batch_dirs = []
        for i in range(10):
            batch_dirs.append(os.path.join(nhq_format_dir, f"batch{i}"))
            

        rows = []
        header = ["Batch", "M", "ef", "SearchTime", "Accuracy"]

        for i, batch in tqdm(enumerate(batch_dirs)):
            batch_dir = os.path.join(batch)
            #print("batch dir :", batch_dir)
            query_fvecs = os.path.join(batch_dir, "query_vectors.fvecs")
            query_label = os.path.join(batch_dir, "query_labels.txt")
            gt_ivecs = os.path.join(batch_dir, "gt.ivecs")
            batch_num = i

            for idx_dir in sorted(os.listdir(nhq_index_root)):
                idx_path = os.path.join(nhq_index_root, idx_dir)
                if not os.path.isdir(idx_path):
                    continue

                bin_path = os.path.join(idx_path, "index.bin")
                txt_path = os.path.join(idx_path, "index.txt")

                args = [
                    "NHQ-NPG_nsw",
                    bin_path,
                    txt_path,
                    query_fvecs,
                    query_label,
                    gt_ivecs
                ]
                input_str = "\n".join(args) + "\n"
                cmd = ["python", "test_hybrid_query.py", "search"]

                workdir = "/home/ec2-user/hybrid_hardness/methods/NHQ"  
                # subprocess 실행 (결과를 직접 파싱하기 위해 stdout/stderr 모두 캡처)
                result = subprocess.run(
                    cmd,
                    input=input_str,
                    text=True,
                    capture_output=True,
                    cwd=workdir
                )
                output = result.stderr
                # print("Return code:", result.returncode)
                # print("STDOUT:\n", result.stdout)
                # print("STDERR:\n", result.stderr)
                #print(output)
                # index 파라미터 추출
                m_match = re.search(r'M(\d+)_ef(\d+)', idx_dir)
                if not m_match:
                    continue
                M = int(m_match.group(1))
                ef = int(m_match.group(2))
                # print(M,ef)
                # SearchTime, Accuracy 파싱 (출력 포맷에 따라 조정 필요)
                # 아래는 예시: 'SearchTime: x, Accuracy: y'
                found = re.findall(r"Search Time.*?([\d.]+).*?accuracy.*?([\d.]+)", output)
                if not found:
                    # 라인별로 숫자만 나오는 경우, 적절히 파싱해야 함
                    print("not found")
                    continue
                # print("found: ", found)
                for search_time, accuracy in found:
                    rows.append([batch_num, M, ef, float(search_time), float(accuracy)])


        # 파일로 저장
        with open(output_summary, "w") as f:
            f.write(" | ".join(header) + "\n")
            for row in rows:
                f.write(" | ".join(map(str, row)) + "\n")

        print("✅ 모든 결과가 하나의 summary 파일로 저장되었습니다.")





        nhq_format_dir = os.path.join(data_path, "nhq_format")
        #output_summary = os.path.join(nhq_format_dir, "search_results.txt")
        output_summary = os.path.join(nhq_format_dir, f"{sort_hardness}_search_results.txt")
        file_path = output_summary

        num_batches = 10
        qps = [[] for _ in range(num_batches)]
        recall = [[] for _ in range(num_batches)]

        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]  # 첫 줄 헤더 무시

            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.strip().split('|')]
                if len(parts) < 5:
                    continue

                batch_id = int(parts[0])
                search_time = float(parts[3])
                accuracy = float(parts[4])

                qps_value = 1000.0 / search_time if search_time > 0 else 0.0

                qps[batch_id].append(qps_value)
                recall[batch_id].append(accuracy)

        # 시각화
        batch_labels = [f"batch {i}" for i in range(num_batches)]
        colors = plt.get_cmap('tab10', num_batches)

        plt.figure(figsize=(10, 6))
        for i in range(num_batches):
            plt.plot(qps[i], recall[i], '-o', label=batch_labels[i], color=colors(i))

        plt.xlabel("QPS (Queries Per Second)")
        plt.ylabel("Recall")
        plt.title("Recall vs QPS by Batch")
        plt.legend(title="Batch", loc="best", fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        pig_path = os.path.join(NHQ_path, f"{sort_hardness}.png")
        plt.savefig(pig_path, dpi=300)
        # plt.show()

        print(data_path)
        print(sort_hardness)