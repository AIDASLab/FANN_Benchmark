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
# 경로 정의
##################################################################################################

dataset_name_list = ["arxiv", "LAION1M", "tripclick", "yfcc"]
dataset_name_list = ["LAION1M"]


for dataset_name in dataset_name_list:

    data_path = f"/home/ec2-user/hybrid_hardness/semi-real/filterbenchmark/{dataset_name}"
    hardness_path = os.path.join(data_path, "hardness_format")
    hardness_json_path = os.path.join(data_path, "hardness")
    mid_path = os.path.join(data_path, "mid_format")
    NHQ_path = os.path.join(data_path, "nhq_format")
    mapping_path = os.path.join(data_path, "mid_format/mapping.json")    

    ## index를 build 하기 위한 전체 base vector, label 만드는 블록
    ## 이 코드는 sift1m 만을 처리하기 위한 코드임
    ## 다른 데이터셋을 처리하기 위해서는 filter mapping 하는 파이프라인을 짜야함

    with open(mapping_path, "r") as f:
        mapping = json.load(f)

        
    num_attribute = len(mapping)
    print("number attribute: ", num_attribute)

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

    all_labels = []
    with open(tests_path, 'r') as fin:
        for line in fin:
            if not line.strip():
                continue
            test = json.loads(line)
            labels = [0] * num_attribute
            # conditions = test["conditions"].get("and", [])
            attr_value_pairs = parse_conditions(test["conditions"])
            for attr, value in attr_value_pairs:
                key = f"{attr}:{value}"
                idx = mapping.get(key)
                if idx is not None:
                    # mapping이 1-based 인덱스라고 했으므로 -1
                    labels[int(idx) - 1] = 1
                else:
                    # mapping에 없는 값은 무시
                    pass
            all_labels.append(labels)

    with open(output_path, 'w') as fout:
        num_rows = len(all_labels)
        num_cols = num_attribute
        # fout.write(f"{num_rows} {num_cols}\n")
        for labels in all_labels:
            fout.write(" ".join(str(v) for v in labels) + "\n")





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


    query_vectors = load_fvecs(os.path.join(data_path, f"{dataset_name}_query_equal.fvecs"))

    num_query = len(query_vectors)


    print(data_path)
# ################### index build
    
    # base_vector = os.path.join(data_path, f"{dataset_name}_base.fvecs")
    # base_label = os.path.join(mid_path, "base_label_NHQ.txt")

    # os.makedirs(os.path.join(NHQ_path, "NHQ_index"), exist_ok=True)

    # maxm0_list = [10,20,30,40,50]
    # efc_list = [30,50,70,100,150]
    # for maxm0, efc in zip(maxm0_list, efc_list):
    #     curr_index_path = os.path.join(NHQ_path, "NHQ_index", f"M{maxm0}_ef{efc}")
    #     os.makedirs(os.path.join(curr_index_path), exist_ok=True)

    #     index_bin_path = os.path.join(curr_index_path, "index.bin")
    #     index_txt_path = os.path.join(curr_index_path, "index.txt")

    #     args = [
    #         "NHQ-NPG_nsw",
    #         base_vector,
    #         base_label,
    #         index_bin_path,
    #         index_txt_path,
    #         str(maxm0),
    #         str(efc),
    #     ]

    #     # 인자를 개행으로 합침 (heredoc 효과)
    #     input_str = "\n".join(args) + "\n"

    #     # 명령어
    #     cmd = ["python", "test_hybrid_query.py", "build"]

    #     workdir = "/home/ec2-user/hybrid_hardness/methods/NHQ"


    #     # subprocess로 실행 (입력은 stdin으로 전달)
    #     result = subprocess.run(
    #         cmd,
    #         input=input_str,
    #         text=True,
    #         capture_output=True,  # 필요시 출력 저장
    #         cwd=workdir
    #     )

    #     lines = []
    #     with open(index_txt_path, "r") as f:
    #         for line in f:
    #             line = line.strip()
    #             if not line:   # 빈 줄은 그대로 유지
    #                 lines.append("")
    #             else:
    #                 lines.append(line + " 0")

    #     with open(index_txt_path, "w") as f:
    #         f.write("\n".join(lines) + "\n")


    #     print(f"{maxm0}_{efc} index save done")







                # 1. Hardness 로드
    for sort_hardness in ["selectivity","Post_Hardness", "correlation"]:


        if sort_hardness == "selectivity" or sort_hardness == "correlation" or sort_hardness == "select_corr_combine":
            baseline = 1
        else:
            baseline = 0

        if baseline == 1:
            with open(os.path.join(hardness_json_path, f"hardness_baseline_{num_query}.json"), "r") as f:
                results = json.load(f)    
        else:
            with open(os.path.join(hardness_json_path, f"hardness_v5.1_{num_query}.json"), "r") as f:
                results = json.load(f)
        
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

        
        gt_array = np.array(gt_list)
        
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
        batch_size = int(num_query / 10)
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

                qps_value = (num_query / 10) / search_time if search_time > 0 else 0.0

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