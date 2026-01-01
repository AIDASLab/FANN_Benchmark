from Index_analyzer import pareto_comp as pc
import os
import numpy as np
import json
import subprocess
import struct
import pandas as pd
import matplotlib.pyplot as plt
import pdb
### python hardness_aware_generator_dev.py --base_vector_path /home/mintaek/hybrid_index/Benchmark/sift1m_original/sift_base.fvecs   --query_vector_path /home/mintaek/hybrid_index/Benchmark/sift1m_original/sift_query_1000.fvecs   --index MILVUS --save_dir /home/mintaek/hybrid_index/Generator/test_1000 --base_complexity 6 "4,6,8,8,8,8" zipf 1.5 "0.3,0.5,0.7,0.3,0.5,0.7" "0.8,0.6,0.9,0.8,0.6,0.9"   --query_complexity high 

##################################################################################################
def Analyze(mode):
    if mode == True:
        return "post_base"
    
    dataset_name = "sift1m"
    
    MILVUS_trade_off = {}
    for d_name, num_attribute, cardinality, distribution in zip(
        ["closer_to_post", "closer_to_pre"],
        [3, 10],
        ([6] * 3, [6] * 10),
        ["random", "zipf"]
    ):
        print(f"\n[Dataset] Loading → {d_name}")
        original_data_path = f"/home/mintaek/hybrid_index/Benchmark/test_dataset/{dataset_name}_original"
        cardi = '_'.join(str(c) for c in cardinality)
        data_path = f"/home/mintaek/hybrid_index/Benchmark/test_dataset/{dataset_name}_A{num_attribute}_{cardi}_{distribution}"
        hardness_path = os.path.join(data_path, "hardness_format")
        
        ### 중간에 생성되는 파일이나 디비는 일단 /home/mintaek/hybrid_index/Generator/Index_analyzer/temp 여기에 저장하도록 해주세요
        #milvus_root = os.path.join(data_path, "milvus_format")
        milvus_root = "/home/mintaek/hybrid_index/Generator/Index_analyzer/temp/milvus_format"


        ### sift1m_A3_6_random (closer_to_post) 과 sift1m_A6_10_zipf (closer_to_pre) 를 불러와서 
        ### Benchmark/{dataset}/hardness_format 을 불러와서 필요한 전처리를 하는 것을 권장
        os.makedirs(os.path.join(milvus_root, "milvus_index"), exist_ok=True)
        base_vector_path = os.path.join(hardness_path, "vectors.npy")
        base_label_path = os.path.join(hardness_path, "payloads.jsonl")
        tests_file = f"{hardness_path}/tests.jsonl"
        tests = []
        with open(tests_file, "r") as f:
            for line in f:
                obj = json.loads(line)

                # 모든 key에서 '-'를 '_'로 치환
                obj = {k.replace("-", "_"): v for k, v in obj.items()}

                tests.append(obj)

        print(f"Loaded {len(tests)} tests")

        # ─────────────────────────────────────────────────────────────
        # [Index Building]
        # ─────────────────────────────────────────────────────────────
        print("[Index] Building MILVUS index …")
        db_path = os.path.join(
            milvus_root,
            "milvus_index",
            f"sift1m_A{num_attribute}_{cardi}_{distribution}.db"
        )

        # 이미 있으면 스킵
        if os.path.exists(db_path):
            print(f"  └─ Index already exists → skip ({db_path})")
        else:
            cmd = [
                "python",
                "/home/mintaek/hybrid_index/methods/Milvus/debug/make_index_hardness_format.py",
                "--db_path", db_path,
                "--collection_name", f"sift1m_A{num_attribute}_{cardi}_{distribution}",
                "--payload_path", base_label_path,
                "--vector_path", base_vector_path,
            ]
            try:
                subprocess.run(cmd, check=True)
                print("  └─ Index build complete ✅")
            except subprocess.CalledProcessError as e:
                print(f"  ⚠️  Index build failed: {e}")

        ### **hardness 기준 batching 없이** 1만개 query를 각각 한번에 돌려서 qps, recall을 측정 후 아래 포멧으로 MILVUS_trade_off에 저장
        ### Milvus는 파라미터 없으니까 param1에 그냥 아무 값이나 넣어도 됨
        # ─────────────────────────────────────────────────────────────
        # [Query Execution]
        # ─────────────────────────────────────────────────────────────
        print("[Querying] Running search …")

        result_prefix = milvus_root + "/milvus_query_results/"
        cmd = [
            "python",
            "/home/mintaek/hybrid_index/methods/Milvus/debug/search_hardness_format.py",
            "--db_path", milvus_root+f"/milvus_index/{dataset_name}_A{num_attribute}_{cardi}_{distribution}.db",
            "--collection_name", f"{dataset_name}_A{num_attribute}_{cardi}_{distribution}",
            "--test_path", tests_file,
            "--result_prefix", result_prefix,
        ]        
        subprocess.run(cmd, check=True)
        print("  └─ Query execution complete ✅")      

        # ─────────────────────────────────────────────────────────────
        # [Results Parsing]
        # ─────────────────────────────────────────────────────────────   
        result_csv = os.path.join(result_prefix, "result.csv")
        df = pd.read_csv(result_csv)

        # 한 번만 돌렸으니까 0번째 행만 쓰면 됨
        recall = float(df.loc[0, "Recall"])
        qps = float(df.loc[0, "QPS"])

        # Milvus는 튜닝 파라미터가 없으니 이름을 고정으로 넣자
        param_name = "param1"

        if d_name not in MILVUS_trade_off:
            MILVUS_trade_off[d_name] = {}

        MILVUS_trade_off[d_name][param_name] = {
            "qps": qps,
            "recall": recall,
        }
        """ 
        MILVUS_trade_off = {"closer_to_post": {
                                    param1: {qps: xxxx, recall: xxxx}
                                    }
                            "closer_to_pre" : {
                                    param1: {qps: xxxx, recall: xxxx}
                                    }
                            }
        """
    
    # --- 결과 요약 ---
    post_score = pc.final_score(MILVUS_trade_off["closer_to_post"])
    pre_score = pc.final_score(MILVUS_trade_off["closer_to_pre"])

    print("\n[Final Scores]")
    print(f"  ├─ Post-base Score : {post_score:.6f}")
    print(f"  └─ Pre-base  Score : {pre_score:.6f}")

    if post_score > pre_score:
        print("[Decision] → ✅ post_base selected")
        return "post_base"
    else:
        print("[Decision] → ✅ pre_base selected")
        return "pre_base"
