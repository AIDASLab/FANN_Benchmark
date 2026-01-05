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


# dataset_list = ["sift_high","sift_low","gist_high","gist_low","sift1m_ACORN", "sift1m_NHQ","sift1m_UNG","sift1m_RWalks"]
# # dataset_list = ["sift1m_UNG_modi"]
# dataset_name_list = ["sift1m", "sift1m","gist1m","gist1m", "sift1m", "sift1m", "sift1m", "sift1m"]
# # dataset_name_list = ["sift1m"]


dataset_list = ["arxiv", "LAION1M", "tripclick", "yfcc"]
# dataset_list = ["yfcc"]
for dataset in dataset_list:
    print()
    original_data_path = f"/home/ec2-user/hybrid_hardness/semi-real/filterbenchmark/{dataset}"
    
    data_path = original_data_path
    hardness_path = os.path.join(data_path, "hardness_format")
    query_fvecs_path = os.path.join(original_data_path, f"{dataset}_query_equal.fvecs")
    base_vector_path = os.path.join(data_path, f"{dataset}_base.bin")
    base_label_path = os.path.join(data_path, "label_base.txt")

    ung_root = os.path.join(data_path, "ung_format")
    ################################## index build
    binary = "/home/ec2-user/hybrid_hardness/methods/Unified-Navigating-Graph/build/apps/build_UNG_index"
    
    cmd = [
        binary,
        "--data_type", "float",
        "--dist_fn", "L2",
        "--num_threads", "20",
        "--max_degree", "32",
        "--Lbuild", "100",
        "--alpha", "1.2",
        "--base_bin_file", base_vector_path,
        "--base_label_file", base_label_path,
        "--index_path_prefix", ung_root+"/ung_index/",
        "--scenario", "general",
        "--num_cross_edges", "6"
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[에러] 명령어 실행 실패: {e}")

