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
dataset_list = ["sift_high","sift_low","gist_high","gist_low","sift1m_ACORN", "sift1m_NHQ","sift1m_UNG","sift1m_RWalks"]
# dataset_list = ["sift1m_UNG_modi"]
dataset_name_list = ["sift1m", "sift1m","gist1m","gist1m", "sift1m", "sift1m", "sift1m", "sift1m"]
# dataset_name_list = ["sift1m"]

# for num_attribute, card, base_distribution, corr, missing in zip (
#   [3,3,3],
#   [[12] * 3,[12] * 3,[12] * 3],
#   ["random", "zipf", "zipf"],
#   [[0.0]*3, [0.5]*3, [0.0]*3],
#   [[0.5]*3, [0.0]*3, [0.5]*3],
# ):
for dataset , dataset_name in zip(dataset_list, dataset_name_list):
    original_data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset_name}_original"
    if dataset_name == "sift1m" or dataset_name == "glove1m" or dataset_name == "gist1m":
        data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset}"
        hardness_path = os.path.join(data_path, "hardness_format")
        query_fvecs_path = os.path.join(original_data_path, f"{dataset_name}_query.fvecs")
        base_vector_path = os.path.join(data_path, "mid_format/base_vector.bin")
        if dataset == "sift_high" or dataset == "sift_low" or dataset == "gist_high" or dataset == "gist_low":
            base_label_path = os.path.join(data_path, "mid_format/base_label.txt")
        else: 
            base_label_path = os.path.join(data_path, "mid_format/base_label_UNG.txt")
        # hardness_json_path = os.path.join(data_path, "hardness/hardness_v3.0_10000.json")

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

