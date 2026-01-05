import numpy as np
import networkx as nx
from collections import Counter
from networkx.algorithms.approximation import steiner_tree
import copy
from tqdm import tqdm
import json
import random
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import umap
import time
import heapq
from sklearn.preprocessing import normalize
import hnswlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import os
import argparse
import time

class BaselineHybridHardnessEstimator:
    def __init__(self, base_vectors, payloads, distance_metric='l2', num_thread=200):
        self.base_vectors = base_vectors
        self.vector_dim = len(base_vectors[0])
        self.distance_metric = distance_metric
        self.num_thread = num_thread
        self.payloads = payloads
        # self.G_base = self._build_graph(base_vectors, base=True)
        
        # cache
        self.query_vector = None
        self.filtered_ids = None
        self.filtered_vectors = None
        self.inverted_index = self._build_inverted_index(payloads)

    # payloads에 idx를 할당하여 찾기 편하게 만듦 
    def _build_inverted_index(self, payloads):
        """
        Build inverted index from payloads.
        Returns: dict[label][value] -> set of indices
        """
        inverted = defaultdict(lambda: defaultdict(set))
        for i, payload in enumerate(payloads):
            for key, val in payload.items():
                inverted[key][val].add(i)
        return inverted

    # query filter를 거친 base 노드들의 filtered_ids를 찾음 
    def _filter_ids_by_condition(self, test):
        self.query_vector = np.array(test["query"])
        conditions = test["conditions"]

        if "and" not in conditions or not conditions["and"]:
            # 조건 없음 → 전체
            self.filtered_ids = list(range(len(self.payloads)))
            return self.filtered_ids

        candidate_sets = []
        for cond in conditions["and"]:
            if not isinstance(cond, dict):
                continue
            for key, rule in cond.items():
                if "match" in rule and "value" in rule["match"]:
                    val = rule["match"]["value"]
                    if key in self.inverted_index and val in self.inverted_index[key]:
                        candidate_sets.append(self.inverted_index[key][val])
                    else:
                        # 조건을 만족하는 데이터가 없음 → empty set
                        self.filtered_ids = []
                        return []
                else:
                    continue  # 미지원 조건

        if not candidate_sets:
            self.filtered_ids = list(range(len(self.payloads)))
        else:
            self.filtered_ids = list(set.intersection(*candidate_sets))

        return self.filtered_ids

    def _distance(self, a, b): 
        if self.distance_metric == 'cosine': 
            # cosine distance = 1 - cosine similarity 
            a_norm = a / np.linalg.norm(a) 
            b_norm = b / np.linalg.norm(b) 
            return 1 - np.dot(a_norm, b_norm) 
        elif self.distance_metric in ('l2', 'euclidean'): 
            return np.linalg.norm(a - b) 
        else: raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _build_graph(self, vectors, oracle=None, base=None):
        p = hnswlib.Index(space=self.distance_metric, dim=self.vector_dim)
        p.init_index(max_elements=len(vectors), ef_construction=50, M=32)
        p.add_items(vectors, num_threads=self.num_thread)
        if base: 
            self.base_index = p
        if oracle :
            self.oracle_index = p
        adjacency_list = p.get_adjacency_list_level0()
        return adjacency_list

    def compute_H_corr(self):
        ids = np.asarray(self.filtered_ids)
        self.filtered_vectors = self.base_vectors[ids]
        # self.query_vector와 self.filtered_vectors 사이의 최소 거리 계산
        true_dist = np.min([
            self._distance(self.query_vector, v) for v in self.filtered_vectors
        ])
        # self.base_vectors에서 랜덤으로 self.filtered_ids의 개수만큼 노드 선택
        rand_ids = np.random.choice(
            len(self.base_vectors), size=len(self.filtered_ids), replace=False
        )
        rand_vectors = self.base_vectors[rand_ids]
        # 선택된 노드와 self.query_vector사이의 거리의 평균 계산
        num_trials = 2
        rand_dist = np.mean([
            np.min([self._distance(self.query_vector, v) for v in rand_vectors])
            for _ in range(num_trials)  # 여러 번 랜덤 집합 뽑기
        ])

        H_corr = rand_dist - true_dist
        if(H_corr == 0) :
            print ("true_dist: ",true_dist)
            print ("len(filtered_ids): ",len(self.filtered_ids))
            print ("rand_dist: ",rand_dist)
        
        return H_corr, rand_dist, true_dist

    def compute_H_sel(self):
        total_nodes = len(self.base_vectors)
        filtered_nodes = len(self.filtered_ids)
        
        self.selectivity = filtered_nodes / total_nodes
        H_selectivity = self.selectivity
        return H_selectivity


    def compute_baseline_hardness(self, test):
        self._filter_ids_by_condition(test)
        if len(self.filtered_ids) == 0:
            return {
                "selectivity": 0,
                "correlation": 0,
                "rand_dist": 0,
                "true_dist": 0,
                "no data satisfy filter condition": 1,
            } 

        H_sel = self.compute_H_sel()
        H_corr, rand_dist, true_dist = self.compute_H_corr()


        return {
            "selectivity": H_sel,
            "correlation": H_corr,
            "rand_dist": rand_dist,
            "true_dist": true_dist,
        }


    def compute_final_hardness (self, test):
        baseline_hardness_dict = self.compute_baseline_hardness(test)

        selectivity = baseline_hardness_dict["selectivity"] # 선택된 비율 (크면 어려움)
        correlation = baseline_hardness_dict["correlation"] # 랜덤거리 - 실제거리 (크면 쉬움)
        rand_dist = baseline_hardness_dict["rand_dist"] # rand_dist
        true_dist = baseline_hardness_dict["true_dist"] # true_dist
        if(correlation > 0):
            # 랜덤거리 > 실제거리 (쉬움)
            related_corr = 10*(correlation / rand_dist) + 1 # 1~10 normalize (크면 쉬움)
            combine =  selectivity / related_corr # 크면 어려움
        elif(correlation < 0):
            # 랜덤거리 < 실제거리 (어려움)
            related_corr = 10*(-1 * correlation / true_dist) + 1 # 1~10 normalize (크면 어려움)
            combine = selectivity * related_corr # 크면 어려움
        else:
            # correlation = 0 이라 related_corr = 1로 설정하여 영향 무시
            # filter 통과 없어도 여기로 옴
            combine =  selectivity

        total_hardness = baseline_hardness_dict
        total_hardness["select_corr_combine"] = combine

        return total_hardness



def main(data_dir, save_dir):
    # 파일 경로
    vectors_file = os.path.join(data_dir, "vectors.npy")
    payloads_file = os.path.join(data_dir, "payloads.jsonl")
    tests_file = os.path.join(data_dir, "tests.jsonl")

    # 1. Load vectors.npy
    vectors = np.load(vectors_file)
    print("vectors.shape =", vectors.shape)

    # 2. Load payloads.jsonl
    payloads = []
    with open(payloads_file, "r") as f:
        for line in f:
            payloads.append(json.loads(line))
    print(f"Loaded {len(payloads)} payloads")

    # 3. Load tests.jsonl
    tests = []
    with open(tests_file, "r") as f:
        for line in f:
            tests.append(json.loads(line))
    print(f"Loaded {len(tests)} tests")
    print("\nTest Numbers", len(tests))

    estimator = BaselineHybridHardnessEstimator(vectors, payloads, distance_metric="l2")

    os.makedirs(save_dir, exist_ok=True)

    result = []
    for i, test in enumerate(tqdm(tests)):
        result.append(estimator.compute_final_hardness(test))
        # 1000개마다 중간 저장
        if (i + 1) % 5000 == 0 or i == 0 or i == 9 or i == 199 or i == 999:
            fname = f'hardness_baseline_{i+1}.json'
            with open(os.path.join(save_dir, fname), 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved {fname} at {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Hardness Computation")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Directory containing vectors.npy, payloads.jsonl, tests.jsonl")
    parser.add_argument('--save_dir', type=str, required=True,
                        help="Directory to save partial result files")
    args = parser.parse_args()
    main(args.data_dir, args.save_dir)
