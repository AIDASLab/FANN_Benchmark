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

class HybridHardnessEstimator:
    def __init__(self, base_vectors, payloads, distance_metric='cosine', num_thread=4):
        self.base_vectors = base_vectors
        self.vector_dim = len(base_vectors[0])
        self.distance_metric = distance_metric
        self.num_thread = num_thread
        self.payloads = payloads
        self.base_index = None
        self.G_base = self._build_graph(base_vectors, base=True)
        self.total_num = len(self.base_vectors)
        

        # 캐시
        self.query_vector = None
        self.filtered_ids = None
        self.filtered_vectors = None
        self.G_filter = None
        self.G_oracle = None
        self.oracle_index = None
        self.filter_components = None
        self.oracle_components = None
        self.G_oracle_form1 = None
        self.key_error = 0
        self.alpha = 0
        self.inverted_index = self._build_inverted_index(payloads)

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

        ids = np.asarray(self.filtered_ids)
        self.filtered_vectors = self.base_vectors[ids]
        return self.filtered_ids

    def _build_graph(self, vectors, oracle=None, base=None):
        p = hnswlib.Index(space=self.distance_metric, dim=self.vector_dim)
        p.init_index(max_elements=len(vectors), ef_construction=50, M=32)
        p.add_items(vectors, num_threads=self.num_thread)
        if base: 
            self.base_index = p
        if oracle :
            self.oracle_index = p
        # adjacency_list = p.get_adjacency_list_level0()
        return 1



    def compute_H_cover(self):
        total_nodes = self.total_num
        filtered_nodes = len(self.filtered_ids)
        self.real_selectivity = filtered_nodes / total_nodes
        self.selectivity = np.log10(total_nodes / filtered_nodes) + 1 if filtered_nodes > 0 else 1
        return self.selectivity

    def _compute_components(self, G):
        if len(G) == 1:
            # print("G has only one node")
            return [[1]]
        visited = set()
        comps = []
        for node in G:
            if node not in visited:
                stack = [node]
                comp = []
                while stack:
                    n = stack.pop()
                    if n in visited:
                        continue
                    visited.add(n)
                    comp.append(n)
                    for nei in G.get(n, []):
                        if nei not in visited:
                            stack.append(nei)
                comps.append(comp)
        comps.sort(key=len, reverse=True)
        return comps


    def _distance(self, a, b):
        if self.distance_metric == 'cosine':
            # cosine distance = 1 - cosine similarity
            a_norm = a / np.linalg.norm(a)
            b_norm = b / np.linalg.norm(b)
            return 1 - np.dot(a_norm, b_norm)
        elif self.distance_metric in ('l2', 'euclidean'):
            return np.linalg.norm(a - b)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")


    def compute_H_scan (self):
        # ids = np.asarray(self.filtered_ids)
        # self.filtered_vectors = self.base_vectors[ids]
        # self.query_vector와 self.filtered_vectors 사이의 최소 거리 계산
        # print("filterd node number: ",len(self.filtered_ids))
        if len(self.filtered_ids) == 0:
            self.alpha = 0
            return 0
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

        self.density = rand_dist / (true_dist + 1)
        self.alpha = self.selectivity * self.density * 10
        if not np.isscalar(self.alpha):
            self.alpha = 10  # 기본값 지정 (필요 시 변경)
        return self.alpha






    def compute_H_fetch (self):
        query_vector = self.query_vector.reshape(1,-1)
        s_time = time.time()
        labels, distances = self.base_index.knn_query(query_vector, k=int(self.alpha))
        e_time = time.time()

        latency = (e_time - s_time) * 1000
        return (1.0 + latency)

    def compute_post_hardness (self, test):
        H_scan = self.compute_H_scan()
        H_fetch = self.compute_H_fetch()
        Post_hardness = H_scan * H_fetch / 100 ### scaleing 을 위한 나누기
        return {
            "selectivity": self.real_selectivity,
            "H_scan": H_scan,
            "H_fetch": H_fetch,
            "Post_Hardness": Post_hardness,
        }


    def compute_total_hardness (self, test):
        # pre_hardness_dict = self.compute_pre_hardness(test)
        self._filter_ids_by_condition(test)
        self.compute_H_cover()
        post_hardness_dict = self.compute_post_hardness(test)

        return post_hardness_dict



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

    estimator = HybridHardnessEstimator(vectors, payloads, distance_metric="l2")

    os.makedirs(save_dir, exist_ok=True)

    result = []
    for i, test in enumerate(tqdm(tests)):
        result.append(estimator.compute_total_hardness(test))
        # 1000개마다 중간 저장
        if (i + 1) % 5000 == 0 or i == 0 or i == 9 or i == 19 or i == 999:
            fname = f'hardness_v4.2_{i+1}.json'
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