
import numpy as np
"""
def read_fvecs(fname):
    with open(fname, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.int32)
    dim = arr[0]
    return np.frombuffer(data, dtype=np.float32).reshape(-1, dim + 1)[:, 1:]

vecs = read_fvecs("../../../hybrid_benchmark_outdated/hnm/converted_hnm/hnm_base.fvec")
print(vecs.shape)
print(vecs[0])
"""


import os

def count_lines(path: str) -> int:
    # 큰 파일도 빠르게 줄 수만 세는 유틸
    with open(path, "rb") as f:
        return sum(1 for _ in f)

def read_first_row_bitmap(path_binary: str, nb: int) -> np.ndarray:
    """
    bit_map 바이너리에서 '첫 행'을 0/1 ndarray(uint8)로 반환.
    저장 방식(바이트당 0/1 vs 비트패킹)을 자동으로 추정해서 처리.
    """
    # 한 행을 위해 필요한 바이트 수(비트패킹 가정)
    packed_bytes = (nb + 7) // 8

    with open(path_binary, "rb") as f:
        # 우선 nb 바이트 + packed_bytes 중 큰 쪽만큼 슬쩍 읽어서 판별
        peek = f.read(max(nb, packed_bytes))

        if len(peek) >= nb and set(peek[:nb]) <= {0, 1}:
            # 앞 nb 바이트가 모두 0/1이면: 바이트당 0/1 저장 형식
            row_bytes = peek[:nb]
            row = np.frombuffer(row_bytes, dtype=np.uint8)
            return row

        # 아니면 비트패킹으로 간주 → 처음 packed_bytes만 다시 사용
        raw = np.frombuffer(peek[:packed_bytes], dtype=np.uint8)
        bits = np.unpackbits(raw)[:nb]
        return bits.astype(np.uint8)

# ===== 사용 예시 =====
bit_map_path = "/home/mintaek/hybrid_index/Benchmark/sift1m_A3_6_6_6_random/ACORN_format/batch0/bit_map.txt"
db_filters_path = "/home/mintaek/hybrid_index/Benchmark/sift1m_A3_6_6_6_random/mid_format/db_filter_ACORN.txt"

# nb 추정(= DB 벡터 개수): db_filters.txt의 줄 수
nb = count_lines(db_filters_path)

first_row = read_first_row_bitmap(bit_map_path, nb)
print("nb =", nb, " / first_row.shape =", first_row.shape)
print("first_row (첫 64개) =", first_row[:64])


"""
with open("/home/mintaek/hybrid_index/Benchmark/sift1m_A1_12/ACORN_format/batch9/query_gt_ACORN.txt", "r") as f:
    for line in f:
        if line.strip():  # 빈 줄 스킵
            vecs1.append(list(map(int, line.strip().split())))

vecs1 = np.array(vecs1)  # numpy array로 변환
print(vecs1.shape)

vec_ = vecs1[0]
print(vec_)
"""
"""
vecs2 = []
with open("/home/mintaek/hybrid_index/hybrid_benchmark_outdated/hnm/ACORN_hnm/query_gt.txt", "r") as f:
    for line in f:
        if line.strip():  # 빈 줄 스킵
            vecs2.append(list(map(int, line.strip().split())))

print(vecs2[9900])  # 9900번째 줄 출력
"""

"""
def read_fvecs(fname):
    with open(fname, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.int32)
    dim = arr[0]
    return np.frombuffer(data, dtype=np.float32).reshape(-1, dim + 1)[:, 1:]

vecs_1 = read_fvecs("/home/mintaek/hybrid_index/Benchmark/sift1m_A1_12/mid_format/db_vectors_ACORN.fvec")
print(vecs_1.shape)
print(vecs_1[0][0])
"""
"""
import numpy as np

path = "/home/mintaek/hybrid_index/Benchmark/sift1m_A1_12/hardness_format/vectors.npy"
arr = np.load(path, mmap_mode='r')   # 메모리맵: 필요한 부분만 읽음

print("shape:", arr.shape, "dtype:", arr.dtype)

# 첫 '줄' 출력 (2D 이상이면 첫 행, 1D면 첫 원소)
first_line = arr[0] if arr.ndim >= 2 else arr[:1]
print(first_line[0])
"""