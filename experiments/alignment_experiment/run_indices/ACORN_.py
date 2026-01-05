import json, os
import numpy as np
import struct
from pathlib import Path
from tqdm import tqdm

# 먼저 환경변수로 전역 런타임 제한
os.environ["OMP_NUM_THREADS"] = "8"        # OpenMP (FAISS 포함)
os.environ["OPENBLAS_NUM_THREADS"] = "1"    # OpenBLAS는 1 추천
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

import faiss
faiss.omp_set_num_threads(8)

#dataset_name ="glove1m"

def read_fvecs(fname):
    """fvec 파일을 읽어서 (N, d) float32 배열로 반환"""
    with open(fname, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.int32)
    dim = arr[0]
    return np.frombuffer(data, dtype=np.float32).reshape(-1, dim + 1)[:, 1:]

def write_fvecs(fname, arr):
    """(N, d) float32 배열을 fvec 포맷으로 저장"""
    # 상위 디렉터리 자동 생성
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    n, d = arr.shape
    with open(fname, 'wb') as f:
        for vec in arr:
            f.write(np.array([d], dtype=np.int32).tobytes())
            f.write(vec.astype(np.float32).tobytes())

def read_txt(fname):
    data = []
    with open(fname) as f:
        for line in f:
            vals = line.strip().split()
            if vals:
                data.append([int(v) for v in vals])
    return data  # numpy array로 변환 X, ragged list 유지

def write_txt(fname, arr):
    """
    리스트 of 리스트 (혹은 (N, d) ndarray) 를 txt로 저장.
    행마다 길이가 달라도 저장 가능.
    """
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    with open(fname, "w", encoding="utf-8") as f:
        for row in arr:
            # numpy array나 list 다 지원
            line = " ".join(map(str, row))
            f.write(line + "\n")

def find_recall(k, knn_path, gt_path, nv):
    counts = []

    sum_topk = 0
    with open(knn_path, "r") as f_knn, open(gt_path, "r") as f_gt:
        for i, (knn_line, gt_line) in enumerate(zip(f_knn, f_gt)):
            # 공백 라인 안전 처리
            if not knn_line.strip() or not gt_line.strip():
                counts.append(0)
                continue

            # KNN은 membership 조회가 많으니 set으로
            knn_ids = set(map(int, knn_line.strip().split()))
            # GT는 순서가 필요하니 list로 -> 앞 k개를 취함
            gt_ids  = list(map(int, gt_line.strip().split()))
            topk_gt = []
            
            for i in range(k):
                if len(gt_ids) <= i:
                    break

                if gt_ids[i] <= nv:
                    topk_gt.append(gt_ids[i])
                else:
                    break

            sum_topk += len(topk_gt)

            match_count = sum(1 for x in topk_gt if x in knn_ids)
            counts.append(match_count)

    if not counts:
        return 0.0

    avg_hits = sum(counts) / len(counts)
    avg_topk = sum_topk / len(counts)

    if avg_topk == 0:
        recall_at_k = 1
    else:
        recall_at_k = avg_hits / avg_topk  # 평균적으로 k개 중 몇 개가 맞았는지 (비율)
    
    return recall_at_k

CHUNK_BYTES = 1 << 20  # 1MB

class BitWriter:
    """
    LSB-first 비트스트림 라이터.
    비트 8개 모이면 1바이트로 누적, 큰 청크로 파일에 기록.
    """
    def __init__(self, f, chunk_bytes=CHUNK_BYTES):
        self.f = f
        self.chunk_bytes = chunk_bytes
        self.cur = 0
        self.bitpos = 0
        self.buf = bytearray()

    def write_bit(self, bit):
        if bit:
            self.cur |= (1 << self.bitpos)
        self.bitpos += 1
        if self.bitpos == 8:
            self.buf.append(self.cur)
            if len(self.buf) >= self.chunk_bytes:
                self.f.write(self.buf)
                self.buf.clear()
            self.cur = 0
            self.bitpos = 0

    def write_bits_from_list(self, bits_list):
        for b in bits_list:
            self.write_bit(1 if b else 0)

    def close(self):
        if self.bitpos > 0:
            self.buf.append(self.cur)
        if self.buf:
            self.f.write(self.buf)
        self.buf.clear()

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "methods", "ACORN"))

from ACORN_build import (
    build_and_run_acorn_from_python,
    run_acorn_search_from_python,
    build_bit_map_from_python,
)


from pathlib import Path


def _read_row_bits_lsbfirst(fp, start_bit: int, length_bits: int) -> list[int]:
    """
    파일 핸들 fp에서 start_bit부터 length_bits만큼의 비트를 LSB-first 규칙으로 읽어
    [0/1] 리스트로 반환한다. (랜덤 액세스)
    """
    if length_bits <= 0:
        return []

    start_byte = start_bit // 8
    offset_in_byte = start_bit % 8
    # 이 범위를 덮기 위해 필요한 바이트 수
    need_bytes = (offset_in_byte + length_bits + 7) // 8

    fp.seek(start_byte)
    data = fp.read(need_bytes)
    if len(data) < need_bytes:
        raise RuntimeError("파일이 예상보다 일찍 끝났습니다.")

    out = []
    for k in range(length_bits):
        bit_idx = offset_in_byte + k
        byte_i = bit_idx // 8
        bit_in_byte = bit_idx % 8
        b = data[byte_i]
        out.append((b >> bit_in_byte) & 1)
    return out

def slice_bitmap_rows_in_order(inp_path: str, nb: int, rows_in_order: list[int], out_path: str) -> dict:
    """
    nb가 8의 배수일 때: 행 단위(row_bytes) 그대로 복사해서 빠르게 출력.
    """
    if nb % 8 != 0:
        print("nb is not 8*n")
        size_bytes = Path(inp_path).stat().st_size
        total_bits = size_bytes * 8
        if total_bits % nb != 0:
            raise RuntimeError(f"전체 비트({total_bits})가 nb({nb})로 나누어떨어지지 않습니다.")
        nq = total_bits // nb

        # 범위 검사
        if rows_in_order:
            rmin, rmax = min(rows_in_order), max(rows_in_order)
            if rmin < 0 or rmax >= nq:
                raise ValueError(f"row 인덱스 범위 오류: [0, {nq-1}] 바깥 값 포함")

        with open(inp_path, "rb") as fi, open(out_path, "wb") as fo:
            bw = BitWriter(fo)
            for r in tqdm(rows_in_order):
                start_bit = r * nb
                row_bits = _read_row_bits_lsbfirst(fi, start_bit, nb)  # [0/1] 리스트
                bw.write_bits_from_list(row_bits)
            bw.close()

        out_bits = len(rows_in_order) * nb
        out_bytes = (out_bits + 7) // 8
        info = dict(
            nq=nq,
            in_cols=nb,
            kept_rows=len(rows_in_order),
            out_bits=out_bits,
            out_bytes=out_bytes,
            out_path=out_path,
        )
        return info

    print("nb is 8*n")
    size_bytes = Path(inp_path).stat().st_size
    total_bits = size_bytes * 8
    if total_bits % nb != 0:
        raise RuntimeError(f"전체 비트({total_bits})가 nb({nb})로 나누어떨어지지 않습니다.")
    nq = total_bits // nb

    row_bytes = nb // 8

    # 범위 검사
    if rows_in_order:
        rmin, rmax = min(rows_in_order), max(rows_in_order)
        if rmin < 0 or rmax >= nq:
            raise ValueError(f"row 인덱스 범위 오류: [0, {nq-1}] 바깥 값 포함")

    with open(inp_path, "rb") as fi, open(out_path, "wb") as fo:
        for r in tqdm(rows_in_order):
            fi.seek(r * row_bytes)             # 행 시작 바이트로 바로 이동
            row_data = fi.read(row_bytes)      # 행 전체 읽기
            if len(row_data) != row_bytes:
                raise RuntimeError(f"행 {r} 읽기 실패")
            fo.write(row_data)                 # 그대로 출력

    out_bits = len(rows_in_order) * nb
    out_bytes = (out_bits + 7) // 8
    return dict(
        nq=nq,
        in_cols=nb,
        kept_rows=len(rows_in_order),
        out_bits=out_bits,
        out_bytes=out_bytes,
        out_path=out_path,
    )


def save_recall_qps(filename, batch_data):
    """
    filename: 저장할 txt 경로
    batch_data: dict[int, list[tuple]]
        {
          batch_id: [(M, Gamma, M_beta, qps, recall), ...],
          ...
        }
    """
    # 1) 기존 파일 읽기 (| 기준)
    existing = {}  # key: (batch, M, Gamma, M_beta) -> val: (qps, recall)
    if os.path.exists(filename):
        with open(filename, "r") as f:
            lines = f.readlines()

        # 헤더 스킵
        for line in lines[1:]:
            s = line.strip()
            if not s:
                continue
            parts = [p.strip() for p in s.split("|")]
            if len(parts) < 6:
                continue  # 안전장치

            batch   = int(parts[0])
            M       = int(parts[1])
            Gamma   = int(parts[2])
            M_beta  = int(parts[3])
            qps     = float(parts[4])
            recall  = float(parts[5])

            existing[(batch, M, Gamma, M_beta)] = (qps, recall)

    # 2) 새 데이터 반영 (덮어쓰기/추가)
    for batch, entries in batch_data.items():
        for m, gamma, m_beta, qps, recall in entries:
            existing[(batch, m, gamma, m_beta)] = (qps, recall)

    if not existing:
        # 비어있을 때도 헤더만 만들어 놓자
        with open(filename, "w") as f:
            f.write(f"{'Batch':^5} | {'M':^2} | {'Gamma':^5} | {'M_beta':^6} | {'QPS':^12} | Recall\n")
        return

    # 3) 정렬: M → Gamma → M_Beta → Batch
    sorted_items = sorted(
        existing.items(),
        key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])
    )

    # 4) 저장 (열 정렬 + 배치 바뀔 때 '...' 출력)
    with open(filename, "w") as f:
        f.write(f"{'Batch':^5} | {'M':^2} | {'Gamma':^5} | {'M_beta':^6} | {'QPS':^12} | Recall\n")
        # 첫 줄 기준 배치
        prev_batch = sorted_items[0][0][0]
        for (batch, M, Gamma, M_beta), (qps, recall) in sorted_items:
            if batch != prev_batch:
                prev_batch = batch
            # 칼럼 폭은 예쁘게 맞춤 (원하는 폭으로 조절 가능)
            f.write(f"{batch:<5} | {M:<2} | {Gamma:<5} | {M_beta:<6} | {qps:<12.6f} | {recall:.6f}\n")
    
    print(f"Save txt file: {filename}")


# for num_attribute, card, base_distribution, corr, missing in zip (
#   [3,3,3],
#   [[12] * 3,[12] * 3,[12] * 3],
#   ["random", "zipf", "zipf"],
#   [[0.0]*3, [0.5]*3, [0.0]*3],
#   [[0.5]*3, [0.0]*3, [0.5]*3],
# ):
    # if dataset_name == "sift1m" or dataset_name == "gist1m" or dataset_name == "glove1m":
    #     # cardi = '_'.join(str(c) for c in cardinality)
    #     cardinality = '_'.join(str(c) for c in card)
    #     correlation = '_'.join(str(c) for c in corr)
    #     missing_prob = '_'.join(str(c) for c in missing)
    #     data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset_name}_A{num_attribute}_{cardinality}_{base_distribution}_{missing_prob}_{correlation}"
    # elif dataset_name == "HnM" or dataset_name == "mtg-40K": 
    #     data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset_name}"

    # elif dataset_name == "ArXiv":
    #     data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/ArXiv/medium/include"

for dataset_name in ["sift_high", "sift_low", "gist_high", "gist_low"]:
    data_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset_name}"
    in_hard_json = 1


    # -----------------------------
    # 1. 스키마(필터 조건) 로드
    # -----------------------------
    with open(f"{data_path}/filters.json", "r", encoding="utf-8") as f: #hardness_format
        schema = json.load(f)

    # 모든 카테고리 값을 하나의 리스트로 합침
    all_values = []
    for entry in schema:
        field = entry["name"]
        all_values.extend([f"{field}:{v}" for v in entry["values"]])


    value_to_index = {v: i for i, v in enumerate(all_values)}

    def encode_onehot(item):
        vec = np.zeros(len(all_values), dtype=np.int32)
        for entry in schema:
            field = entry["name"]
            val = item.get(field)
            key = f"{field}:{val}"
            if key in value_to_index:
                vec[value_to_index[key]] = 1
        return vec

    #db에 대한 query filter

    input_path = f"{data_path}/payloads.jsonl"  # 한 줄당 하나의 JSON 객체 #hardness_format
    output_path = f"{data_path}/mid_format/db_filter_ACORN.txt"


    encoded_vectors = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {line_no}번째 줄 JSON 파싱 실패: {e}")
                continue

            vec = encode_onehot(item)
            encoded_vectors.append(vec)

    encoded_vectors = np.array(encoded_vectors, dtype=np.int8)

    # -----------------------------
    # 4. 저장
    # -----------------------------
    np.savetxt(output_path, encoded_vectors, fmt="%d")

    nb = encoded_vectors.shape[0]

    print(f"[OK] 총 {encoded_vectors.shape[0]}개 인코딩 완료, shape={encoded_vectors.shape}")
    print(f"저장 위치: {output_path}")

    # 원본 .npy 파일 경로
    npy_path = f"{data_path}/mid_format/base_vectors.npy" #hardness_format
    # 변환된 .fvec 파일 저장 경로
    fvec_path = f"{data_path}/mid_format/db_vectors_ACORN.fvec"

    # .npy 파일 불러오기
    data = np.load(npy_path)
    print("원본 shape:", data.shape, "dtype:", data.dtype)

    # float32로 변환 (fvec은 float32 사용)
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # fvec 파일로 저장
    with open(fvec_path, "wb") as f:
        for vec in data:
            # 차원 저장 (int32)
            f.write(struct.pack('i', data.shape[1]))
            # 벡터 값 저장 (float32)
            f.write(vec.tobytes())    

    print("변환 완료:", fvec_path)


    with open(f"{data_path}/filters.json", "r", encoding="utf-8") as f: #hardness_format
        schema = json.load(f)

    # 모든 카테고리 값을 하나의 리스트로 합침
    all_values = []
    for entry in schema:
        field = entry["name"]
        all_values.extend([f"{field}:{v}" for v in entry["values"]])


    value_to_index = {v: i for i, v in enumerate(all_values)}

    def encode_onehot(item_list):
        """
        item_list: [{'field_name': {'match': {'value': '...'}}}, ...] 형태의 리스트
        """
        vec = np.zeros(len(all_values), dtype=np.int32)

        if not isinstance(item_list, list):
            return vec  # list가 아니면 그냥 0벡터 반환

        for cond in item_list:
            if not isinstance(cond, dict) or not cond:
                continue

            # 첫 번째 (key, value) 꺼내기
            field, field_obj = next(iter(cond.items()))

            # value 추출
            val = None
            if isinstance(field_obj, dict) and "match" in field_obj:
                match_obj = field_obj["match"]
                if isinstance(match_obj, dict) and "value" in match_obj:
                    val = match_obj["value"]
            elif isinstance(field_obj, (str, int, float)):
                val = field_obj

            # 매핑 후 원-핫 벡터에 반영
            if val is not None:
                key = f"{field}:{val}"
                if key in value_to_index:
                    vec[value_to_index[key]] = 1

        return vec


    input_path = f"{data_path}/tests.jsonl"  # 한 줄당 하나의 JSON 객체 #hardness_format
    output_path = f"{data_path}/mid_format/query_filter_ACORN.txt"

    encoded_vectors = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)  # 문자열 -> dict
            except json.JSONDecodeError as e:
                print(f"[WARN] {line_no}번째 줄 JSON 파싱 실패: {e}")
                continue

            and_part = item.get("conditions", {}).get("and", {})

            vec = encode_onehot(and_part)
            encoded_vectors.append(vec)

    encoded_vectors = np.array(encoded_vectors, dtype=np.int8)

    np.savetxt(output_path, encoded_vectors, fmt="%d")

    print(f"[OK] 총 {encoded_vectors.shape[0]}개 인코딩 완료, shape={encoded_vectors.shape}")
    print(f"저장 위치: {output_path}")


    input_path = f"{data_path}/tests.jsonl"  # 한 줄당 하나의 JSON 객체 #hardness_format
    output_path = f"{data_path}/mid_format/query_vectors_ACORN.fvec"

    dim_ref = None
    n_ok, n_bad = 0, 0

    with open(input_path, "r", encoding="utf-8") as fr, open(output_path, "wb") as fw:
        for line_no, line in enumerate(fr, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {line_no} JSON 파싱 실패: {e}")
                n_bad += 1
                continue

            query_part = item.get("query")
            if query_part is None:
                print(f"[WARN] {line_no} query 없음 — 건너뜀")
                n_bad += 1
                continue

            # list -> float32 1D array
            try:
                vec = np.asarray(query_part, dtype=np.float32)
                if vec.ndim != 1:
                    raise ValueError(f"ndim={vec.ndim}")
            except Exception as e:
                print(f"[WARN] {line_no} 벡터 변환 실패: {e}")
                n_bad += 1
                continue

            # 차원 고정 체크
            if dim_ref is None:
                dim_ref = int(vec.shape[0])
            elif vec.shape[0] != dim_ref:
                print(f"[WARN] {line_no} 차원 불일치: {vec.shape[0]} (기대 {dim_ref}) — 건너뜀")
                n_bad += 1
                continue

            # fvec: int32(dim) + float32[dim]
            fw.write(struct.pack("i", dim_ref))
            fw.write(vec.tobytes())
            n_ok += 1

    print(f"[OK] 쓴 벡터: {n_ok}, 건너뜀: {n_bad}, dim={dim_ref}, 파일={output_path}")

    output_path = f"{data_path}/mid_format/query_gt_ACORN.txt"

    gt = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)  # 문자열 -> dict
            except json.JSONDecodeError as e:
                print(f"[WARN] {line_no}번째 줄 JSON 파싱 실패: {e}")
                continue

            gt_line = item.get("closest_ids", [])

            if not isinstance(gt_line, list):
                print(f"[WARN] {line_no}번째 줄: closest_ids가 list가 아님")
                continue

            gt.append(gt_line)  # 한 줄씩 추가


    gt_obj = np.array(gt, dtype=object)

    with open(output_path, "w") as f:
        for row in gt_obj:           # row가 [12, 37, 41] 같은 리스트라고 가정
            f.write(" ".join(map(str, row)) + "\n")

    print(f"[OK] saved to {output_path}")


    def normalize(arr):
        if np.max(arr) == np.min(arr):
            return np.zeros_like(arr)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))





    fname = f"{data_path}/mid_format/db_vectors_ACORN.fvec"
    assert os.path.exists(fname), f"입력 파일 없음: {fname}"

    # fvec 헤더에서 dimension 읽기
    with open(fname, "rb") as f:
        d_from_file = np.fromfile(f, dtype=np.int32, count=1)[0]
    print("d_from_file =", d_from_file)

    # (선택) 파일 정렬/레코드 수도 확인
    size = os.path.getsize(fname)
    rec_bytes = (d_from_file + 1) * 4
    print("aligned =", size % rec_bytes == 0, "  n =", size // rec_bytes)

    faiss_include_dirs = [
        "/home/ec2-user/hybrid_hardness/methods/ACORN",  # 이 경로 아래에 faiss/IndexACORN.h 가 있어야 함
    ]
    faiss_lib_dirs = [
        "/home/ec2-user/hybrid_hardness/methods/ACORN/build/faiss",
    ]

    
    idx_path = f"{data_path}/ACORN_format/ACORN_index_32_12_64.faiss"
    if not Path(idx_path).is_file():
        build_and_run_acorn_from_python(
            d=int(d_from_file), M=32, gamma=12, M_beta=64,
            filename=fname,
            out_index_path=idx_path,
            include_dirs=faiss_include_dirs,
            lib_dirs=faiss_lib_dirs
            #libs=["faiss"]  # GPU 빌드면 CUDA 관련 라이브러리들 추가
        )

    bit_map_path = f"{data_path}/ACORN_format/bit_map.txt"
    if os.path.exists(bit_map_path):
        print(f"skip build total bit map (already exists)")

    else:
        build_bit_map_from_python(
            d=int(d_from_file),
            faiss_index_path=f"{data_path}/ACORN_format/ACORN_index_32_12_64.faiss",
            db_vectors_path=f"{data_path}/mid_format/db_vectors_ACORN.fvec",
            query_vectors_path=f"{data_path}/mid_format/query_vectors_ACORN.fvec",
            db_filters_path=f"{data_path}/mid_format/db_filter_ACORN.txt",
            query_filters_path=f"{data_path}/mid_format/query_filter_ACORN.txt",
            out_I_path=f"",
            bit_map=bit_map_path,
            include_dirs=faiss_include_dirs,
            lib_dirs=faiss_lib_dirs
        )

    # for i in range(10):
    #     rows_in_order = sorted_idx[i*int(nq/10) : (i+1)*int(nq/10)]

    #     out_path=f"{data_path}/ACORN_format/batch{i}{hardness_type}/bit_map.txt"

    #     if os.path.exists(out_path):
    #         print("skip build bit map")
    #     else:
    #         info = slice_bitmap_rows_in_order(
    #             inp_path=f"{data_path}/ACORN_format/bit_map.txt",
    #             nb=nb,
    #             rows_in_order=rows_in_order.tolist(),
    #             out_path=out_path,
    #         )
    
    k = 10

    params_list =[
        (80, 30, 160),
        (96, 36, 192),
        (128, 48, 256)
    ]

    batches = [[] for _ in range(10)]  # batch_0 ~ batch_9



    # 만약 데이터가 바뀌었다면 ACORN_format/ACORN_index_{M}_{gamma}_{M_beta}.faiss 파일 지우고 다시 실행할 것!

    l = 1
    for M, gamma, M_beta in params_list:
        idx_path = f"{data_path}/ACORN_format/ACORN_index_{M}_{gamma}_{M_beta}.faiss"
        if not Path(idx_path).is_file():
            build_and_run_acorn_from_python(
                d=int(d_from_file), M=M, gamma=gamma, M_beta=M_beta,
                filename=fname,
                out_index_path=idx_path,
                include_dirs=faiss_include_dirs,
                lib_dirs=faiss_lib_dirs
                #libs=["faiss"]  # GPU 빌드면 CUDA 관련 라이브러리들 추가
            )
            print(f"\n\n\nfinish build {M}_{gamma}_{M_beta} parameters\n\n\n")

        l += 1


    # npy_path = f"{data_path}/vectors.npy" #hardness_format

    # data = np.load(npy_path)
    # nv = data.shape[0]

    # for M, gamma, M_beta in params_list:
    #     for i in range(10):
    #         idx_path = f"{data_path}/ACORN_format/ACORN_index_{M}_{gamma}_{M_beta}.faiss"
    #         print(f"batch: {i}")
    #         qps_batch = run_acorn_search_from_python(
    #             d=int(d_from_file), M=M, gamma=gamma, M_beta=M_beta, k=k,
    #             faiss_index_path=idx_path,
    #             db_vectors_path=f"{data_path}/mid_format/db_vectors_ACORN.fvec",
    #             query_vectors_path=f"{data_path}/ACORN_format/batch{i}{hardness_type}/query_vectors_ACORN.fvec",
    #             db_filters_path=f"{data_path}/mid_format/db_filter_ACORN.txt",
    #             query_filters_path=f"{data_path}/ACORN_format/batch{i}{hardness_type}/query_filters_ACORN.txt",
    #             out_I_path=f"{data_path}/ACORN_format/batch{i}{hardness_type}/knn_I.txt",
    #             bit_map=f"{data_path}/ACORN_format/batch{i}{hardness_type}/bit_map.txt",
    #             include_dirs=faiss_include_dirs,
    #             lib_dirs=faiss_lib_dirs
    #             #libs=["faiss"],   # GPU 빌드라면 CUDA 관련 libs 추가
    #         )
    #         recall = find_recall(k, f"{data_path}/ACORN_format/batch{i}{hardness_type}/knn_I.txt", f"{data_path}/ACORN_format/batch{i}{hardness_type}/query_gt_ACORN.txt", nv)
    #         batches[i].append((recall, qps_batch))

    # batch_data = {}

    # filename_recall = f"{data_path}/ACORN_format/{sort_hardness}_search_results.txt"

    # for i in range(10):
    #     batch_data[i] = []
    #     for j, (m, gamma, m_beta) in enumerate(params_list, start=0):
    #         batch_data[i].append((m, gamma, m_beta, batches[i][j][1], batches[i][j][0]))

    # print(f"{batch_data}\n")


    # save_recall_qps(filename_recall, batch_data)

    # import matplotlib.pyplot as plt

    # num_batches = 10
    # num_params = len(params_list)

    # plt.figure(figsize=(8, 6))

    # for i in range(num_batches):
    #     # 이 batch에서 param들을 따라가며 (qps, recall) 수집
    #     qps_list = []
    #     recall_list = []
    #     for j in range(num_params):
    #         recall_ij, qps_ij = batches[i][j]  # (recall, qps)
    #         qps_list.append(qps_ij)
    #         recall_list.append(recall_ij)

    #     # 보기 좋게 QPS 기준으로 정렬 (선이 덜 꼬이게)
    #     pairs = sorted(zip(qps_list, recall_list), key=lambda x: x[0])
    #     qps_sorted = [p[0] for p in pairs]
    #     recall_sorted = [p[1] for p in pairs]

    #     plt.plot(qps_sorted, recall_sorted, marker='o', label=f"Batch {i}")

    # plt.xlabel("QPS (Queries per second)")
    # plt.ylabel(f"Recall@{k}")
    # plt.title("Recall-QPS Trade-off per Batch")
    # plt.grid(True)
    # plt.legend(title="Batch")
    # plt.tight_layout()

    # out_png = f"{data_path}/ACORN_format/{sort_hardness}_recall_qps_per_batch.png"
    # plt.savefig(out_png, dpi=300)
    # plt.close()

    # print(f"Saved per-batch Recall–QPS plot to {out_png}")



    # import shutil

    # for i in range(10):

    #     if dataset_name == "HnM":
    #         file_path = [f"{data_path}/ACORN_format/batch{i}{hardness_type}/knn_I.txt",
    #                     f"{data_path}/ACORN_format/batch{i}{hardness_type}/query_filters_ACORN.txt",
    #                     f"{data_path}/ACORN_format/batch{i}{hardness_type}/query_gt_ACORN.txt",
    #                     f"{data_path}/ACORN_format/batch{i}{hardness_type}/query_vectors_ACORN.fvec"]

    #         for file_ in file_path:
    #             if os.path.exists(file_):
    #                 os.remove(file_)
    #                 print(f"✅ 삭제 완료: {file_}")
    #             else:
    #                 print(f"⚠️ 파일이 존재하지 않습니다: {file_}")
    #     else:
    #         file_ = f"{data_path}/ACORN_format/batch{i}{hardness_type}"
    #         if os.path.exists(file_):
    #             shutil.rmtree(file_)
    #             print(f"✅ 삭제 완료: {file_}")
    #         else:
    #             print(f"⚠️ 파일이 존재하지 않습니다: {file_}")

# 그래프 만드는 거 + 저장은 아직 안함.