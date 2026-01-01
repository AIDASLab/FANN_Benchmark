from Index_analyzer import pareto_comp as pc
import os, subprocess, tempfile, shutil
import json
import numpy as np
import struct
from pathlib import Path

##################################################################################################
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

# --- Bit I/O 유틸 (LSB-first) ---

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


CPP_SOURCE_BUILD = r"""
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>
// ---- Thread control shim (put near the top) ----
#if __has_include(<faiss/utils/omp_utils.h>)
#include <faiss/utils/omp_utils.h>
static inline void set_threads_from_env() {
    if (const char* s = std::getenv("OMP_NUM_THREADS"))
        faiss::omp_set_num_threads(std::atoi(s));
}
#elif __has_include(<omp.h>)
#include <omp.h>
static inline void set_threads_from_env() {
    if (const char* s = std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::atoi(s));
}
#else
static inline void set_threads_from_env() { /* no-op */ }
#endif


float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = std::fopen(fname, "rb");
    if (!f) throw std::runtime_error(std::string("cannot open fvecs: ") + fname);

    int32_t d32 = 0;
    if (std::fread(&d32, sizeof(int32_t), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("failed to read dimension");
    }
    if (d32 <= 0) {
        std::fclose(f);
        throw std::runtime_error("invalid dimension in fvecs");
    }
    const size_t d = (size_t)d32;

    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        throw std::runtime_error("fseek failed");
    }
    long long fsize = std::ftell(f);
    if (fsize < 0) {
        std::fclose(f);
        throw std::runtime_error("ftell failed");
    }
    std::rewind(f);

    const size_t rec_bytes = (d + 1) * sizeof(int32_t);
    if ((unsigned long long)fsize % rec_bytes != 0ULL) {
        std::fclose(f);
        throw std::runtime_error("file size is not aligned to (d+1)*4 bytes — not an fvecs?");
    }
    const size_t n = (size_t)((unsigned long long)fsize / rec_bytes);

    float* xb = new float[n * d];

    for (size_t i = 0; i < n; ++i) {
        int32_t cur_d = 0;
        if (std::fread(&cur_d, sizeof(int32_t), 1, f) != 1) {
            delete[] xb; std::fclose(f);
            throw std::runtime_error("failed to read record header");
        }
        if (cur_d != d32) {
            delete[] xb; std::fclose(f);
            throw std::runtime_error("dimension mismatch inside fvecs file");
        }
        if (std::fread(xb + i * d, sizeof(float), d, f) != d) {
            delete[] xb; std::fclose(f);
            throw std::runtime_error("truncated fvecs file");
        }
    }

    std::fclose(f);
    *d_out = d;
    *n_out = n;
    return xb;
}

int main(int argc, char** argv){
    // argv: 1:d 2:M 3:gamma 4:M_beta 5:input.fvec 6:out.faiss
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                << " <d> <M> <gamma> <M_beta> <db_vectors.fvec> <out_index.faiss>\n";
        return 1;
    }
    int d     = std::stoi(argv[1]);
    int M     = std::stoi(argv[2]);
    int gamma = std::stoi(argv[3]);
    int M_beta= std::stoi(argv[4]);
    std::string filename = argv[5];
    std::string outpath  = argv[6];

    if (d <= 0) {
        std::cerr << "invalid d\n"; return 1;
    }

    try {
        size_t nb=0, d2=0;
        float* xb = fvecs_read(filename.c_str(), &d2, &nb);
        if ((size_t)d != d2) {
            delete[] xb;
            throw std::runtime_error("dataset dimension is not as expected");
        }

        // 메타데이터는 벡터 수에 맞춰 0으로 채움
        std::vector<int> metadata((size_t)nb, 0);

        // 인덱스 생성: Flat 저장 + ACORN 네비게이션 (L2)
        faiss::IndexACORNFlat acorn_gamma(d, M, gamma, metadata, M_beta);
        acorn_gamma.add(nb, xb);
        delete[] xb;

        faiss::write_index(&acorn_gamma, outpath.c_str());
        std::cout << "Index written to: " << outpath << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 2;
    }
}
"""

def build_and_run_acorn_from_python(
    d, M, gamma, M_beta,
    filename, out_index_path,
    include_dirs,
    lib_dirs,
    libs=None,
    cxx="g++",
    extra_cxxflags=None,
    extra_ldflags=None,
    keep_temp=False,
    num_threads: int = 200,
    blas_threads: int = 1,
    env_overrides: dict | None = None,
    timeout: float | None = None,
):
    """
    ACORN 인덱스를 빌드해 out_index_path에 저장.
    - /tmp 그대로 사용
    - LD_LIBRARY_PATH 보강 (conda lib + lib_dirs)
    - 컴파일/실행 로그 출력
    필요 전역: CPP_SOURCE_BUILD (C++ 소스 문자열)
    """
    import os, tempfile, shutil, subprocess
    from pathlib import Path

    libs = list(libs or ["faiss"])
    extra_cxxflags = list(extra_cxxflags or [])
    extra_ldflags  = list(extra_ldflags or [])
    include_dirs   = list(include_dirs or [])
    lib_dirs       = list(lib_dirs or [])

    if not include_dirs:
        raise ValueError("include_dirs가 비었습니다. faiss/IndexACORN.h 상위 경로를 넣어주세요.")
    if not lib_dirs:
        raise ValueError("lib_dirs가 비었습니다. libfaiss.so(.a) 경로를 넣어주세요.")

    # ✅ 환경변수 세팅
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(num_threads))
    env.setdefault("OPENBLAS_NUM_THREADS", str(blas_threads))
    env.setdefault("MKL_NUM_THREADS", str(blas_threads))
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("OMP_DYNAMIC", "FALSE")
    env.setdefault("LC_ALL", "C")
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})

    # ✅ LD_LIBRARY_PATH 보강
    conda_prefix = os.environ.get("CONDA_PREFIX")
    ld_paths = []
    if conda_prefix:
        ld_paths.append(os.path.join(conda_prefix, "lib"))
    ld_paths.extend(lib_dirs)
    cur_ld = env.get("LD_LIBRARY_PATH", "")
    if cur_ld:
        ld_paths.append(cur_ld)
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths)

    tmpdir = None
    try:
        # tmp 밑에 그대로 생성
        tmpdir = tempfile.mkdtemp(prefix="acorn_build_")
        cpp_path = Path(tmpdir) / "acorn_build.cpp"
        exe_path = Path(tmpdir) / "acorn_build"

        with open(cpp_path, "w") as f:
            f.write(CPP_SOURCE_BUILD)

        # ✅ 컴파일 명령: -pipe + rpath
        cmd = [cxx, "-O3", "-std=c++17", "-fopenmp", "-pipe",
            str(cpp_path), "-o", str(exe_path)]
        for inc in include_dirs:
            cmd.append(f"-I{inc}")
        for libd in lib_dirs:
            cmd += [f"-L{libd}", f"-Wl,-rpath,{libd}"]
        if conda_prefix:
            cmd.append(f"-Wl,-rpath,{os.path.join(conda_prefix, 'lib')}")
        for lib in libs:
            cmd.append(f"-l{lib}")
        if "openblas" not in libs and "blas" not in libs:
            cmd.append("-lopenblas")
        if "pthread" not in libs:
            cmd.append("-lpthread")
        cmd += extra_cxxflags + extra_ldflags

        comp = subprocess.run(cmd, env=env, text=True,
                            capture_output=True, timeout=timeout)
        if comp.stdout:
            print("g++ STDOUT:\n", comp.stdout)
        if comp.stderr:
            print("g++ STDERR:\n", comp.stderr)
        if comp.returncode != 0:
            raise RuntimeError(f"compile failed (rc={comp.returncode})")

        os.makedirs(os.path.dirname(out_index_path) or ".", exist_ok=True)
        run_cmd = [
            str(exe_path),
            str(int(d)), str(int(M)), str(int(gamma)), str(int(M_beta)),
            filename, out_index_path
        ]
        proc = subprocess.run(run_cmd, text=True,
                            capture_output=True, env=env, timeout=timeout)
        proc.check_returncode()


    finally:
        if not keep_temp and tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)

CPP_SOURCE_SEARCH = r"""
#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>
#include <memory>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <cstdio>
#include <cstdint>
// ---- Thread control shim (put near the top) ----
#if __has_include(<faiss/utils/omp_utils.h>)
#include <faiss/utils/omp_utils.h>
static inline void set_threads_from_env() {
    if (const char* s = std::getenv("OMP_NUM_THREADS"))
        faiss::omp_set_num_threads(std::atoi(s));
}
#elif __has_include(<omp.h>)
#include <omp.h>
static inline void set_threads_from_env() {
    if (const char* s = std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::atoi(s));
}
#else
static inline void set_threads_from_env() { /* no-op */ }
#endif


static void make_dirs(const std::string& path) {
    if (path.empty()) return;
    std::string p = path;
    while (!p.empty() && p.back() == '/') p.pop_back();
    if (p.empty()) return;
    auto pos = p.find_last_of('/');
    if (pos != std::string::npos) {
        make_dirs(p.substr(0, pos));
    }
    if (::mkdir(p.c_str(), 0755) != 0 && errno != EEXIST) {
        // ignore; customize if needed
    }
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = std::fopen(fname, "rb");
    if (!f) throw std::runtime_error(std::string("cannot open fvecs: ") + fname);

    int32_t d32 = 0;
    if (std::fread(&d32, sizeof(int32_t), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("failed to read dimension");
    }
    if (d32 <= 0) {
        std::fclose(f);
        throw std::runtime_error("invalid dimension in fvecs");
    }
    const size_t d = (size_t)d32;

    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        throw std::runtime_error("fseek failed");
    }
    long long fsize = std::ftell(f);
    if (fsize < 0) {
        std::fclose(f);
        throw std::runtime_error("ftell failed");
    }
    std::rewind(f);

    const size_t rec_bytes = (d + 1) * sizeof(int32_t);
    if ((unsigned long long)fsize % rec_bytes != 0ULL) {
        std::fclose(f);
        throw std::runtime_error("file size is not aligned to (d+1)*4 bytes — not an fvecs?");
    }
    const size_t n = (size_t)((unsigned long long)fsize / rec_bytes);

    float* xb = new float[n * d];

    for (size_t i = 0; i < n; ++i) {
        int32_t cur_d = 0;
        if (std::fread(&cur_d, sizeof(int32_t), 1, f) != 1) {
            delete[] xb; std::fclose(f);
            throw std::runtime_error("failed to read record header");
        }
        if (cur_d != d32) {
            delete[] xb; std::fclose(f);
            throw std::runtime_error("dimension mismatch inside fvecs file");
        }
        if (std::fread(xb + i * d, sizeof(float), d, f) != d) {
            delete[] xb; std::fclose(f);
            throw std::runtime_error("truncated fvecs file");
        }
    }

    std::fclose(f);
    *d_out = d;
    *n_out = n;
    return xb;
}

int main(int argc, char** argv) {
    // argv:
    // 1: d
    // 2: M
    // 3: gamma
    // 4: M_beta
    // 5: k
    // 6: faiss_index_path
    // 7: db_vectors_path (현재는 사용하지 않음; 인덱스에서 nb=ntotal 사용)
    // 8: query_vectors_path
    // 9: db_filters_path
    // 10: query_filters_path
    // 11: out_I_path
    // 12: bit_map
    if (argc < 10) {
        std::cerr << "Usage: " << argv[0]
                << " <M> <gamma> <M_beta> <faiss_index.faiss>"
                << " <db_vectors.fvec> <query_vectors.fvec>"
                << " <db_filters.txt> <query_filters.txt>"
                << " <out_I.txt>\n";
        return 1;
    }
    
    int d      = std::stoi(argv[1]);
    int M      = std::stoi(argv[2]);
    int gamma  = std::stoi(argv[3]);
    int M_beta = std::stoi(argv[4]);
    int k      = std::stoi(argv[5]);
    std::string idx_path   = argv[6];
    std::string dbv_path   = argv[7]; // not used
    std::string qv_path    = argv[8];
    std::string dbf_path   = argv[9];
    std::string qf_path    = argv[10];
    std::string out_I      = argv[11];
    std::string bit_map    = argv[12];

    try {
        // Read index
        std::unique_ptr<faiss::Index> base(faiss::read_index(idx_path.c_str()));
        auto* acorn = dynamic_cast<faiss::IndexACORNFlat*>(base.get());
        if (!acorn) {
            throw std::runtime_error("Loaded index is not IndexACORNFlat");
        }
        const int d = acorn->d;
        const size_t nb = acorn->ntotal;

        // Read queries
        size_t nq = 0, dq = 0;
        float* xq = fvecs_read(qv_path.c_str(), &dq, &nq);
        if ((int)d != (int)dq) {
            delete[] xq;
            throw std::runtime_error("query dim != index dim");
        }

        std::ifstream qf(qf_path);
        if (!qf.is_open()) { delete[] xq; throw std::runtime_error("cannot open query_filters.txt"); }
        std::ifstream dbf(dbf_path);
        if (!dbf.is_open()) { delete[] xq; throw std::runtime_error("cannot open db_filters.txt"); }

        // ---- bit_map 로드: 텍스트/바이너리 자동 처리 ----
        std::vector<char> filter_ids_map;
        {
            const size_t expected_bits = (size_t)nq * (size_t)nb;

            // 1) 바이너리로 열어 크기 확인
            std::ifstream bm_bin(bit_map, std::ios::binary);
            if (!bm_bin.is_open()) {
                delete[] xq;
                throw std::runtime_error("cannot open bit_map file: " + bit_map);
            }
            bm_bin.seekg(0, std::ios::end);
            std::streamoff fsize = bm_bin.tellg();
            bm_bin.seekg(0, std::ios::beg);

            const size_t expected_bytes_packed = (expected_bits + 7) / 8; // 비트패킹(1bit/flag)
            const size_t expected_bytes_byte   = expected_bits;           // 바이트당 0/1

            if ((size_t)fsize == expected_bytes_byte) {
                // ---- 케이스 A: 바이트당 0/1 저장 ----
                std::vector<uint8_t> buf(expected_bytes_byte);
                if (!bm_bin.read(reinterpret_cast<char*>(buf.data()), buf.size())) {
                    delete[] xq;
                    throw std::runtime_error("failed to read byte-per-flag bitmap");
                }
                filter_ids_map.resize(expected_bits);
                for (size_t i = 0; i < expected_bits; ++i) {
                    filter_ids_map[i] = (buf[i] != 0) ? 1 : 0;
                }
            } else if ((size_t)fsize == expected_bytes_packed) {
                // ---- 케이스 B: 비트패킹(8개 플래그가 1바이트) ----
                std::vector<uint8_t> buf(expected_bytes_packed);
                if (!bm_bin.read(reinterpret_cast<char*>(buf.data()), buf.size())) {
                    delete[] xq;
                    throw std::runtime_error("failed to read bit-packed bitmap");
                }
                filter_ids_map.resize(expected_bits);
                // 주의: 작성할 때 LSB-first( bitpos 0..7 )로 넣었다면 같은 방식으로 풀어야 함
                size_t out = 0;
                for (size_t i = 0; i < buf.size(); ++i) {
                    uint8_t byte = buf[i];
                    for (int b = 0; b < 8 && out < expected_bits; ++b) {
                        filter_ids_map[out++] = ((byte >> b) & 1) ? 1 : 0; // LSB-first
                    }
                }
            } else {
                // ---- 케이스 C: 텍스트(공백/개행 구분 "0"/"1")로 가정하여 파싱 ----
                bm_bin.close(); // 텍스트로 다시 엶
                std::ifstream bm_txt(bit_map);
                if (!bm_txt.is_open()) {
                    delete[] xq;
                    throw std::runtime_error("cannot open bit_map file (text): " + bit_map);
                }
                filter_ids_map.resize(expected_bits, 0);
                size_t cnt = 0;
                std::string tok;
                while (bm_txt >> tok) {
                    if (cnt >= expected_bits) break;
                    // '0' 또는 '1'만 허용
                    if (tok == "0" || tok == "1") {
                        filter_ids_map[cnt++] = (tok == "1") ? 1 : 0;
                    } else {
                        // 숫자 문자열일 가능성에도 대비
                        try {
                            int v = std::stoi(tok);
                            filter_ids_map[cnt++] = (v != 0) ? 1 : 0;
                        } catch (...) {
                            // 무시하고 계속
                        }
                    }
                }
                if (cnt != expected_bits) {
                    delete[] xq;
                    throw std::runtime_error(
                        "bit_map entries (" + std::to_string(cnt) +
                        ") != nq*nb (" + std::to_string(expected_bits) + ")");
                }
            }
        }

        // Search & timing
        std::vector<faiss::idx_t> I(nq * k);
        std::vector<float> D(nq * k);

        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();
        acorn->search(nq, xq, k, D.data(), I.data(), filter_ids_map.data());
        auto t1 = clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double s  = ms / 1000.0;
        double qps = (s > 0.0) ? (double)nq / s : 0.0;
        double ms_per_q = (nq > 0) ? ms / (double)nq : 0.0;

        std::cout << std::fixed << std::setprecision(3)
                << "[ACORN search] elapsed: " << ms << " ms, "
                << "ms/query: " << ms_per_q << ", "
                << "QPS: " << qps << "\n";

        // Save I
        auto slash = out_I.find_last_of('/');
        if (slash != std::string::npos) {
            make_dirs(out_I.substr(0, slash));
        }
        std::ofstream ofs(out_I);
        if (!ofs.is_open()) {
            delete[] xq;
            throw std::runtime_error(std::string("cannot open output: ") + out_I);
        }
        for (size_t i = 0; i < nq; ++i) {
            for (int j = 0; j < k; ++j) {
                ofs << I[i * k + j];
                if (j + 1 < k) ofs << ' ';
            }
            ofs << '\n';
        }
        ofs.close();

        delete[] xq;

        // 기계적으로 파싱하기 쉬운 라인도 함께 출력
        std::cout << "QPS=" << std::setprecision(10) << qps << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 2;
    }
}
"""

def run_acorn_search_from_python(
    d: int,
    M: int,
    gamma: int,
    M_beta: int,
    k: int,
    faiss_index_path: str,
    db_vectors_path: str,
    query_vectors_path: str,
    db_filters_path: str,
    query_filters_path: str,
    out_I_path: str,
    bit_map: str,
    include_dirs,
    lib_dirs,
    libs=None,
    cxx: str = "g++",
    extra_cxxflags=None,
    extra_ldflags=None,
    keep_temp: bool = False,
    num_threads: int = 200,
    blas_threads: int = 1,
    env_overrides: dict | None = None,
    timeout: float | None = None,
) -> float:
    import os, tempfile, shutil, subprocess, re
    from pathlib import Path

    libs = list(libs or ["faiss"])
    extra_cxxflags = list(extra_cxxflags or [])
    extra_ldflags = list(extra_ldflags or [])
    include_dirs = list(include_dirs or [])
    lib_dirs = list(lib_dirs or [])

    if not include_dirs:
        raise ValueError("include_dirs 가 비었습니다. faiss/IndexACORN.h 상위 경로를 넣어주세요.")
    if not lib_dirs:
        raise ValueError("lib_dirs 가 비었습니다. libfaiss.so(.a) 경로를 넣어주세요.")

    # ✅ 서브프로세스 환경변수 세팅
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(num_threads))
    env.setdefault("OPENBLAS_NUM_THREADS", str(blas_threads))
    env.setdefault("MKL_NUM_THREADS", str(blas_threads))
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("OMP_DYNAMIC", "FALSE")
    env.setdefault("LC_ALL", "C")
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})

    # ✅ LD_LIBRARY_PATH 보강 (conda lib + lib_dirs 포함)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    ld_paths = []
    if conda_prefix:
        ld_paths.append(os.path.join(conda_prefix, "lib"))
    ld_paths.extend(lib_dirs)
    current_ld = env.get("LD_LIBRARY_PATH", "")
    if current_ld:
        ld_paths.append(current_ld)
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths)

    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="acorn_search_")
        cpp_path = Path(tmpdir) / "acorn_search.cpp"
        exe_path = Path(tmpdir) / "acorn_search"

        with open(cpp_path, "w") as f:
            f.write(CPP_SOURCE_SEARCH)

        cmd = [cxx, "-O3", "-std=c++17", "-fopenmp", str(cpp_path), "-o", str(exe_path)]
        for inc in include_dirs:
            cmd.append(f"-I{inc}")
        for libd in lib_dirs:
            cmd += [f"-L{libd}", f"-Wl,-rpath,{libd}"]
        if conda_prefix:
            cmd.append(f"-Wl,-rpath,{os.path.join(conda_prefix,'lib')}")
        for lib in libs:
            cmd.append(f"-l{lib}")
        if "openblas" not in libs and "blas" not in libs:
            cmd.append("-lopenblas")
        if "pthread" not in libs:
            cmd.append("-lpthread")
        cmd += extra_cxxflags + extra_ldflags

        subprocess.check_call(cmd, env=env, timeout=timeout)

        os.makedirs(os.path.dirname(out_I_path), exist_ok=True)

        run_cmd = [
            str(exe_path),
            str(int(d)),
            str(int(M)),
            str(int(gamma)),
            str(int(M_beta)),
            str(int(k)),
            faiss_index_path,
            db_vectors_path,
            query_vectors_path,
            db_filters_path,
            query_filters_path,
            out_I_path,
            bit_map,
        ]

        proc = subprocess.run(run_cmd, text=True, capture_output=True, env=env, timeout=timeout)

        proc.check_returncode()

        m = re.search(r"QPS=([0-9.+-eE]+)", proc.stdout)
        if not m:
            raise RuntimeError("QPS 값을 STDOUT에서 찾지 못했습니다.")
        qps = float(m.group(1))
        return qps

    finally:
        if not keep_temp and tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)

CPP_SOURCE_bit_map = r"""
#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>
#include <memory>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

// ---- Thread control shim (optional) ----
#if __has_include(<faiss/utils/omp_utils.h>)
#include <faiss/utils/omp_utils.h>
static inline void set_threads_from_env() {
    if (const char* s = std::getenv("OMP_NUM_THREADS"))
        faiss::omp_set_num_threads(std::atoi(s));
}
#elif __has_include(<omp.h>)
#include <omp.h>
static inline void set_threads_from_env() {
    if (const char* s = std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::atoi(s));
}
#else
static inline void set_threads_from_env() { /* no-op */ }
#endif

static void make_dirs(const std::string& path) {
    if (path.empty()) return;
    std::string p = path;
    while (!p.empty() && p.back() == '/') p.pop_back();
    if (p.empty()) return;
    auto pos = p.find_last_of('/');
    if (pos != std::string::npos) {
        make_dirs(p.substr(0, pos));
    }
#if defined(_WIN32)
    _mkdir(p.c_str());
#else
    if (::mkdir(p.c_str(), 0755) != 0 && errno != EEXIST) {
        // ignore
    }
#endif
}

// fvecs 파일에서 전체 레코드 수(n)만 계산 (메모리 로드X)
static size_t fvecs_count_records(const std::string& path, size_t& d_out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open fvecs: " + path);
    int32_t d32 = 0;
    if (std::fread(&d32, sizeof(int32_t), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("failed to read dimension from: " + path);
    }
    if (d32 <= 0) {
        std::fclose(f);
        throw std::runtime_error("invalid dimension in: " + path);
    }
    d_out = static_cast<size_t>(d32);

    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        throw std::runtime_error("fseek failed: " + path);
    }
    long long fsize = std::ftell(f);
    std::fclose(f);
    if (fsize < 0) throw std::runtime_error("ftell failed: " + path);

    const size_t rec_bytes = (d_out + 1) * sizeof(int32_t);
    if ((unsigned long long)fsize % rec_bytes != 0ULL) {
        throw std::runtime_error("file size not multiple of (d+1)*4: " + path);
    }
    return static_cast<size_t>((unsigned long long)fsize / rec_bytes);
}

int main(int argc, char** argv) {
    // argv:
    // 1: d (unused here, optional sanity check)
    // 2: faiss_index_path
    // 3: db_vectors_path (unused)
    // 4: query_vectors_path (.fvecs)
    // 5: db_filters_path (text)
    // 6: query_filters_path (text)
    // 7: out_I_path (unused)
    // 8: bit_map (output path; **binary** bit-packed)
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0]
                << " <d> <faiss_index.faiss>"
                << " <db_vectors.fvec> <query_vectors.fvec>"
                << " <db_filters.txt> <query_filters.txt>"
                << " <out_I.txt> <bit_map.bin>\n";
        return 1;
    }

    set_threads_from_env();

    int d_arg  = std::stoi(argv[1]);  (void)d_arg;
    std::string idx_path   = argv[2];
    std::string dbv_path   = argv[3]; (void)dbv_path; // unused
    std::string qv_path    = argv[4];
    std::string dbf_path   = argv[5];
    std::string qf_path    = argv[6];
    std::string out_I      = argv[7]; (void)out_I;    // unused
    std::string bit_map    = argv[8];                 // **binary output**

    try {
        // 1) 인덱스에서 nb 확인
        std::unique_ptr<faiss::Index> base(faiss::read_index(idx_path.c_str()));
        auto* acorn = dynamic_cast<faiss::IndexACORNFlat*>(base.get());
        if (!acorn) {
            throw std::runtime_error("Loaded index is not IndexACORNFlat");
        }
        const size_t nb = acorn->ntotal;

        // 2) 쿼리 fvecs에서 nq만 계산(파일 크기 기반) + dq 확인
        size_t dq = 0;
        size_t nq = fvecs_count_records(qv_path, dq);
        // (선택) d_arg와 dq 체크하려면 아래 주석 해제
        // if (d_arg > 0 && (size_t)d_arg != dq) {
        //     throw std::runtime_error("d mismatch: argv[1]=" + std::to_string(d_arg) +
        //                              " vs qvecs d=" + std::to_string(dq));
        // }

        // 3) 필터 파일 열기
        std::ifstream qf(qf_path);
        if (!qf.is_open()) throw std::runtime_error("cannot open query_filters.txt: " + qf_path);
        std::ifstream dbf(dbf_path);
        if (!dbf.is_open()) throw std::runtime_error("cannot open db_filters.txt: " + dbf_path);

        // 4) 출력 준비 (바이너리)
        auto slash = bit_map.find_last_of('/');
        if (slash != std::string::npos) make_dirs(bit_map.substr(0, slash));
        std::ofstream bm(bit_map, std::ios::binary);
        if (!bm.is_open()) throw std::runtime_error("cannot open bit_map for write: " + bit_map);

        // 진행률/시간
        const unsigned long long total = (unsigned long long)nq * (unsigned long long)nb;
        const unsigned long long step  = (total / 100ULL) ? (total / 100ULL) : 1ULL;
        auto t_start = std::chrono::steady_clock::now();

        // 비트 패킹 버퍼
        std::vector<uint8_t> bytebuf;
        bytebuf.reserve(1 << 20); // 1MB
        uint8_t cur_byte = 0;
        int     bitpos   = 0;

        auto flush_byte = [&]() {
            bytebuf.push_back(cur_byte);
            cur_byte = 0;
            bitpos = 0;
            if (bytebuf.size() >= (1 << 20)) {
                bm.write(reinterpret_cast<const char*>(bytebuf.data()), bytebuf.size());
                bytebuf.clear();
            }
        };
        auto put_bit = [&](bool bit) {
            if (bit) cur_byte |= (1u << bitpos);
            if (++bitpos == 8) flush_byte();
        };

        // 5) 비트맵 생성 (qid-major, 총 nq*nb개 비트)
        std::string qline, dbline;
        size_t qid = 0;
        unsigned long long wrote = 0ULL;

        while (std::getline(qf, qline)) {
            if (qid >= nq) break;

            // DB 필터를 처음부터 다시
            dbf.clear();
            dbf.seekg(0, std::ios::beg);

            size_t did = 0;
            while (std::getline(dbf, dbline)) {
                if (did >= nb) break;

                bool ok = true;
                const size_t L = qline.size();
                if (dbline.size() < L) {
                    ok = false;
                } else {
                    for (size_t i = 0; i < L; ++i) {
                        if (qline[i] != dbline[i]) {
                            if (!(qline[i] == '0' && dbline[i] == '1')) {
                                ok = false; break;
                            }
                        }
                    }
                }

                put_bit(ok);
                ++wrote; ++did;

                                // 진행률(1% 단위)
                if ((wrote % step) == 0 || wrote == total) {
                    double pct = (total ? (100.0 * (double)wrote / (double)total) : 100.0);
                    auto   now = std::chrono::steady_clock::now();
                    double sec = std::chrono::duration<double>(now - t_start).count();
                    std::cout << "\r진행률: " << std::fixed << std::setprecision(2)
                            << pct << "% (" << wrote << "/" << total
                            << ") 경과: " << sec << "s" << std::flush;
                }
            }

            if (did != nb) {
                throw std::runtime_error("db_filters lines < nb (" +
                                        std::to_string(did) + " < " + std::to_string(nb) + ")");
            }
            ++qid;
        }

        if (qid != nq) {
            throw std::runtime_error("query_filters lines < nq (" +
                                    std::to_string(qid) + " < " + std::to_string(nq) + ")");
        }

        if (wrote != total) {
            throw std::runtime_error("bitmap size mismatch, wrote=" +
                                    std::to_string(wrote) + " expected=" + std::to_string(total));
        }

        // 남은 비트 플러시
        if (bitpos > 0) flush_byte();
        if (!bytebuf.empty()) {
            bm.write(reinterpret_cast<const char*>(bytebuf.data()), bytebuf.size());
            bytebuf.clear();
        }

        std::cerr << "\n[INFO] BITMAP_DONE nq=" << nq
                << " nb=" << nb
                << " bits=" << total
                << " bytes≈" << ((total + 7ULL) / 8ULL) << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 2;
    }
}

"""

def build_bit_map_from_python(
    d: int,
    faiss_index_path: str,
    db_vectors_path: str,
    query_vectors_path: str,
    db_filters_path: str,
    query_filters_path: str,
    out_I_path: str,   # 여기선 사용 안하지만 argv 형식 맞춤
    bit_map: str,
    include_dirs,            # e.g. ["/home/mintaek/hybrid_index/methods/ACORN"]
    lib_dirs,                # e.g. ["/home/mintaek/hybrid_index/methods/ACORN/build/faiss"]
    libs=None,               # e.g. ["faiss"] or ["faiss_avx2"]
    cxx: str = "g++",
    extra_cxxflags=None,     # e.g. ["-march=native"]
    extra_ldflags=None,      # e.g. ["-Wl,--no-as-needed"]
    keep_temp: bool = False,
    # ↓ 실행 제어 옵션
    num_threads: int = 64,
    blas_threads: int = 1,
    env_overrides: dict | None = None,
    timeout: float | None = None,
):
    """
    bit_map 생성 전용. 성공 시 (nq, nb) 튜플 반환.
    CPP_SOURCE_bit_map 이 전역에 정의되어 있어야 하며,
    C++ main은 아래 argv 형식을 파싱해야 합니다:
    d idx_path db_vectors q_vectors db_filters q_filters out_I out_bitmap
    """
    import os, tempfile, shutil, subprocess, re, sys, time
    from pathlib import Path

    libs = list(libs or ["faiss"])
    extra_cxxflags = list(extra_cxxflags or [])
    extra_ldflags = list(extra_ldflags or [])
    include_dirs = list(include_dirs or [])
    lib_dirs = list(lib_dirs or [])

    if not include_dirs:
        raise ValueError("include_dirs 가 비었습니다. faiss/IndexACORN.h 상위 경로를 넣어주세요.")
    if not lib_dirs:
        raise ValueError("lib_dirs 가 비었습니다. libfaiss.so(.a) 경로를 넣어주세요.")

    # 입력 파일 체크
    for p in [faiss_index_path, db_vectors_path, query_vectors_path, db_filters_path, query_filters_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[INPUT MISSING] {p}")
    os.makedirs(os.path.dirname(bit_map) or ".", exist_ok=True)

    # 환경변수(스레드 고정)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(num_threads))
    env.setdefault("OPENBLAS_NUM_THREADS", str(blas_threads))
    env.setdefault("MKL_NUM_THREADS", str(blas_threads))
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("OMP_DYNAMIC", "FALSE")
    env.setdefault("LC_ALL", "C")
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})

    # ✅ LD_LIBRARY_PATH 보강 (conda lib + lib_dirs 포함)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    ld_paths = []
    if conda_prefix:
        ld_paths.append(os.path.join(conda_prefix, "lib"))
    ld_paths.extend(lib_dirs)
    current_ld = env.get("LD_LIBRARY_PATH", "")
    if current_ld:
        ld_paths.append(current_ld)
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths)

    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="acorn_bitmap_")
        cpp_path = Path(tmpdir) / "make_bitmap.cpp"
        exe_path = Path(tmpdir) / "make_bitmap"

        with open(cpp_path, "w") as f:
            f.write(CPP_SOURCE_bit_map)

        cmd = [cxx, "-O3", "-std=c++17", "-fopenmp", str(cpp_path), "-o", str(exe_path)]
        for inc in include_dirs:
            cmd.append(f"-I{inc}")
        for libd in lib_dirs:
            cmd += [f"-L{libd}", f"-Wl,-rpath,{libd}"]
        # ✅ conda lib 경로도 rpath에 추가 (런타임 .so 탐색 안정화)
        if conda_prefix:
            cmd.append(f"-Wl,-rpath,{os.path.join(conda_prefix,'lib')}")
        for lib in libs:
            cmd.append(f"-l{lib}")
        if "openblas" not in libs and "blas" not in libs:
            cmd.append("-lopenblas")
        if "pthread" not in libs:
            cmd.append("-lpthread")
        cmd += extra_cxxflags + extra_ldflags

        subprocess.check_call(cmd, env=env, timeout=timeout)

        run_cmd = [
            str(exe_path),
            str(int(d)),
            faiss_index_path,
            db_vectors_path,
            query_vectors_path,
            db_filters_path,
            query_filters_path,
            out_I_path,
            bit_map,
        ]

        # ✅ 방법 A: 실시간 스트리밍 + 로그 누적
        p = subprocess.Popen(
            run_cmd,
            env=env,                      # ← 중요: 실행에도 같은 env 전달
            text=True,
            bufsize=1,                    # line-buffered
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,     # cout/cerr 합쳐서 읽기
        )

        captured = []
        t0 = time.time()
        try:
            for line in iter(p.stdout.readline, ""):
                captured.append(line)
                sys.stdout.flush()
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            raise TimeoutError(f"subprocess timed out after {timeout}s")
        finally:
            if p.stdout:
                p.stdout.close()

        if rc != 0:
            raise RuntimeError(f"subprocess exited with code {rc}")

        full_log = "".join(captured)

        # BITMAP_DONE 라인에서 nq, nb 파싱 (cout/cerr 어디든 OK)
        m = re.search(r"BITMAP_DONE\s+nq=([0-9]+)\s+nb=([0-9]+)", full_log)
        if not m:
            raise RuntimeError("nq/nb 파싱 실패: 출력에서 BITMAP_DONE 라인을 찾지 못했습니다.")
        nq, nb = int(m.group(1)), int(m.group(2))

        if not os.path.isfile(bit_map):
            raise RuntimeError(f"bit_map 파일이 생성되지 않았습니다: {bit_map}")

        return nq, nb

    finally:
        if not keep_temp and tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


# --- 핵심 함수: 행(row) 선택 ---

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


def Analyze(mode):
    if mode == True:
        return "post_base"

    dataset_name = "sift1m"

    ACORN_trade_off = {}
    num_file = 0

    k = 10

    params_list = [
        (32, 12, 64),
        (40, 15, 80),
        (48, 18, 96),
        (56, 21, 112),
        (64, 24, 128)
    ]

    batches = [[] for _ in range(2)]

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


        # -----------------------------
        # 1. 스키마(필터 조건) 로드
        # -----------------------------
        with open(f"{data_path}/hardness_format/filters.json", "r", encoding="utf-8") as f:
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

        input_path = f"{data_path}/hardness_format/payloads.jsonl"  # 한 줄당 하나의 JSON 객체
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

        # 원본 .npy 파일 경로
        npy_path = f"{data_path}/hardness_format/vectors.npy"
        # 변환된 .fvec 파일 저장 경로
        fvec_path = f"{data_path}/mid_format/db_vectors_ACORN.fvec"

        # .npy 파일 불러오기
        data = np.load(npy_path)

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



        with open(f"{data_path}/hardness_format/filters.json", "r", encoding="utf-8") as f:
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


        input_path = f"{data_path}/hardness_format/tests.jsonl"  # 한 줄당 하나의 JSON 객체
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


        input_path = f"{data_path}/hardness_format/tests.jsonl"  # 한 줄당 하나의 JSON 객체
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

        fname = f"{data_path}/mid_format/db_vectors_ACORN.fvec"
        assert os.path.exists(fname), f"입력 파일 없음: {fname}"

        # fvec 헤더에서 dimension 읽기
        with open(fname, "rb") as f:
            d_from_file = np.fromfile(f, dtype=np.int32, count=1)[0]

        # (선택) 파일 정렬/레코드 수도 확인
        size = os.path.getsize(fname)
        rec_bytes = (d_from_file + 1) * 4

        faiss_include_dirs = [
            "/home/mintaek/hybrid_index/methods/ACORN",  # 이 경로 아래에 faiss/IndexACORN.h 가 있어야 함
        ]
        faiss_lib_dirs = [
            "/home/mintaek/hybrid_index/methods/ACORN/build/faiss",
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
        if not os.path.exists(bit_map_path):
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

            l += 1

        npy_path = f"{data_path}/hardness_format/vectors.npy"

        data = np.load(npy_path)
        nv = data.shape[0]


        for M, gamma, M_beta in params_list:
            idx_path = f"{data_path}/ACORN_format/ACORN_index_{M}_{gamma}_{M_beta}.faiss"
            qps_batch = run_acorn_search_from_python(
                d=int(d_from_file), M=M, gamma=gamma, M_beta=M_beta, k=k,
                faiss_index_path=idx_path,
                db_vectors_path=f"{data_path}/mid_format/db_vectors_ACORN.fvec",
                query_vectors_path=f"{data_path}/mid_format/query_vectors_ACORN.fvec",
                db_filters_path=f"{data_path}/mid_format/db_filter_ACORN.txt",
                query_filters_path=f"{data_path}/mid_format/query_filter_ACORN.txt",
                out_I_path=f"{data_path}/ACORN_format/knn_I.txt",
                bit_map=f"{data_path}/ACORN_format/bit_map.txt",
                include_dirs=faiss_include_dirs,
                lib_dirs=faiss_lib_dirs
                #libs=["faiss"],   # GPU 빌드라면 CUDA 관련 libs 추가
            )
            recall = find_recall(k, f"{data_path}/ACORN_format/knn_I.txt", f"{data_path}/mid_format/query_gt_ACORN.txt", nv)
            batches[num_file].append((recall, qps_batch))
        
        num_file += 1

    for i in [0, 1]:
        a={}
        for j, par in enumerate(params_list):
            a[par] = {"qps": batches[i][j][1], "avg_recall": batches[i][j][0]}
        if i == 0:
            ACORN_trade_off["closer_to_post"] = a
        else:
            ACORN_trade_off["closer_to_pre"] = a


            # --- 결과 요약 ---
    post_score = pc.final_score(ACORN_trade_off["closer_to_post"])
    pre_score = pc.final_score(ACORN_trade_off["closer_to_pre"])

    print("\n[Final Scores]")
    print(f"  ├─ Post-base Score : {post_score:.6f}")
    print(f"  └─ Pre-base  Score : {pre_score:.6f}")

    if post_score > pre_score:
        print("[Decision] → ✅ post_base selected")
        return "post_base"
    else:
        print("[Decision] → ✅ pre_base selected")
        return "pre_base"
    
    if __name__ == "__main__":
        main()

