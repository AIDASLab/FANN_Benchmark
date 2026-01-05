#include <iostream>
#include <random>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <fstream>

#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = std::fopen(fname, "rb");
    if (!f) throw std::runtime_error(std::string("cannot open fvecs: ") + fname);

    // 1) 첫 헤더에서 차원 d 읽기
    int32_t d32 = 0;
    if (std::fread(&d32, sizeof(int32_t), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("failed to read dimension");
    }
    if (d32 <= 0) {
        std::fclose(f);
        throw std::runtime_error("invalid dimension in fvecs");
    }
    const size_t d = static_cast<size_t>(d32);

    // 2) 파일 크기에서 총 벡터 수 n 계산
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

    const size_t rec_bytes = (d + 1) * sizeof(int32_t);   // 헤더(int32) + d * float32(=int32 size)
    if (static_cast<unsigned long long>(fsize) % rec_bytes != 0ULL) {
        std::fclose(f);
        throw std::runtime_error("file size is not aligned to (d+1)*4 bytes — not an fvecs?");
    }
    const size_t n = static_cast<size_t>(static_cast<unsigned long long>(fsize) / rec_bytes);

    // 3) 결과 버퍼 할당 (n * d floats)
    float* xb = new float[n * d];

    // 4) 레코드 단위로 읽기 (헤더 d 확인 후 데이터 복사)
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

int main(){
    int d=2048;
    int M=32; 
    int gamma=12;
    int M_beta=32;

    // (옵션) 메타데이터: 노드별 정수 속성 (예: 카테고리)
    std::vector<int> metadata(10, 0);

    // 인덱스 생성: Flat 저장 + ACORN 네비게이션 (L2)
    faiss::IndexACORNFlat acorn_gamma(d, M, gamma, metadata, M_beta);

    size_t nb, d2;
    std::string filename = "/home/mintaek/hybrid_index/hybrid_benchmark_outdated/hnm/ACORN_hnm/db_vectors.fvec"; // your fvec file
    float* xb = fvecs_read(filename.c_str(), &d2, &nb);
    assert(d == d2 || !"dataset dimension is not as expected");
    acorn_gamma.add(nb, xb);

    faiss::write_index(&acorn_gamma, "/home/mintaek/hybrid_index/methods/ACORN/build/tutorial/cpp/acorn_hnm_index.faiss");
}