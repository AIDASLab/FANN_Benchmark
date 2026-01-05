#include <iostream>
#include <random>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <fstream>

#include <faiss/IndexACORN.h>

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


int main() {
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
    
    std::cout << "Add Done." << std::endl;

    size_t nq, d3;
    std::string filename_q = "/home/mintaek/hybrid_index/hybrid_benchmark_outdated/hnm/ACORN_hnm/query_vectors.fvec"; // your fvec file
    float* xq = fvecs_read(filename_q.c_str(), &d3, &nq);
    assert(d == d3 || !"dataset dimension is not as expected");


    // 검색
    int k = 10;
    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);

    std::vector<char> filter_ids_map(nq * nb);
    
    std::ifstream file("/home/mintaek/hybrid_index/hybrid_benchmark_outdated/hnm/ACORN_hnm/query_filters.txt"); // 파일 열기
    if (!file.is_open()) {
        std::cerr << "파일을 열 수 없습니다.\n";
        return 1;
    }

    std::ifstream file2("/home/mintaek/hybrid_index/hybrid_benchmark_outdated/hnm/ACORN_hnm/db_filters.txt"); // 파일 열기
    if (!file2.is_open()) {
        std::cerr << "파일을 열 수 없습니다.\n";
        return 1;
    }

    std::string line;
    std::string line2;

    int line_no = 0;
    int line2_no = 0;
    int n = 0;
    int n1 = 0;


    while(std::getline(file, line)){
        line2_no = 0;

        file2.clear();                  // EOF/에러 플래그 초기화
        file2.seekg(0, std::ios::beg);  // 파일 포인터 맨 앞으로

        while(std::getline(file2, line2)){
            n1++;
            for(int i = 0; i < line.size(); i++){
                filter_ids_map[line_no * nb + line2_no] = true;
                if(line[i] != line2[i]){
                    if(!(line[i] == '0' && line2[i] == '1')){
                        n++;
                        filter_ids_map[line_no * nb + line2_no] = false;
                        break;
                    }
                }
            }

            //filter_ids_map[line_no * nb + line2_no] = false;
            line2_no++;
        }
        line_no++;
        if(line_no % 10 == 0){
            std::cout << "진행률: " << line_no*100.0/nq << "%" << std::endl;
        }
    }


    acorn_gamma.search(nq, xq, k, D.data(), I.data(), filter_ids_map.data());

    for (size_t i = 0; i < nq; ++i) {
        if(i % 100 == 0){
            std::cout << "q" << i << ": \n";
            for (int j = 0; j < k; ++j) std::cout //<< "MetaData: " << metadata_data[I[i * k + j]]
            << "Data: " << I[i * k + j] << ", Distance: " << D[i * k + j] << "\n";
            std::cout << "\n";
        }
    }

    std::cout << line_no << " " << line2_no << " " << n << " " << n1 << std::endl;

    delete[] xb;
    delete[] xq;
}