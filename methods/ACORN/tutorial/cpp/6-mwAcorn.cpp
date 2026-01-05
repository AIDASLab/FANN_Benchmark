#include <iostream>
#include <random>
#include <vector>

#include <faiss/IndexACORN.h>

int main() {
    int d=128;
    int M=32; 
    int gamma=12;
    int M_beta=32;

    int nb = 100000;

    int nq = 10000;

    // (옵션) 메타데이터: 노드별 정수 속성 (예: 카테고리)
    std::vector<int> metadata_data(nb, 0);
    for (size_t i = 0; i < metadata_data.size(); ++i) metadata_data[i] = i % 13;

    std::vector<int> metadata_query(nq, 0);
    for (size_t i = 0; i < metadata_query.size(); ++i) metadata_query[i] = i % 13;

    // 인덱스 생성: Flat 저장 + ACORN 네비게이션 (L2)
    faiss::IndexACORNFlat index(d, M, gamma, metadata_data, M_beta);

    // 더미 데이터 생성
    std::mt19937 rng(12345);
    std::uniform_real_distribution<> distrib;
    std::vector<float> xb(nb * d), xq(nq * d);

    for (size_t i = 0; i < nb; ++i) {
        for (int j = 0; j < d; ++j) xb[i * d + j] = distrib(rng);
        xb[i * d + 0] += float(i) / 1000.f; // 이웃 구조를 만들기 위한 살짝의 오프셋
    }
    for (size_t i = 0; i < nq; ++i) {
        for (int j = 0; j < d; ++j) xq[i * d + j] = distrib(rng);
        xq[i * d + 0] += float(i) / 1000.f;
    }

    // (필요시) 학습형 스토리지일 때만 train 필요. Flat은 불필요지만 호출해도 무해.
    index.train(nb, xb.data());

    // 추가(순차 삽입)
    index.add(nb, xb.data());

    // 검색
    int k = 10;
    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);

    std::vector<char> filter_ids_map(nq * nb);
    for (int xq = 0; xq < nq; xq++) {
        for (int xb = 0; xb < nb; xb++) {
            filter_ids_map[xq * nb + xb] = false;//(bool) (metadata_data[xb] == metadata_query[xq]);
        }
    }

    index.search(nq, xq.data(), k, D.data(), I.data(), filter_ids_map.data());

    for (size_t i = 0; i < nq; ++i) {
        if(i % 100 == 0){
            std::cout << "q" << i << ": \n";
            for (int j = 0; j < k; ++j) std::cout << "MetaData: " << metadata_data[I[i * k + j]] << ", Data: "
            << I[i * k + j] << ", Distance: " << D[i * k + j] << "\n";
            std::cout << "\n";
        }
    }
}