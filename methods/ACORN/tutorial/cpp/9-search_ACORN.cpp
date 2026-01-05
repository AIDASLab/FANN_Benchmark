#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>
#include <memory>
#include <fstream>
#include <chrono>
#include <iomanip>     // (선택) 출력 포맷
#include <filesystem>  // 파일 저장 전에 상위 폴더 생성용 (C++17)

#include <sys/stat.h>
#include <sys/types.h>
#include <cstring>

// 부모 디렉터리(재귀) 생성
static void make_dirs(const std::string& path);

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out);


int main(){

    int d=2048;
    int M=32; 
    int gamma=12;
    int M_beta=32;


    // 읽기
    std::unique_ptr<faiss::Index> base(faiss::read_index("/home/mintaek/hybrid_index/Benchmark/HnM/ACORN_format/ACORN_index.faiss"));

    // ACORN 전용 search(filter_ids_map 있는 오버로드)를 쓰려면 캐스팅
    auto* acorn_gamma = dynamic_cast<faiss::IndexACORNFlat*>(base.get());
    if (!acorn_gamma) {
        throw std::runtime_error("Loaded index is not IndexACORNFlat");
    }
    
    size_t nb, d2;
    std::string filename = "/home/mintaek/hybrid_index/Benchmark/HnM/mid_format/db_vectors_ACORN.fvec"; // your fvec file
    float* xb = fvecs_read(filename.c_str(), &d2, &nb);
    assert(d == d2 || !"dataset dimension is not as expected");

    size_t nq, d3;
    std::string filename_q = "/home/mintaek/hybrid_index/Benchmark/HnM/ACORN_format/batch0/query_vectors_ACORN.fvec"; // your fvec file
    float* xq = fvecs_read(filename_q.c_str(), &d3, &nq);
    assert(d == d3 || !"dataset dimension is not as expected");


    // 검색
    int k = 10;
    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);

    std::vector<char> filter_ids_map(nq * nb);
    
    std::ifstream file("/home/mintaek/hybrid_index/Benchmark/HnM/ACORN_format/batch0/query_filters_ACORN.txt"); // 파일 열기
    if (!file.is_open()) {
        std::cerr << "파일을 열 수 없습니다.\n";
        return 1;
    }

    std::ifstream file2("/home/mintaek/hybrid_index/Benchmark/HnM/mid_format/db_filter_ACORN.txt"); // 파일 열기
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
            for(int i = 0; i < line.size(); i++){
                filter_ids_map[line_no * nb + line2_no] = true;
                if(line[i] != line2[i]){
                    if(!(line[i] == '0' && line2[i] == '1')){
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
}


// 부모 디렉터리(재귀) 생성
static void make_dirs(const std::string& path) {
    if (path.empty()) return;
    // 끝의 '/' 제거
    std::string p = path;
    while (!p.empty() && p.back() == '/') p.pop_back();
    if (p.empty()) return;

    // 부모 먼저
    auto pos = p.find_last_of('/');
    if (pos != std::string::npos) {
        make_dirs(p.substr(0, pos));
    }
    // 자신 생성 (이미 있으면 통과)
    if (::mkdir(p.c_str(), 0755) != 0 && errno != EEXIST) {
        // 필요시 에러 처리
    }
}



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