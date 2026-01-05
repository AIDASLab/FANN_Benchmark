#include <omp.h>
#include <fstream>
#include <string>
#include <cstring>
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include "utils.h"
#include "storage.h"


namespace ANNS {

    // claim the class
    template class Storage<float>;
    template class Storage<int8_t>;
    template class Storage<uint8_t>;


    // obtain the corresponding storage class
    std::shared_ptr<IStorage> create_storage(const std::string& data_type, bool verbose) {
        if (data_type == "float") 
            return std::make_shared<Storage<float>>(DataType::FLOAT, verbose);
        else if (data_type == "int8")
            return std::make_shared<Storage<int8_t>>(DataType::INT8, verbose);
        else if (data_type == "uint8")
            return std::make_shared<Storage<uint8_t>>(DataType::UINT8, verbose);
        else {
            std::cerr << "Error: invalid data type " << data_type << std::endl;
            exit(-1);
        }
    }


    // obtain the corresponding storage class
    std::shared_ptr<IStorage> create_storage(std::shared_ptr<IStorage> storage, IdxType start, IdxType end) {
        DataType data_type = storage->get_data_type();
        if (data_type == DataType::FLOAT) 
            return std::make_shared<Storage<float>>(storage, start, end);
        else if (data_type == DataType::INT8)
            return std::make_shared<Storage<int8_t>>(storage, start, end);
        else if (data_type == DataType::UINT8)
            return std::make_shared<Storage<uint8_t>>(storage, start, end);
        else {
            std::cerr << "Error: invalid data type " << data_type << std::endl;
            exit(-1);
        }
    }
    

    // construct the class
    template<typename T>
    Storage<T>::Storage(DataType data_type, bool verbose) {
        this->data_type = data_type;
        this->verbose = verbose;
    }



    // load from storage without copying data
    template<typename T>
    Storage<T>::Storage(std::shared_ptr<IStorage> storage, IdxType start, IdxType end) {
        data_type = storage->get_data_type();
        num_points = end - start;
        dim = storage->get_dim();
        vecs = reinterpret_cast<T *>(storage->get_vector(start));
        label_sets = storage->get_offseted_label_sets(start);
        prefetch_byte_num = dim * sizeof(T);
        verbose = false;
    }

    

    // load data
    template<typename T>
    void Storage<T>::load_from_file(const std::string& bin_file, const std::string& label_file, IdxType max_num_points) {
        if (verbose)
            std::cout << "Loading data from " << bin_file << " and " << label_file << " ..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // open the binary file
        std::ifstream file(bin_file, std::ios::binary | std::ios::ate);
        if (!file.is_open())
            throw std::runtime_error("Failed to open file: " + bin_file);

        // read vector data
        file.seekg(0, std::ios::beg);
        file.read((char *)&num_points, sizeof(IdxType));
        file.read((char *)&dim, sizeof(IdxType));
        num_points = std::min(num_points, max_num_points);
        vecs = static_cast<T*>(std::aligned_alloc(32, num_points * dim * sizeof(T)));
        file.read((char *)vecs, num_points * dim * sizeof(T));
        file.close();

        // for prefetch
        prefetch_byte_num = dim * sizeof(T);

        // read label data if exists
        std::map<LabelType, IdxType> label_cnts;
        label_sets = new std::vector<LabelType>[num_points];
        file.open(label_file);
        if (file.is_open()) {
            std::string line, label;
            for (auto i=0; i<num_points && i<max_num_points; ++i) {
                std::getline(file, line);
                std::stringstream ss(line);
                while (std::getline(ss, label, ',')) {
                    label_sets[i].emplace_back(std::stoi(label));
                    if (label_cnts.find(std::stoi(label)) == label_cnts.end())
                        label_cnts[std::stoi(label)] = 1;
                    else
                        label_cnts[std::stoi(label)]++;
                }
                std::sort(label_sets[i].begin(), label_sets[i].end());
                label_sets[i].shrink_to_fit();
            }
            file.close();

        // unfiltered ANNS when label file not found 
        } else {
            std::cout << "- Warning: label file not found, set all labels to 1" << std::endl;
            for (auto i=0; i<num_points && i<max_num_points; ++i)
                label_sets[i] = {1};
            label_cnts[1] = num_points;
        }

        // statistics
        if (verbose) {
            std::cout << "- Number of points: " << num_points << std::endl;
            std::cout << "- Dimension: " << dim << std::endl;
            std::cout << "- Number of labels: " << label_cnts.size() << std::endl;
            std::cout << "- Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
            std::cout << SEP_LINE;
        }
    }



    // write data
    template<typename T>
    void Storage<T>::write_to_file(const std::string& bin_file, const std::string& label_file) {

        // write vector data
        std::ofstream file(bin_file, std::ios::binary);
        file.write((char *)&num_points, sizeof(IdxType));
        file.write((char *)&dim, sizeof(IdxType));
        file.write((char *)vecs, num_points * dim * sizeof(T));
        file.close();

        // write label data
        file.open(label_file);
        for (auto i=0; i<num_points; ++i) {
            file << label_sets[i][0];
            for (auto j=1; j<label_sets[i].size(); ++j)
                file << "," << label_sets[i][j];
            file << std::endl;
        }
        file.close();
    }



    // reorder the vector data
    template<typename T>
    void Storage<T>::reorder_data(const std::vector<IdxType>& new_to_old_ids) {

        // move the vectors and labels
        auto new_vecs = static_cast<T*>(std::aligned_alloc(32, num_points * dim * sizeof(T)));
        auto new_label_sets = new std::vector<LabelType>[num_points];
        for (auto i=0; i<num_points; ++i) {
            std::memcpy(new_vecs + i * dim, vecs + new_to_old_ids[i] * dim, dim * sizeof(T));
            new_label_sets[i] = label_sets[new_to_old_ids[i]];
        }
        
        // clean up
        delete[] vecs;
        delete[] label_sets;
        vecs = new_vecs;
        label_sets = new_label_sets;
    }



    // obtain a point cloest to the center
    template<typename T>
    IdxType Storage<T>::choose_medoid(uint32_t num_threads, std::shared_ptr<DistanceHandler> distance_handler) {

        // compute center
        T* center = new T[dim]();
        for (auto id=0; id<num_points; ++id) 
            for (auto d=0; d<dim; ++d)
                center[d] += *(vecs + id * dim + d);
        for (auto d=0; d<dim; ++d)
            center[d] /= num_points;

        // obtain the closet point to the center
        std::vector<float> dists(num_points);
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic, 2048)
        for (auto id=0; id<num_points; ++id)
            dists[id] = distance_handler->compute((const char *)center, get_vector(id), dim);
        IdxType medoid = std::min_element(dists.begin(), dists.end()) - dists.begin();

        // clean up
        delete[] center;
        return medoid;
    }
}