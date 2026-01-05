#include <fstream>
#include <iostream>
#include <cstring>
#include <boost/program_options.hpp>
#include "config.h"

namespace po = boost::program_options;


/*
.fvec files start with 4 bytes for the number of dimensions, then each vector takes 4+dim*4 bytes
.bin files start with 4 bytes for the number of vectors, then 4 bytes for the number of dimensions, and each vector takes dim*4 bytes
*/

int main(int argc, char** argv) {
    std::string data_type, input_file, output_file;
    try {
        po::options_description desc{"Arguments"};

        desc.add_options()("help", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                           "Data type of the vectors: float/int8/uint8");
        desc.add_options()("input_file", po::value<std::string>(&input_file)->required(),
                           "Filename for input *.fvecs file");
        desc.add_options()("output_file", po::value<std::string>(&output_file)->required(),
                           "Filename for output *.bin file");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }
    
    // check data type
    uint32_t data_size = sizeof(float);
    if (data_type == "int8" || data_type == "uint8") {
        data_size = sizeof(uint8_t);
    } else if (data_type != "float") {
        std::cerr << "Error: type not supported. Use float/int8/uint8" << std::endl;
        exit(-1);
    }

    // obtain dimension and number of vectors
    uint32_t dim;
    ANNS::IdxType num_vecs;
    std::ifstream fvec_file(input_file, std::ios::binary | std::ios::ate);
    size_t file_size = fvec_file.tellg();
    fvec_file.seekg(0, std::ios::beg);
    fvec_file.read((char *)&dim, sizeof(uint32_t));
    num_vecs = file_size / (dim * data_size + sizeof(uint32_t));
    std::cout << "Dataset: #pts = " << num_vecs << ", # dims = " << dim << std::endl;

    // dump to binary file
    std::ofstream bin_file(output_file, std::ios::binary);
    bin_file.write((char *)&num_vecs, sizeof(ANNS::IdxType));
    bin_file.write((char *)&dim, sizeof(uint32_t));

    // dump vector data from fvec_file to bin_file
    char *buffer = new char[dim * data_size], *tmp = new char[sizeof(uint32_t)];
    for (ANNS::IdxType i = 0; i < num_vecs; i++) {
        if (i > 0)
            fvec_file.read(tmp, sizeof(uint32_t));
        fvec_file.read(buffer, dim * data_size);
        bin_file.write(buffer, dim * data_size);
    }
    delete[] buffer;

    // clean
    fvec_file.close();
    bin_file.close();
    return 0;
}