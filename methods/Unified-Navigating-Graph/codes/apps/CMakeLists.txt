set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

add_executable(build_UNG_index build_UNG_index.cpp)
target_link_libraries(build_UNG_index ${PROJECT_NAME} Vamana Boost::program_options Boost::filesystem)

add_executable(search_UNG_index search_UNG_index.cpp)
target_link_libraries(search_UNG_index ${PROJECT_NAME} Vamana Boost::program_options Boost::filesystem)

add_executable(filtered_scan filtered_scan.cpp)
target_link_libraries(filtered_scan ${PROJECT_NAME} Vamana Boost::program_options Boost::filesystem)
