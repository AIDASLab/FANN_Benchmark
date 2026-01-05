set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

add_executable(fvecs_to_bin fvecs_to_bin.cpp)
target_link_libraries(fvecs_to_bin ${PROJECT_NAME} Boost::program_options)

add_executable(generate_base_labels generate_base_labels.cpp)
target_link_libraries(generate_base_labels ${PROJECT_NAME} Boost::program_options)

add_executable(generate_query_labels generate_query_labels.cpp)
target_link_libraries(generate_query_labels ${PROJECT_NAME} Boost::program_options)

add_executable(compute_groundtruth compute_groundtruth.cpp)
target_link_libraries(compute_groundtruth ${PROJECT_NAME} Boost::program_options)