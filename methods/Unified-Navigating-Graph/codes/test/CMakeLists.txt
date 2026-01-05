set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

add_executable(test_build_vamana test_build_vamana.cpp)
target_link_libraries(test_build_vamana ${PROJECT_NAME} Vamana Boost::program_options)

add_executable(test_search_vamana test_search_vamana.cpp)
target_link_libraries(test_search_vamana ${PROJECT_NAME} Vamana Boost::program_options)