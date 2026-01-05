set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(CPP_SOURCES vamana.cpp)
add_library(Vamana ${CPP_SOURCES})

target_link_libraries(Vamana ${PROJECT_NAME} Boost::filesystem)