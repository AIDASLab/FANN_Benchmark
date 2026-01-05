set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(CPP_SOURCES utils.cpp storage.cpp trie.cpp distance.cpp search_queue.cpp filtered_scan.cpp uni_nav_graph.cpp)
add_library(${PROJECT_NAME} ${CPP_SOURCES})
