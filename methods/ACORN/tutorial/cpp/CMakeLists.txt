# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(1-Flat EXCLUDE_FROM_ALL 1-Flat.cpp)
target_link_libraries(1-Flat PRIVATE faiss)

add_executable(2-IVFFlat EXCLUDE_FROM_ALL 2-IVFFlat.cpp)
target_link_libraries(2-IVFFlat PRIVATE faiss)

add_executable(3-IVFPQ EXCLUDE_FROM_ALL 3-IVFPQ.cpp)
target_link_libraries(3-IVFPQ PRIVATE faiss)

add_executable(4-GPU EXCLUDE_FROM_ALL 4-GPU.cpp)
target_link_libraries(4-GPU PRIVATE faiss)

add_executable(5-Multiple-GPUs EXCLUDE_FROM_ALL 5-Multiple-GPUs.cpp)
target_link_libraries(5-Multiple-GPUs PRIVATE faiss)

add_executable(6-mwAcorn EXCLUDE_FROM_ALL 6-mwAcorn.cpp)
target_link_libraries(6-mwAcorn PRIVATE faiss)

add_executable(7-ACORN_hnm EXCLUDE_FROM_ALL 7-ACORN_hnm.cpp)
target_link_libraries(7-ACORN_hnm PRIVATE faiss)

add_executable(8-mk_ACORN_idx EXCLUDE_FROM_ALL 8-mk_ACORN_idx.cpp)
target_link_libraries(8-mk_ACORN_idx PRIVATE faiss)

add_executable(9-search_ACORN EXCLUDE_FROM_ALL 9-search_ACORN.cpp)
target_link_libraries(9-search_ACORN PRIVATE faiss)