set(tgt hog_gpu)
cuda_add_library(${tgt} STATIC hog.cu)

set(tgt hog_cpu)
add_library(${tgt} STATIC
    conv.h cpu/conv.cpp
    lk.h   cpu/lk.c cpu/diff.cpp
)
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_property(TARGET ${tgt} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${tgt} PROPERTY C_STANDARD   99)
