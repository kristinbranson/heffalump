add_subdirectory(cpu/pdollar gradientHist)

set(tgt hog_gpu)
cuda_add_library(${tgt} STATIC 
    conv.h gpu/conv.cu
    gpu/gradientHist.h gpu/gradientHist.cu
    gpu/hog.c
)
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_property(TARGET ${tgt} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${tgt} PROPERTY C_STANDARD   99)

set(tgt hog_cpu)
add_library(${tgt} STATIC
    conv.h cpu/conv.cpp
    lk.h   cpu/lk.c cpu/diff.cpp
    hog.h  cpu/hog.c
    hof.h  cpu/hof.c
)
target_link_libraries(${tgt} gradientHist)
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_property(TARGET ${tgt} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${tgt} PROPERTY C_STANDARD   99)
