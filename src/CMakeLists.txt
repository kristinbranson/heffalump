#   Copyright 2017 Vidrio Technologies
#   by Nathan Clack <nathan@vidriotech.com>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0

add_subdirectory(cpu/pdollar gradientHist)

set(tgt heffalump_gpu)
cuda_add_library(${tgt} STATIC 
    conv.h gpu/conv.h gpu/conv.cu
    gpu/gradientHist.h gpu/gradientHist.cu
    gpu/hog.c
    gpu/absmax.h gpu/absmax.cu
    lk.h gpu/lk.h gpu/lk.cu
    hof.h gpu/hof.cpp
    version.h version.c
)
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_property(TARGET ${tgt} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${tgt} PROPERTY C_STANDARD   99)

set(tgt heffalump_cpu)
add_library(${tgt} STATIC
    conv.h cpu/conv.cpp
    lk.h   cpu/lk.c cpu/diff.cpp
    hog.h  cpu/hog.c
    hof.h  cpu/hof.c
    version.h version.c
)
target_link_libraries(${tgt} gradientHist)
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_property(TARGET ${tgt} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${tgt} PROPERTY C_STANDARD   99)

install(FILES conv.h lk.h hog.h hof.h version.h DESTINATION include/heffalump)
install(TARGETS heffalump_gpu heffalump_cpu DESTINATION lib)