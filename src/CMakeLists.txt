# Copyright 2017 Nathan Clack
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
add_subdirectory(cpu/pdollar gradientHist)

set(tgt hog_gpu)
cuda_add_library(${tgt} STATIC 
    conv.h gpu/conv.h gpu/conv.cu
    gpu/gradientHist.h gpu/gradientHist.cu
    gpu/hog.c
    gpu/absmax.h gpu/absmax.cu
    lk.h gpu/lk.h gpu/lk.cu
    hof.h gpu/hof.cpp
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

install(FILES conv.h lk.h hog.h hog.h DESTINATION include)
install(TARGETS hog_gpu hog_cpu DESTINATION lib)