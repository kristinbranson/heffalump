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

cmake_minimum_required(VERSION 3.0)
project(hog)

include(CMakeToolsHelpers OPTIONAL)

find_package(CUDA)
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS_DEBUG          -g;-G CACHE STRING "" FORCE)
set(CUDA_NVCC_FLAGS_RELEASE        --use_fast_math CACHE STRING "" FORCE)
set(CUDA_NVCC_FLAGS_RELWITHDEBINFO --use_fast_math;-g;-lineinfo CACHE STRING "" FORCE)
set(CUDA_NVCC_FLAGS_MINSIZEREL     --use_fast_math CACHE STRING "" FORCE)

include(cmake/git-tag.cmake)
include(cmake/ide.cmake)

add_subdirectory(src)
add_subdirectory(bench)

if(WIN32)
    add_subdirectory(example)    
endif()

include(InstallRequiredSystemLibraries)
include(CPack) # enable packaging
