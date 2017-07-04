#   Copyright 2017 Vidrio Technologies
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Author: Nathan Clack <nathan at vidriotech dot com>
cmake_minimum_required(VERSION 3.0)
project(heffalump)

include(CMakeToolsHelpers OPTIONAL)

find_package(CUDA)
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Common)
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

enable_testing()
add_subdirectory(test)

include(InstallRequiredSystemLibraries)
include(CPack) # enable packaging
