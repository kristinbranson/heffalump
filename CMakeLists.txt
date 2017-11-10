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
IF(NOT WIN32)
set(CUDA_HOST_FLAGS "-Xcompiler -fPIC -std=c++11")
ENDIF()
set(CUDA_NVCC_FLAGS_DEBUG          "${CUDA_FPIC_FLAGS}-g;-G" CACHE STRING "" FORCE)
set(CUDA_NVCC_FLAGS_RELEASE        "${CUDA_FPIC_FLAGS}--use_fast_math" CACHE STRING "" FORCE)
set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "${CUDA_FPIC_FLAGS}--use_fast_math;-g;-lineinfo" CACHE STRING "" FORCE)
set(CUDA_NVCC_FLAGS_MINSIZEREL     "${CUDA_FPIC_FLAGS}--use_fast_math" CACHE STRING "" FORCE)

message("CUDA_HOST_FLAGS is ${CUDA_HOST_FLAGS}")
message("CUDA_NVCC_FLAGS_RELEASE is ${CUDA_NVCC_FLAGS_RELEASE}")


include(cmake/git-tag.cmake)
include(cmake/ide.cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_STANDARD   99)
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(src)
add_subdirectory(bench)
if(WIN32)
    add_subdirectory(example)    
endif()

enable_testing()
add_subdirectory(test)

include(InstallRequiredSystemLibraries)
include(CPack) # enable packaging
