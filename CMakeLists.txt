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
