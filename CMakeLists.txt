cmake_minimum_required(VERSION 3.0)

find_package(CUDA)

include(cmake/git-tag.cmake)
include(cmake/ide.cmake)

add_subdirectory(src)
add_subdirectory(example)

