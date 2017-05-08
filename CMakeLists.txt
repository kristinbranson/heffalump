cmake_minimum_required(VERSION 3.0)
project(hog)

find_package(CUDA)

include(cmake/git-tag.cmake)
include(cmake/ide.cmake)

add_subdirectory(src)
add_subdirectory(example)

