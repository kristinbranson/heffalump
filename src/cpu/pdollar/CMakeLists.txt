set(tgt gradientHist) 
add_library(${tgt} STATIC gradientHist.cpp sse.hpp wrappers.hpp)
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR})