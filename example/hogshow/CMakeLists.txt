set(tgt hogshow)
add_library(${tgt} STATIC hogshow.h hogshow.c)
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(${tgt} PROPERTIES FOLDER examples/libs)