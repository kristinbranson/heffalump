add_library(sfmt STATIC 
    SFMT.h SFMT.c
)
target_include_directories(sfmt PUBLIC ${CMAKE_CURRENT_LIST_DIR})