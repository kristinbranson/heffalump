add_subdirectory(SFMT-src-1.5.1 sfmt)

set(tgt main)
add_executable(${tgt} WIN32 main.c tictoc.c tictoc.h)
target_link_libraries(${tgt} hog sfmt)
