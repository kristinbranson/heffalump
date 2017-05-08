add_subdirectory(SFMT-src-1.5.1 sfmt)
add_subdirectory(imshow)

set(tgt main)
add_executable(${tgt} WIN32 main.c tictoc.c tictoc.h)
target_link_libraries(${tgt} hog_cpu sfmt imshow)
