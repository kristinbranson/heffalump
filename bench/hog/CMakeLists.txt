set(tgt bench-hog-disks)
add_executable(${tgt} WIN32 disks.c tictoc.c tictoc.h)
target_link_libraries(${tgt} hog_gpu)
set_target_properties(${tgt} PROPERTIES FOLDER bench)