set(tgt bench-conv-delta)
add_executable(${tgt} WIN32 delta.c tictoc.c tictoc.h)
target_link_libraries(${tgt} heffalump_gpu)
set_target_properties(${tgt} PROPERTIES FOLDER bench)