set(tgt bench-lk-gpu)
add_executable(${tgt} WIN32 main.c tictoc.c tictoc.h)
target_link_libraries(${tgt} heffalump_gpu)
set_target_properties(${tgt} PROPERTIES FOLDER bench)

set(tgt bench-lk-cpu)
add_executable(${tgt} WIN32 main.c tictoc.c tictoc.h)
target_link_libraries(${tgt} heffalump_cpu)
set_target_properties(${tgt} PROPERTIES FOLDER bench)