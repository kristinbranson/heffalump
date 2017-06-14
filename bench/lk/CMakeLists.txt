set(tgt bench-lk-gpu)
add_executable(${tgt} WIN32 main.c tictoc.c tictoc.h)
target_link_libraries(${tgt} hog_gpu)
set_target_properties(${tgt} PROPERTIES FOLDER bench)

set(tgt bench-lk-cpu)
add_executable(${tgt} WIN32 main.c tictoc.c tictoc.h)
target_link_libraries(${tgt} hog_cpu)
set_target_properties(${tgt} PROPERTIES FOLDER bench)