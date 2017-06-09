set(tgt max_gpu)
cuda_add_executable(${tgt} WIN32 max.cu)
set_target_properties(${tgt} PROPERTIES FOLDER examples)