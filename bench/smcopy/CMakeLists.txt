set(tgt smcopy)
cuda_add_executable(${tgt} main.cu)
set_target_properties(${tgt} PROPERTIES FOLDER bench)