set(tgt scan)
cuda_add_executable(${tgt} WIN32 scan.cu)
set_target_properties(${tgt} PROPERTIES FOLDER examples)