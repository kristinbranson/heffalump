SET(tgt streams)
cuda_add_executable(${tgt} WIN32 stream.cu)
set_target_properties(${tgt} PROPERTIES FOLDER examples)