#   Copyright 2017 Vidrio Technologies
#   by Nathan Clack <nathan@vidriotech.com>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0

set(tgt bench-hof-gpu)
add_executable(${tgt} WIN32 main.c tictoc.c tictoc.h)
target_link_libraries(${tgt} heffalump_gpu)
set_target_properties(${tgt} PROPERTIES FOLDER bench)

set(tgt bench-hof-nopipelines-gpu)
add_executable(${tgt} WIN32 main.nopipeline.c tictoc.c tictoc.h)
target_link_libraries(${tgt} heffalump_gpu)
set_target_properties(${tgt} PROPERTIES FOLDER bench)

set(tgt bench-hof-cpu)
add_executable(${tgt} WIN32 main.c tictoc.c tictoc.h)
target_link_libraries(${tgt} heffalump_cpu)
set_target_properties(${tgt} PROPERTIES FOLDER bench)