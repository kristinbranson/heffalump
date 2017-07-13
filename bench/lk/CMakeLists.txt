#   Copyright 2017 Vidrio Technologies
#   by Nathan Clack <nathan@vidriotech.com>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0

set(tgt bench-lk-gpu)
add_executable(${tgt} main.c tictoc.c tictoc.h)
target_link_libraries(${tgt} heffalump_gpu)
set_target_properties(${tgt} PROPERTIES FOLDER bench)

set(tgt bench-lk-cpu)
add_executable(${tgt} main.c tictoc.c tictoc.h)
target_link_libraries(${tgt} heffalump_cpu)
set_target_properties(${tgt} PROPERTIES FOLDER bench)
