#   Copyright 2017 Vidrio Technologies
#   by Nathan Clack <nathan@vidriotech.com>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0

set(tgt hogshow)
add_library(${tgt} STATIC hogshow.h hogshow.c)
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(${tgt} PROPERTIES FOLDER examples/libs)