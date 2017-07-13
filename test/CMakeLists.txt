#   Copyright 2017 Vidrio Technologies
#   by Nathan Clack <nathan@vidriotech.com>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0

foreach(arch
    gpu
    cpu
)
    foreach(label
        memcheck        
        conv-param-check
        lk-param-check
        hog-param-check
        hof-param-check
        hog-output-size
        hof-output-size
    )
        set(tgt ${label}-${arch})
        add_executable(${tgt} ${label}.cpp)
        target_link_libraries(${tgt} heffalump_${arch})
        set_target_properties(${tgt} PROPERTIES FOLDER test/${arch})
        add_test(NAME test-${tgt} COMMAND ${tgt})
        target_compile_definitions(${tgt} PRIVATE HEFFALUMP_TEST_${arch}=1)
    endforeach()
endforeach()