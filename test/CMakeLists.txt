foreach(arch
    gpu
    cpu
)
    foreach(label
        memcheck
        lk-param-check
        conv-param-check
    )
        set(tgt ${label}-${arch})
        add_executable(${tgt} ${label}.cpp)
        target_link_libraries(${tgt} heffalump_${arch})
        set_target_properties(${tgt} PROPERTIES FOLDER test)
        add_test(NAME test-${tgt} COMMAND ${tgt})
    endforeach()
endforeach()