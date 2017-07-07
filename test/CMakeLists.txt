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
    )
        set(tgt ${label}-${arch})
        add_executable(${tgt} ${label}.cpp)
        target_link_libraries(${tgt} heffalump_${arch})
        set_target_properties(${tgt} PROPERTIES FOLDER test/${arch})
        add_test(NAME test-${tgt} COMMAND ${tgt})
        target_compile_definitions(${tgt} PRIVATE HEFFALUMP_TEST_${arch}=1)
    endforeach()
endforeach()