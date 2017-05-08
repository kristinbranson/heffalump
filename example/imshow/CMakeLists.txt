set(tgt imshow)
add_library(${tgt} STATIC
    imshow.c   imshow.h
    app.c      app.h
    mingl.c    mingl.h
)
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(${tgt} opengl32)

