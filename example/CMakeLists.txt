add_subdirectory(SFMT-src-1.5.1 sfmt)
add_subdirectory(imshow)

set_target_properties(sfmt   PROPERTIES FOLDER examples/libs)
set_target_properties(imshow PROPERTIES FOLDER examples/libs)

set(tgt conv-rand)
add_executable(${tgt} WIN32 conv/rand.c tictoc.c tictoc.h)
target_link_libraries(${tgt} hog_cpu sfmt imshow)
set_target_properties(${tgt} PROPERTIES FOLDER examples)

set(tgt conv-delta)
add_executable(${tgt} WIN32 conv/delta.c tictoc.c tictoc.h)
target_link_libraries(${tgt} hog_cpu imshow)
set_target_properties(${tgt} PROPERTIES FOLDER examples)

set(tgt lk)
add_executable(${tgt} WIN32 lk/main.c tictoc.c tictoc.h)
target_link_libraries(${tgt} hog_cpu sfmt imshow)
set_target_properties(${tgt} PROPERTIES FOLDER examples)

set(tgt hog)
add_executable(${tgt} WIN32 hog/main.c tictoc.c tictoc.h hog/hogshow.c)
target_link_libraries(${tgt} hog_cpu sfmt imshow)
set_target_properties(${tgt} PROPERTIES FOLDER examples)