add_subdirectory(SFMT-src-1.5.1 sfmt)
add_subdirectory(imshow)
add_subdirectory(hogshow)

set_target_properties(sfmt   PROPERTIES FOLDER examples/libs)
set_target_properties(imshow PROPERTIES FOLDER examples/libs)

foreach(impl cpu gpu)

set(tgt conv-rand-${impl})
add_executable(${tgt} WIN32 conv/rand.c tictoc.c tictoc.h)
target_link_libraries(${tgt} hog_${impl} sfmt imshow)
set_target_properties(${tgt} PROPERTIES FOLDER examples)

set(tgt conv-delta-${impl})
add_executable(${tgt} WIN32 conv/delta.c tictoc.c tictoc.h)
target_link_libraries(${tgt} hog_${impl} imshow)
set_target_properties(${tgt} PROPERTIES FOLDER examples)

set(tgt lk-${impl})
add_executable(${tgt} WIN32 lk/main.c tictoc.c tictoc.h)
target_link_libraries(${tgt} hog_${impl} sfmt imshow)
set_target_properties(${tgt} PROPERTIES FOLDER examples)

set(tgt hog-${impl})
add_executable(${tgt} WIN32 hog/main.c tictoc.c tictoc.h)
target_link_libraries(${tgt} hog_${impl} sfmt imshow hogshow)
set_target_properties(${tgt} PROPERTIES FOLDER examples)

set(tgt hof-${impl})
add_executable(${tgt} WIN32 hof/main.c tictoc.c tictoc.h)
target_link_libraries(${tgt} hog_${impl} sfmt imshow hogshow)
set_target_properties(${tgt} PROPERTIES FOLDER examples)

endforeach(impl cpu gpu)

