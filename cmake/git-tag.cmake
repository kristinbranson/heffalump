# Copyright 2017 Nathan Clack
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
find_program(GIT git)
if(GIT)
    exec_program(${GIT} ${CMAKE_CURRENT_SOURCE_DIR} ARGS "describe --tags" OUTPUT_VARIABLE GIT_TAG)
    exec_program(${GIT} ${CMAKE_CURRENT_SOURCE_DIR} ARGS "describe --always" OUTPUT_VARIABLE GIT_HASH)
    add_definitions(-DGIT_TAG=${GIT_TAG})
    add_definitions(-DGIT_HASH=${GIT_HASH})
    set(CPACK_PACKAGE_VERSION ${GIT_TAG})
    message("Version ${GIT_TAG} ${GIT_HASH}")
else()
    add_definitions(-DGIT_TAG="Unknown" -DGIT_HASH=" ")
endif()
