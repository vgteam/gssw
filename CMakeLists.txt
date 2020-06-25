# Specify the minimum version for CMake

cmake_minimum_required(VERSION 3.1)

# Project's name
project(gssw)
# We build using c++14
set(CMAKE_CXX_STANDARD 14)

# Use all standard-compliant optimizations
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")

#set(CMAKE_BUILD_TYPE Release)

# set up our target executable and specify its dependencies and includes
add_library(gssw_objs OBJECT
  ${CMAKE_SOURCE_DIR}/src/gssw.c)
add_library(libgssw STATIC $<TARGET_OBJECTS:gssw_objs>)
set_target_properties(libgssw PROPERTIES OUTPUT_NAME "gssw")

add_executable(gssw-test
  $<TARGET_OBJECTS:gssw_objs>
  ${CMAKE_SOURCE_DIR}/src/gssw_test.c)
target_include_directories(gssw-test PUBLIC
  "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(gssw-test
  -lm)

if (APPLE)
elseif (TRUE)
  if (BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "-static")
  endif()
endif()
