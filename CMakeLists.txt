cmake_minimum_required(VERSION 3.13)
project(nagatolib)

set(CMAKE_CXX_STANDARD 17)

# Build Options
option(NAGATO_BUILD_EXAMPLE "Build example" ON)
option(NAGATO_MATH "Use nagato math function" ON)
option(NAGATO_TEST "Build Test" ON)

# find packages
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (NAGATO_MATH)
    add_definitions(-DNAGATO_MATH)
endif ()

# compile options
set(CMAKE_CXX_FLAGS_DEBUG
        "-O0 -Wall ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE
        "-O3 -Wall ${CMAKE_CXX_FLAGS_RELEASE}")

# Build Nagato Lib
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
file(GLOB NAGATO_HEADER
        src/*.hpp)

file(GLOB NAGATO_SOURCE
        src/*.cpp)

add_library(nagato ${NAGATO_HEADER} ${NAGATO_SOURCE})

function(add_example target_name)
    add_executable(${target_name}
            example/${target_name}.cpp)
    target_link_libraries(${target_name} PUBLIC
            nagato
            "-pthread")
    target_include_directories(${target_name} PUBLIC
            ${CMAKE_SOURCE_DIR}/src)
endfunction()

# Build Example
if (NAGATO_BUILD_EXAMPLE)
    add_example(vector_example)
    add_example(random_example)
    add_example(thread_example)
    add_example(matrix_example)

endif ()

# Build Tests
if (NAGATO_TEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test/gtest)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test/gtest/googletest/include)
    file(GLOB
            NAGATO_TEST_SOURCE
            ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp)
    add_executable(nagato_test
            ${NAGATO_TEST_SOURCE})
    target_link_libraries(nagato_test PUBLIC
            nagato
            gtest
            "-pthread")
    target_include_directories(nagato_test PUBLIC
            ${CMAKE_SOURCE_DIR}/src)
endif ()

function(add_verify target_name)
    add_executable(${target_name} ${target_name}.test.cpp)
    target_link_libraries(${target_name} PUBLIC
            nagato
            "-pthread")
    target_include_directories(${target_name} PUBLIC
            ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${target_name} PUBLIC "-pthread")
endfunction()

add_subdirectory(verify)