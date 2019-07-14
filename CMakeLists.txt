cmake_minimum_required(VERSION 3.13)
project(nagatolib)

set(CMAKE_CXX_STANDARD 17)

# Build Options
option(NAGATO_BUILD_EXAMPLE "Build example" ON)
option(NAGATO_MATH "Use nagato math function" ON)
option(NAGATO_TEST "Build Test" ON)

if (NAGATO_MATH)
    add_definitions(-DNAGATO_MATH)
endif ()

# compile options
set(CMAKE_CXX_FLAGS_DEBUG
        "-O0 -Wall ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE
        "-O3 -Wall ${CMAKE_CXX_FLAGS_RELEASE}")

# Build Nagato Lib
include_directories(${CMAKE_SOURCE_DIR}/src)
file(GLOB NAGATO_HEADER
        src/*.hpp)

file(GLOB NAGATO_SOURCE
        src/*.cpp)

add_library(nagato ${NAGATO_HEADER} ${NAGATO_SOURCE})

function(add_example target_name)
    add_executable(${target_name}
            example/${target_name}.cpp)
    target_link_libraries(${target_name}
            nagato)
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
    add_subdirectory(${CMAKE_SOURCE_DIR}/test/gtest)
    include_directories(${CMAKE_SOURCE_DIR}/test/gtest/googletest/include)
    file(GLOB
            NAGATO_TEST_SOURCE
            ${CMAKE_SOURCE_DIR}/test/*.cpp)
    add_executable(nagato_test
            ${NAGATO_TEST_SOURCE})
    target_link_libraries(nagato_test
            nagato
            gtest)
endif ()