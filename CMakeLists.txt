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


# Build Nagato Lib
include_directories(${CMAKE_SOURCE_DIR}/src)
file(GLOB NAGATO_HEADER
        src/*.h)

file(GLOB NAGATO_SOURCE
        src/*.cpp)

add_library(nagato ${NAGATO_HEADER} ${NAGATO_SOURCE})

# Build Example
if (NAGATO_BUILD_EXAMPLE)
    add_executable(vector example/vector_example.cpp)
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
endif ()