cmake_minimum_required(VERSION 3.13)
project(nagatolib)

set(CMAKE_CXX_STANDARD 17)

option(NAGATO_BUILD_EXAMPLE "Build example" ON)
option(NAGATO_MATH "Use nagato math function" ON)
if(NAGATO_MATH)
    add_definitions(-DNAGATO_MATH)
endif()


include_directories(${CMAKE_SOURCE_DIR}/src)
file(GLOB NAGATO_HEADER
        src/*.h)

file(GLOB NAGATO_SOURCE
        src/*.cpp)

add_library(nagato ${NAGATO_HEADER} ${NAGATO_SOURCE})

if (NAGATO_BUILD_EXAMPLE)
    add_executable(vector example/vector_example.cpp)
endif ()