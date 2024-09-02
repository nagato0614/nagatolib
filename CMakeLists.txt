cmake_minimum_required(VERSION 3.25)
project(nagatolib)

set(CMAKE_CXX_STANDARD 23)


# Build Options
option(NAGATO_BUILD_EXAMPLE "Build example" ON)
option(NAGATO_MATH "Use nagato math function" ON)
option(NAGATO_TEST "Build Test" ON)
option(NAGATO_METAL "Use Metal" ON)

if (NAGATO_MATH)
    add_definitions(-DNAGATO_MATH)
endif ()

# mac の場合は Metal を使用したライブラリを有効にする
if (APPLE)
    message(STATUS "## Apple System")

    option(NAGATO_METAL "Use Metal" ON)
    if (NAGATO_METAL)
        message(STATUS "## Enable Metal Library")

        add_definitions(-DNAGATO_METAL)

        # metal_kernel ディレクトリごとビルド先にコピーする
        file(GLOB METAL_KERNEL
                metal_kernel/*.metal)
        foreach (file ${METAL_KERNEL})
            file(COPY ${file} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/metal_kernel)
        endforeach ()

        # Metal を使用するソース
        file(GLOB METAL_SOURCES
                Metal/*.cpp)

        # example をビルド
        if (NAGATO_BUILD_EXAMPLE)
            add_executable(
                    metal_example
                    example/metal_example.cpp
                    ${METAL_SOURCES}
            )
            target_include_directories(
                    metal_example
                    PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/Metal
            )
            target_link_libraries(
                    metal_example
                    "-framework Cocoa"
                    "-framework Foundation"
                    "-framework Metal"
                    "-framework QuartzCore"
            )
        endif ()
    endif ()
endif ()

# compile options
set(CMAKE_CXX_FLAGS_DEBUG
        "-O0 -Wall ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE
        "-O2 -Wall ${CMAKE_CXX_FLAGS_RELEASE}")

# Build Nagato Lib
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
file(GLOB NAGATO_HEADER
        src/*.hpp)

file(GLOB NAGATO_SOURCE
        src/*.cpp)

# metal が有効な場合 NAGATO_SOURCE に Metal のソースを追加する
if (NAGATO_METAL)
    list(APPEND NAGATO_SOURCE ${METAL_SOURCES})
endif ()

add_library(nagato ${NAGATO_HEADER} ${NAGATO_SOURCE})

function(add_example target_name)
    add_executable(${target_name}
            example/${target_name}.cpp)
    target_link_libraries(${target_name}
            nagato
    )
endfunction()

# Build Example
if (NAGATO_BUILD_EXAMPLE)
    add_example(vector_example)
    add_example(random_example)
    add_example(thread_example)
    add_example(matrix_example)
    add_example(narray_example)
endif ()

# Build Tests
if (NAGATO_TEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test/gtest)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test/gtest/googletest/include)
    file(GLOB
            NAGATO_TEST_SOURCE
            ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp)
    add_executable(nagato_test
            ${NAGATO_TEST_SOURCE}
    )
    target_link_libraries(nagato_test
            nagato
            gtest
    )

    # metal が有効な場合
    if (NAGATO_METAL)
        target_link_libraries(nagato_test
                "-framework Cocoa"
                "-framework Foundation"
                "-framework Metal"
                "-framework QuartzCore"
        )
        target_include_directories(
                nagato_test
                PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/Metal
        )
    endif ()
endif ()