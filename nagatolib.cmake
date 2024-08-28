
# 使用するファイルを探索
file(GLOB_RECURSE nagatolib_source ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
file(GLOB_RECURSE nagatolib_header ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp)

message("## ${CMAKE_CURRENT_LIST_DIR}")

# Add the library
add_library(nagatolib STATIC ${nagatolib_source})
