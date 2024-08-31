# --------------------------------------------------------------------------------------------
# nagatolib.cmake の使い方
# ライブラリをビルドするための cmake ファイル
# このファイルはライブラリを呼び出すことができる
# 呼び出し側で
# include(../nagatolib/nagatolib.cmake)
# を実行するでstatic library としてビルドされる
# また、Metal を使用する場合は
# option(NAGATO_METAL "Use Metal" ON) (デフォルトは ON)
# を設定することで Metal を使用したライブラリをビルドすることができる
# ヘッダーとして ${nagatolib_include_dir} を include することで
# nagatolib のヘッダーを使用することができる
# ライブラリとして ${nagatolib} をリンクすることで nagatolib を使用することができる
# --------------------------------------------------------------------------------------------


# 使用するファイルを探索
file(GLOB_RECURSE nagatolib_source ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)


# include するディレクトリ
set(nagatolib_include_dir
        ${CMAKE_CURRENT_LIST_DIR}/src
        )

message("## ${CMAKE_CURRENT_LIST_DIR}")

# mac の場合は Metal を使用したライブラリを有効にする
if (APPLE)
    message("## APPLE detected")
    option(NAGATO_METAL "Use Metal" ON)
    if (NAGATO_METAL)
        message("## Enable Metal Library")

        add_definitions(-DNAGATO_METAL)

        # metal_kernel ディレクトリごとビルド先にコピーする
        file(GLOB METAL_KERNEL
                ${CMAKE_CURRENT_LIST_DIR}/metal_kernel/*.metal)
        foreach (file ${METAL_KERNEL})
            file(COPY ${file} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/metal_kernel)
        endforeach ()

        # Metal を使用するソース
        file(GLOB METAL_SOURCES
                ${CMAKE_CURRENT_LIST_DIR}/Metal/*.cpp)

        # metal 関連のソース, ヘッダーを追加
        list(APPEND nagatolib_source ${METAL_SOURCES})

        # metal 関連のヘッダーを追加
        list(APPEND nagatolib_include_dir
                ${CMAKE_CURRENT_LIST_DIR}/Metal
                )
    endif (NAGATO_METAL)
endif (APPLE)


# Add the library
add_library(nagatolib STATIC
        ${nagatolib_source}
)

if (NAGATO_METAL)
    target_link_libraries(
            nagatolib
            "-framework Cocoa"
            "-framework Foundation"
            "-framework Metal"
            "-framework QuartzCore"
    )
endif (NAGATO_METAL)
