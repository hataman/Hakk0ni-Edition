cmake_minimum_required(VERSION 3.20)
project(Hakk0niEdition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# WIN32 = GUI subsystem: no black console window.
add_executable(Hakk0niEdition WIN32
    src/main.cpp
    resources/speechhelper.rc
)

target_include_directories(Hakk0niEdition PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_library(
    SHERPA_ONNX_LIB
    NAMES sherpa-onnx-c-api sherpa-onnx-c-api.lib
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    NO_DEFAULT_PATH
)

if (NOT SHERPA_ONNX_LIB)
    message(FATAL_ERROR "sherpa-onnx C API import library not found in runtime/")
endif()

target_link_libraries(Hakk0niEdition PRIVATE
    winmm
    comctl32
    ${SHERPA_ONNX_LIB}
)

if (MSVC)
    target_compile_options(Hakk0niEdition PRIVATE /W4 /O2)
endif()
