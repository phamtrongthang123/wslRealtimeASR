cmake_minimum_required(VERSION 3.21)
project(wsl_realtime_asr CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/whisper.cpp whispercpp_build)

add_executable(realtime_server
    src/main.cpp
)

target_include_directories(realtime_server PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/whisper.cpp/examples/server
)

target_link_libraries(realtime_server PRIVATE
    whisper
    Threads::Threads
)

if (WIN32)
    target_link_libraries(realtime_server PRIVATE ws2_32)
endif()
