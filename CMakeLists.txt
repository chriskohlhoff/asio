cmake_minimum_required(VERSION 2.8)
project(asio)

add_definitions(-DASIO_STANDALONE)
add_definitions(-DASIO_SEPARATE_COMPILATION)

if (BUILD_SHARED_LIBS)
    add_library(${PROJECT_NAME} SHARED asio/src/asio.cpp)
else()
    add_library(${PROJECT_NAME} STATIC asio/src/asio.cpp)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC asio/include)
