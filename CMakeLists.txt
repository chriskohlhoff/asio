cmake_minimum_required(VERSION 3.15)

project(learn_asio CXX)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall)

include_directories(asio/include)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

add_definitions(-DASIO_ENABLE_HANDLER_TRACKING)
file(GLOB_RECURSE CPP_SRC_LIST ${CMAKE_CURRENT_LIST_DIR}/asio/src/examples/*.cpp)
foreach (file ${CPP_SRC_LIST})
    string(REGEX REPLACE ".*examples|/" "." target_name ${file})
    string(REGEX REPLACE "^\\." "" target_name ${target_name})
    # message("${target_name}")
    add_executable(${target_name} ${file})
endforeach ()
