cmake_minimum_required(VERSION 3.0.0)
project(asio_examples VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)

find_package(asio REQUIRED)
find_package(Threads REQUIRED)

set(fileList "")
file(GLOB fileList *.cpp)
foreach(file ${fileList})
	set(fileName "")
	get_filename_component(fileName ${file} NAME_WLE)
    
	add_executable(${fileName} ${file})
	target_link_libraries(${fileName} PRIVATE asio Threads::Threads)
endforeach()
