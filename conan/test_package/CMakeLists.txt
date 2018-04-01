project(PackageTest CXX)
cmake_minimum_required(VERSION 3.1.2)
set(CMAKE_CXX_STANDARD 14)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

add_executable(example example.cpp)
conan_target_link_libraries(example)
# or explicit:
# target_link_libraries(example ${CONAN_LIBS})

# CTest is a testing tool that can be used to test your project.
enable_testing()
add_test(NAME example
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
         COMMAND example)
