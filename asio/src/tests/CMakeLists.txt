cmake_minimum_required(VERSION 3.21...3.29)

project(asio_test LANGUAGES CXX)

include(../../cmake/project-is-top-level.cmake OPTIONAL)
include(../../cmake/folders.cmake OPTIONAL)
include(../../cmake/windows-set-path.cmake OPTIONAL)

# ---- Dependencies ----

if(PROJECT_IS_TOP_LEVEL)
  find_package(asio REQUIRED)
  enable_testing()
endif()

# ---- Tests ----

add_executable(asio_test unit/read_until.cpp)

target_link_libraries(asio_test PRIVATE asio::asio)
target_compile_features(asio_test PRIVATE cxx_std_17)

# ---- Setup strict Compiler Warnings ----

include(../../cmake/WarningsAsErrors.cmake OPTIONAL)

add_test(NAME asio_test COMMAND asio_test)

# ---- End-of-file commands ----

if(NOT PROJECT_IS_TOP_LEVEL)
  add_folders(Tests)
endif()
