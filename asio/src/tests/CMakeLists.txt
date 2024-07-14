cmake_minimum_required(VERSION 3.23...3.29)

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

if(TARGET asio::asio)
  target_link_libraries(asio_test PRIVATE asio::asio)
else()
  target_link_libraries(asio_test PRIVATE asio::asio_header)
endif()

# ---- Setup strict Compiler Warnings ----

add_test(NAME asio_test COMMAND asio_test)

# ---- End-of-file commands ----

if(NOT PROJECT_IS_TOP_LEVEL)
  add_folders(Tests)
endif()
