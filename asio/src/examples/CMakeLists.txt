# Minimum supported CMake version
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# =================================================
project(asio_examples VERSION 1.13.0 LANGUAGES CXX)
# =================================================

# Options
set(ASIO_BUILD_EXAMPLES_CPP11 ON CACHE BOOL "Build asio C++11 examples")
set(ASIO_BUILD_EXAMPLES_CPP14 ON CACHE BOOL "Build asio C++14 examples")
if(NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set(ASIO_BUILD_EXAMPLES_CPP17 ON CACHE BOOL "Build asio C++17 examples")
endif()

# Check asio targets
if(TARGET standalone)
  add_library(asio::standalone ALIAS standalone)
  set(ASIO_FOUND TRUE)
endif()

# Find asio package
if(NOT ASIO_FOUND)
  find_package(asio CONFIG REQUIRED)
endif()

# Installation configuration
include(GNUInstallDirs)

if(Boost_FOUND)
  add_subdirectory(cpp03)
else()
  message(WARNING "examples/cpp03 needs boost!")
endif()

# C++11 only examples
if(ASIO_BUILD_EXAMPLES_CPP11)
  add_subdirectory(cpp11)
endif()

# C++14 only examples
if(ASIO_BUILD_EXAMPLES_CPP14)
  add_subdirectory(cpp14)
endif()

# C++17 only examples
#NOTE: the examples needs COROUTINE! CK
if(ASIO_BUILD_EXAMPLES_CPP17)
  add_subdirectory(cpp17)
endif()
