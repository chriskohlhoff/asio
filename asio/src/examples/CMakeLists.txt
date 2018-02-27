# Minimum supported CMake version
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

# =================================================
project(asio_examples VERSION 1.11.0 LANGUAGES CXX)
# =================================================

# Options
set(ASIO_BUILD_EXAMPLES_CPP11 ON CACHE BOOL "Build asio C++11 examples")
set(ASIO_BUILD_EXAMPLES_CPP14 ON CACHE BOOL "Build asio C++14 examples")

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

#NOTE: examples and tutorial needs boost date_time! CK
if(Boost_FOUND)
  add_subdirectory(cpp03)
else()
  message(WARNING "examples/cpp03 needs boost::date_time!")
endif()

# C++11 only examples
if(ASIO_BUILD_EXAMPLES_CPP11)
  add_subdirectory(cpp11)
endif()

# C++14 only examples
if(ASIO_BUILD_EXAMPLES_CPP14)
  add_subdirectory(cpp14)
endif()
