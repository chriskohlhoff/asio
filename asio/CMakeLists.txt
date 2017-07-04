# Minimum supported CMake version
cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

# use ccache if found
find_program(CCACHE_EXECUTABLE "ccache" HINTS /usr/local/bin /opt/local/bin)
if(CCACHE_EXECUTABLE AND NOT CMAKE_TOOLCHAIN_FILE)
  message(STATUS "use ccache")
  find_program(GXX_EXECUTABLE "g++" HINTS /usr/local/bin /opt/local/bin)
  if(GXX_EXECUTABLE)
    set(CMAKE_COMPILER_IS_GNUCXX yes)
    set(CMAKE_CXX_COMPILER "${CCACHE_EXECUTABLE}" CACHE PATH "ccache" FORCE)
    set(CMAKE_CXX_COMPILER_ARG1 ${GXX_EXECUTABLE} CACHE PATH "the real g++" FORCE)
  endif()
endif()

# ========================================
project(asio VERSION 1.11.0 LANGUAGES CXX)
# ========================================

# Options
set(asio_STANDALONE ON CACHE BOOL "Build asio without boost")
set(asio_BUILD_EXAMPLES OFF CACHE BOOL "Build asio examples")


# enable ctest
enable_testing()

#
# Where to put all the RUNTIME targets when built.  This variable is used to
# initialize the RUNTIME_OUTPUT_DIRECTORY property on all the targets.
#
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)


if(WIN32)
  add_definitions(-D_WIN32_WINNT=0x0501 -D__USE_W32_SOCKETS -DASIO_HAS_VARIADIC_TEMPLATES -DASIO_DISABLE_CONSTEXPR)
  set(EXTRA_LIBS wsock32 ws2_32)
else()
  set(EXTRA_LIBS)
endif()

if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "MINGW.*")
  add_definitions(-D_WIN32_WINNT=0x0501 -D__USE_W32_SOCKETS)
endif()


if(NOT asio_STANDALONE)
  #######################################################################
  # use boost if found
  option(Boost_USE_MULTITHREADED "Set to OFF to use the non-multithreaded" ON)
  option(Boost_DEBUG "Set to ON to enable debug output from FindBoost." OFF)
  option(Boost_DETAILED_FAILURE_MSG "Set to ON to get detailed information" ON)
  # Set Boost_NO_BOOST_CMAKE to ON to disable the search for boost-cmake.
  set(Boost_NO_BOOST_CMAKE CACHE FLAG ON FORCE)
  set(BOOST_ROOT "/opt/local" CACHE FILEPATH "path to the boost lib")
  set(Boost_USE_STATIC_LIBS OFF)
  if(WIN32)
    set(Boost_USE_STATIC_RUNTIME OFF)
  endif()
  find_package(Boost 1.58 COMPONENTS atomic chrono coroutine context
    date_time thread system)
  #######################################################################

  if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
    add_definitions(-DBOOST_ALL_NO_LIB=1
                    -DASIO_HAS_BOOST_DATE_TIME=1
    )
    set(EXTRA_LIBS ${EXTRA_LIBS} ${Boost_LIBRARIES} )
  else()
    if(CMAKE_COMPILER_IS_GNUCXX OR __COMPILER_CLANG)
      set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -Wextra)
    endif()
  endif()
  #######################################################################
endif()


#######################################################################
# use openssl if found
set(OpenSSL_ROOT_DIR "/opt/local" CACHE FILEPATH "path to the openssl lib")
find_package(OpenSSL QUIET NO_MODULE HINTS ${OpenSSL_ROOT_DIR})
#######################################################################

if(OpenSSL_FOUND)
  set(EXTRA_LIBS ${EXTRA_LIBS} ${openSSL_LIBRARIES})
  add_definitions(-DHAVE_OPENSSL)
  include_directories(BEFORE ${openSSL_INCLUDE_DIR})
else()
  find_library(OpenSSL_LIBRARY ssl HINTS ${OpenSSL_ROOT_DIR}/lib )
  find_library(CRYPTO_LIBRARY crypto HINTS ${OpenSSL_ROOT_DIR}/lib )
  if(OpenSSL_LIBRARY AND CRYPTO_LIBRARY )
    set(OpenSSL_FOUND ON)
    add_definitions(-DHAVE_OPENSSL)
    set(EXTRA_LIBS ${EXTRA_LIBS} ${OpenSSL_LIBRARY} ${CRYPTO_LIBRARY})
    include_directories(BEFORE ${OpenSSL_ROOT_DIR}/include)
  endif()
endif()
#######################################################################


# Installation configuration
include(GNUInstallDirs)
if(NOT DEFINED CMAKE_INSTALL_CMAKEDIR)
  set(CMAKE_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/asio)
endif()

# Interface library
add_subdirectory(include)

# Sources: examples, tests
add_subdirectory(src)

# Installation
configure_file(asio-config-version.cmake.in asio-config-version.cmake @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/asio-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_CMAKEDIR})
install(EXPORT asio-targets
  DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
  NAMESPACE asio::)
install(FILES asio-config.cmake DESTINATION ${CMAKE_INSTALL_CMAKEDIR})

