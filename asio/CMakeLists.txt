cmake_minimum_required(VERSION 3.23...3.29)

include(cmake/prelude.cmake)

project(asio
        VERSION 1.29.0
        DESCRIPTION "Asio C++ Library"
        HOMEPAGE_URL "http://think-async.com/Asio/"
        LANGUAGES CXX
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

if(PROJECT_IS_TOP_LEVEL)
  include(CheckCXXSymbolExists)
  check_cxx_symbol_exists(snprintf cstdio ASIO_HAS_SNPRINTF)
  check_cxx_symbol_exists(sprintf_s cstdio ASIO_HAS_SECURE_RTL)
endif()

set(CPPdefinitions ASIO_NO_DEPRECATED)
if(ASIO_HAS_SECURE_RTL)
  list(APPEND CPPdefinitions ASIO_HAS_SECURE_RTL)
elseif(ASIO_HAS_SNPRINTF)
  list(APPEND CPPdefinitions ASIO_HAS_SNPRINTF)
endif()

option(ASIO_SEPARATE_COMPILATION "build asio lib too" ${PROJECT_IS_TOP_LEVEL})

# ---- add dependency libraries ----

find_package(Threads REQUIRED)

# ---- Declare library ----

file(GLOB_RECURSE implementation "include/asio/*/*.hpp" "include/asio/*.ipp")
list(FILTER implementation EXCLUDE REGEX [=[.*/experimental/.*]=])

add_library(asio_header INTERFACE)
add_library(asio::asio_header ALIAS asio_header)
target_sources(asio_header INTERFACE FILE_SET HEADERS BASE_DIRS include FILES ${implementation})

target_compile_definitions(asio_header INTERFACE ${CPPdefinitions})
target_compile_features(asio_header INTERFACE cxx_std_17)
target_link_libraries(asio_header INTERFACE Threads::Threads)

if(ASIO_SEPARATE_COMPILATION)
  #XXX find_package(OpenSSL)
  set(libasio_SOURCES src/asio.cpp)

  if(OpenSSL_FOUND)
    #XXX set(libasio_SOURCES ${libasio_SOURCES} src/asio_ssl.cpp)
  endif()

  set(CMAKE_VERIFY_INTERFACE_HEADER_SETS ${PROJECT_IS_TOP_LEVEL})

  file(GLOB_RECURSE headers "include/asio/*.hpp")
  list(FILTER headers EXCLUDE REGEX [=[.*/experimental/.*\.hpp]=])
  list(FILTER headers EXCLUDE REGEX [=[.*/ssl\.hpp]=])
  list(FILTER headers EXCLUDE REGEX [=[.*/ssl/\.*]=])

  # FIXME: some header in include/asio/detail fails CMAKE_VERIFY_INTERFACE_HEADER_SETS!
  # NOTE: we need them installed, but we used them install with asio_header INTERFACE! CK
  set(public_headers ${headers})
  list(FILTER public_headers EXCLUDE REGEX [=[.*/detail/.*\.hpp]=])
  list(FILTER public_headers EXCLUDE REGEX [=[.*/impl/.*\.hpp]=])
  list(FILTER public_headers EXCLUDE REGEX [=[.*/ssl/\.*]=])
  # foreach(header in LISTS ${public_headers})
  #   message(INFO "${header}")
  # endforeach()

  add_library(asio ${libasio_SOURCES})
  add_library(asio::asio ALIAS asio)
  target_sources(asio PUBLIC FILE_SET public_headers TYPE HEADERS BASE_DIRS include FILES ${public_headers})

  target_compile_definitions(asio PUBLIC ASIO_SEPARATE_COMPILATION ${CPPdefinitions})
  target_link_libraries(asio PUBLIC Threads::Threads)
  target_compile_features(asio INTERFACE cxx_std_17)

  if(OpenSSL_FOUND)
    target_link_libraries(asio PUBLIC OpenSSL::SSL)
  endif()
endif()

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
  configure_file(asio.pc.cmake asio.pc)
  install(FILES ${PROJECT_BINARY_DIR}/asio.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

# ---- Developer mode ----

if(NOT asio_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(AUTHOR_WARNING "Developer mode is intended for developers of asio")
endif()

include(cmake/dev-mode.cmake)
