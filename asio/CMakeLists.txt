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
  option(CMAKE_VERIFY_INTERFACE_HEADER_SETS "FIXME: some header in include/asio/detail fails!" OFF)
  include(CheckCXXSymbolExists)
  check_cxx_symbol_exists(snprintf cstdio ASIO_HAS_SNPRINTF)
  check_cxx_symbol_exists(sprintf_s cstdio ASIO_HAS_SECURE_RTL)
endif()

# ---- Declare library ----

file(GLOB_RECURSE headers "include/asio/*.hpp")
list(FILTER headers EXCLUDE REGEX [=[.*/experimental/.*\.hpp]=])
list(FILTER headers EXCLUDE REGEX [=[.*/ssl\.hpp]=])

set(public_headers ${headers})
list(FILTER public_headers EXCLUDE REGEX [=[.*/impl/.*\.hpp]=])

# foreach(header in LISTS ${public_headers})
#   message(INFO "${header}")
# endforeach()

file(GLOB_RECURSE implementation "include/asio/*/*.hpp" "include/asio/*.ipp")
list(FILTER implementation EXCLUDE REGEX [=[.*/experimental/.*]=])

add_library(asio_header INTERFACE ${implementation})
add_library(asio::asio_header ALIAS asio_header)

target_sources(asio_header INTERFACE FILE_SET HEADERS BASE_DIRS include FILES ${implementation})

set(CPPdefinitions ASIO_NO_DEPRECATED)
if(ASIO_HAS_SECURE_RTL)
  list(APPEND CPPdefinitions ASIO_HAS_SECURE_RTL)
elseif(ASIO_HAS_SNPRINTF)
  list(APPEND CPPdefinitions ASIO_HAS_SNPRINTF)
endif()
target_compile_definitions(asio_header INTERFACE ${CPPdefinitions})
target_compile_features(asio_header INTERFACE cxx_std_17)

# ---- add dependenc libraries ----

find_package(Threads REQUIRED)
target_link_libraries(asio_header INTERFACE Threads::Threads)

option(ASIO_SEPARATE_COMPILATION "build asio lib too" ${PROJECT_IS_TOP_LEVEL})
if(ASIO_SEPARATE_COMPILATION)
  find_package(OpenSSL)
  set(libasio_SOURCES src/asio.cpp)
  if(OpenSSL_FOUND)
    set(libasio_SOURCES ${libasio_SOURCES} src/asio_ssl.cpp)
  endif()

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
