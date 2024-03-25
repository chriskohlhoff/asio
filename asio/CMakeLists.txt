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
  set(CMAKE_VERIFY_INTERFACE_HEADER_SETS ON)
  include(CheckCXXSymbolExists)
  check_cxx_symbol_exists(snprintf cstdio ASIO_HAS_SNPRINTF)
  check_cxx_symbol_exists(sprintf_s cstdio ASIO_HAS_SECURE_RTL)
endif()

# ---- Declare library ----

file(GLOB_RECURSE headers "include/asio/*.hpp")
list(FILTER headers EXCLUDE REGEX [=[.*/experimental/.*\.hpp]=])
list(FILTER headers EXCLUDE REGEX [=[.*/(detail|ssl|impl)/.*\.hpp]=])
list(FILTER headers EXCLUDE REGEX [=[.*/ssl\.hpp]=])

# foreach(header in LISTS ${headers})
#   message(STATUS "${header}")
# endforeach()

file(GLOB_RECURSE implementation "include/asio/*/*.hpp" "include/asio/*.ipp")

add_library(asio INTERFACE ${implementation})
add_library(asio::asio ALIAS asio)

target_sources(asio INTERFACE FILE_SET HEADERS BASE_DIRS include FILES ${headers})

target_compile_definitions(asio INTERFACE ASIO_NO_DEPRECATED)
if(ASIO_HAS_SECURE_RTL)
  target_compile_definitions(asio INTERFACE ASIO_HAS_SECURE_RTL)
elseif(ASIO_HAS_SNPRINTF)
  target_compile_definitions(asio INTERFACE ASIO_HAS_SNPRINTF)
endif()
target_compile_features(asio INTERFACE cxx_std_17)

# ---- add dependenc libraries ----

find_package(Threads REQUIRED)
target_link_libraries(asio INTERFACE Threads::Threads)

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
