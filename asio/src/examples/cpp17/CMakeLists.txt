set(target_prefix asio_cpp17_)

set(noinst_PROGRAMS
  coroutines_ts/chat_server.cpp
  coroutines_ts/echo_server.cpp
  coroutines_ts/range_based_for.cpp
  coroutines_ts/refactored_echo_server.cpp
  coroutines_ts/echo_server_with_default.cpp
)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fcoroutines-ts CXX_COMPILER_HAS_COROUTINES)

if(CXX_COMPILER_HAS_COROUTINES)
  find_library(LIB_CPP_EXPERIMENTAL c++)
  foreach(source ${noinst_PROGRAMS})
    string(REGEX REPLACE "\.cpp$" "" program ${source})
    string(REPLACE "/" "_" target ${program})

    add_executable(${target_prefix}${target} ${program}.cpp)
    set_target_properties(${target_prefix}${target} PROPERTIES CXX_STANDARD 17)
    target_compile_options(${target_prefix}${target} PRIVATE -fcoroutines-ts -stdlib=libc++)
    target_link_libraries(${target_prefix}${target} asio::standalone ${LIB_CPP_EXPERIMENTAL})

    if(EXTRA_LIBS)
      target_link_libraries(${target_prefix}${target} ${EXTRA_LIBS})
    endif()

    # install(TARGETS ${target_prefix}${target}
    #  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endforeach()
endif()

