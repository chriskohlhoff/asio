set(target_prefix asio_cpp17_)

set(noinst_PROGRAMS
  coroutines_ts/chat_server.cpp
  coroutines_ts/echo_server.cpp
  coroutines_ts/range_based_for.cpp
  coroutines_ts/refactored_echo_server.cpp
)

foreach(source ${noinst_PROGRAMS})
  string(REGEX REPLACE "\.cpp$" "" program ${source})
  string(REPLACE "/" "_" target ${program})
  add_executable(${target_prefix}${target} ${program}.cpp)
  set_target_properties(${target_prefix}${target} PROPERTIES CXX_STANDARD 17)

  target_link_libraries(${target_prefix}${target} asio::standalone)
  if(EXTRA_LIBS)
    target_link_libraries(${target_prefix}${target} ${EXTRA_LIBS})
  endif()

  # install(TARGETS ${target_prefix}${target}
  #  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()

