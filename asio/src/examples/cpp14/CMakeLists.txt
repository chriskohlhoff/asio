set(target_prefix asio_cpp14_)

set(noinst_PROGRAMS
  echo/async_tcp_echo_server.cpp
  echo/async_udp_echo_server.cpp
  echo/blocking_tcp_echo_client.cpp
  echo/blocking_tcp_echo_server.cpp
  echo/blocking_udp_echo_client.cpp
  echo/blocking_udp_echo_server.cpp
  executors/actor.cpp
  executors/async_1.cpp
  executors/async_2.cpp
  executors/bank_account_1.cpp
  executors/bank_account_2.cpp
  executors/fork_join.cpp
  executors/pipeline.cpp
  executors/priority_scheduler.cpp
  iostreams/http_client.cpp
)

foreach(source ${noinst_PROGRAMS})
  string(REGEX REPLACE "\.cpp$" "" program ${source})
  string(REPLACE "/" "_" target ${program})
  add_executable(${target_prefix}${target} ${program}.cpp)
  set_target_properties(${target_prefix}${target} PROPERTIES CXX_STANDARD 14)

  target_link_libraries(${target_prefix}${target} asio::standalone)
  if(EXTRA_LIBS)
    target_link_libraries(${target_prefix}${target} ${EXTRA_LIBS})
  endif()

  # install(TARGETS ${target_prefix}${target}
  #  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()

