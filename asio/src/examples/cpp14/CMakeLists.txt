set(target_prefix asio_cpp14_)

set(noinst_PROGRAMS
  echo/async_tcp_echo_server
  echo/async_udp_echo_server
  echo/blocking_tcp_echo_client
  echo/blocking_tcp_echo_server
  echo/blocking_udp_echo_client
  echo/blocking_udp_echo_server
  executors/actor
  executors/async_1
  executors/async_2
  executors/bank_account_1
  executors/bank_account_2
  executors/fork_join
  executors/pipeline
  executors/priority_scheduler
  iostreams/http_client)

foreach(program ${noinst_PROGRAMS})
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

#TODO: target_compile_features(${target_prefix}executors_priority_scheduler PRIVATE cxx_std_14)
