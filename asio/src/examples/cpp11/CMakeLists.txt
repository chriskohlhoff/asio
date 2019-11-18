set(target_prefix asio_cpp11_)

set(noinst_PROGRAMS
  allocation/server.cpp
  buffers/reference_counted.cpp
  chat/chat_client.cpp
  chat/chat_server.cpp
  echo/async_tcp_echo_server.cpp
  echo/async_udp_echo_server.cpp
  echo/blocking_tcp_echo_client.cpp
  echo/blocking_tcp_echo_server.cpp
  echo/blocking_udp_echo_client.cpp
  echo/blocking_udp_echo_server.cpp
  executors/actor.cpp
  executors/bank_account_1.cpp
  executors/bank_account_2.cpp
  executors/fork_join.cpp
  executors/pipeline.cpp
  executors/priority_scheduler.cpp
  fork/daemon.cpp
  fork/process_per_connection.cpp
  futures/daytime_client.cpp
  invocation/prioritised_handlers.cpp
  iostreams/http_client.cpp
  local/connect_pair.cpp
  local/iostream_client.cpp
  local/stream_client.cpp
  local/stream_server.cpp
  multicast/receiver.cpp
  multicast/sender.cpp
  nonblocking/third_party_lib.cpp
  operations/composed_1.cpp
  operations/composed_2.cpp
  operations/composed_3.cpp
  operations/composed_4.cpp
  operations/composed_5.cpp
  operations/composed_6.cpp
  operations/composed_7.cpp
  operations/composed_8.cpp
  socks4/sync_client.cpp
  ssl/client.cpp
  ssl/server.cpp
  timeouts/async_tcp_client.cpp
  timeouts/blocking_tcp_client.cpp
  timeouts/blocking_token_tcp_client.cpp
  timeouts/blocking_udp_client.cpp
  timeouts/server.cpp
  timers/time_t_timer.cpp
)

if(Boost_FOUND)
  set(noinst_PROGRAMS ${noinst_PROGRAMS}
    spawn/echo_server.cpp
    spawn/parallel_grep.cpp
  )
endif()

set(http_server_http_server_SOURCES
  http/server/connection.cpp
  http/server/connection_manager.cpp
  http/server/main.cpp
  http/server/mime_types.cpp
  http/server/reply.cpp
  http/server/request_handler.cpp
  http/server/request_parser.cpp
  http/server/server.cpp
)

foreach(source ${noinst_PROGRAMS})
  string(REGEX REPLACE "\.cpp$" "" program ${source} )
  string(REPLACE "/" "_" target ${program})
  add_executable(${target_prefix}${target} ${source} ${${target}_SOURCES})
  set_target_properties(${target_prefix}${target} PROPERTIES CXX_STANDARD 11)
  target_link_libraries(${target_prefix}${target} asio::standalone)

  if(EXTRA_LIBS)
    target_link_libraries(${target_prefix}${target} ${EXTRA_LIBS})
  endif()
endforeach()
