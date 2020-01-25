set(target_prefix asio_cpp11_)

set(noinst_PROGRAMS
  allocation/server
  buffers/reference_counted
  chat/chat_client
  chat/chat_server
  echo/async_tcp_echo_server
  echo/async_udp_echo_server
  echo/blocking_tcp_echo_client
  echo/blocking_tcp_echo_server
  echo/blocking_udp_echo_client
  echo/blocking_udp_echo_server
  executors/actor
  executors/bank_account_1
  executors/bank_account_2
  executors/fork_join
  executors/pipeline
  executors/priority_scheduler
  fork/daemon
  fork/process_per_connection
  futures/daytime_client
  http/server/http_server
  invocation/prioritised_handlers
  iostreams/http_client
  local/connect_pair
  local/iostream_client
  local/stream_client
  local/stream_server
  multicast/receiver
  multicast/sender
  nonblocking/third_party_lib
  operations/composed_1
  operations/composed_2
  operations/composed_3
  operations/composed_4
  operations/composed_5
  operations/composed_6
  operations/composed_7
  operations/composed_8
  socks4/sync_client
  timeouts/async_tcp_client
  timeouts/blocking_tcp_client
  timeouts/blocking_token_tcp_client
  timeouts/blocking_udp_client
  timeouts/server
  timers/time_t_timer
)

if(Boost_FOUND)
  set(noinst_PROGRAMS ${noinst_PROGRAMS}
    spawn/echo_server
    spawn/parallel_grep
  )
endif()

if(OpenSSL_FOUND)
  set(noinst_PROGRAMS ${noinst_PROGRAMS}
    ssl/client
    ssl/server
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

foreach(program ${noinst_PROGRAMS})
  string(REPLACE "/" "_" target ${program})
  if(NOT DEFINED "${target}_SOURCES")
    set("${target}_SOURCES" ${program})
  endif()
  add_executable(${target_prefix}${target} ${${target}_SOURCES})
  set_target_properties(${target_prefix}${target} PROPERTIES CXX_STANDARD 11)
  target_link_libraries(${target_prefix}${target} asio::standalone)

  if(EXTRA_LIBS)
    target_link_libraries(${target_prefix}${target} ${EXTRA_LIBS})
  endif()
endforeach()
