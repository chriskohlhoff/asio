set(target_prefix asio_unit_)

add_library(asio::standalone ALIAS standalone)

set(test_PROGRAMS
  associated_allocator
  associated_executor
  async_result
  awaitable
  basic_datagram_socket
  basic_deadline_timer
  basic_raw_socket
  basic_seq_packet_socket
  basic_serial_port
  basic_signal_set
  basic_socket
  basic_socket_acceptor
  basic_stream_socket
  basic_streambuf
  basic_waitable_timer
  bind_executor
  buffered_read_stream
  buffered_stream
  buffered_write_stream
  buffer
  buffers_iterator
  co_spawn
  completion_condition
  compose
  connect
  coroutine
  deadline_timer
  defer
  detached
  dispatch
  error
  execution_context
  executor
  executor_work_guard
  generic/basic_endpoint
  generic/datagram_protocol
  generic/raw_protocol
  generic/seq_packet_protocol
  generic/stream_protocol
  high_resolution_timer
  io_context
  io_context_strand
  ip/address
  ip/address_v4
  ip/address_v4_iterator
  ip/address_v4_range
  ip/address_v6
  ip/address_v6_iterator
  ip/address_v6_range
  ip/basic_endpoint
  ip/basic_resolver
  ip/basic_resolver_entry
  ip/basic_resolver_iterator
  ip/basic_resolver_query
  ip/host_name
  ip/icmp
  ip/multicast
  ip/network_v4
  ip/network_v6
  ip/resolver_query_base
  ip/tcp
  ip/udp
  ip/unicast
  ip/v6_only
  is_read_buffered
  is_write_buffered
  local/basic_endpoint
  local/connect_pair
  local/datagram_protocol
  local/stream_protocol
  packaged_task
  placeholders
  posix/basic_descriptor
  posix/basic_stream_descriptor
  posix/descriptor
  posix/descriptor_base
  posix/stream_descriptor
  post
  read
  read_at
  read_until
  redirect_error
  serial_port
  serial_port_base
  signal_set
  socket_base
  steady_timer
  strand
  streambuf
  system_context
  system_executor
  system_timer
  this_coro
  thread
  time_traits
  ts/buffer
  ts/executor
  ts/internet
  ts/io_context
  ts/net
  ts/netfwd
  ts/socket
  ts/timer
  use_awaitable
  use_future
  uses_executor
  wait_traits
  windows/basic_object_handle
  windows/basic_overlapped_handle
  windows/basic_random_access_handle
  windows/basic_stream_handle
  windows/object_handle
  windows/overlapped_handle
  windows/overlapped_ptr
  windows/random_access_handle
  windows/stream_handle
  write
  write_at
)


set(test_header
  ../latency/high_res_clock.hpp
  unit_test.hpp
)


if(OpenSSL_FOUND)
  set(more_test_PROGRAMS
    ssl/context_base
    ssl/context
    ssl/error
    ssl/rfc2818_verification
    ssl/stream_base
    ssl/stream
  )
endif()


### enable_testing() with ctest
foreach(program ${test_PROGRAMS} ${more_test_PROGRAMS})
  string(REPLACE "/" "_" target ${program})
  add_executable(${target_prefix}${target} ${program}.cpp ${test_header})
  set_target_properties(${target_prefix}${target} PROPERTIES CXX_STANDARD 17)

  target_link_libraries(${target_prefix}${target} asio::standalone)
  #NOTE: same as: conan_target_link_libraries(${target_prefix}${target})
  if(EXTRA_LIBS)
    target_link_libraries(${target_prefix}${target} ${EXTRA_LIBS})
  endif()

  add_test(NAME ${target_prefix}${target} COMMAND "${target_prefix}${target}"
  )
endforeach()

