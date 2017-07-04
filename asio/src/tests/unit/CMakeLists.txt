set(target_prefix asio_unit_)

add_library(asio::standalone ALIAS standalone)

set(null_test_PROGRAMS
  associated_allocator
  associated_executor
  async_result
  basic_datagram_socket
  basic_deadline_timer
  basic_raw_socket
  basic_seq_packet_socket
  basic_serial_port
  basic_signal_set
  basic_socket_acceptor
  basic_stream_socket
  basic_streambuf
  basic_waitable_timer
  bind_executor
  completion_condition
  connect
  datagram_socket_service
  deadline_timer
  deadline_timer_service
  defer
  dispatch
  execution_context
  executor
  executor_work_guard
  high_resolution_timer
  ip/address_v4_iterator
  ip/address_v4_range
  ip/address_v6_iterator
  ip/address_v6_range
  ip/basic_endpoint
  ip/basic_resolver
  ip/basic_resolver_entry
  ip/basic_resolver_iterator
  ip/basic_resolver_query
  ip/resolver_query_base
  ip/resolver_service
  local/basic_endpoint
  packaged_task
  placeholders
  posix/basic_descriptor
  posix/basic_stream_descriptor
  posix/descriptor
  posix/descriptor_base
  posix/stream_descriptor_service
  post
  raw_socket_service
  seq_packet_socket_service
  serial_port_service
  signal_set_service
  socket_acceptor_service
  steady_timer
  stream_socket_service
  thread
  time_traits
  uses_executor
  wait_traits
  waitable_timer_service
  windows/basic_handle
  windows/basic_object_handle
  windows/basic_random_access_handle
  windows/basic_stream_handle
  windows/object_handle_service
  windows/overlapped_handle
  windows/random_access_handle_service
  windows/stream_handle_service
)

set(noinst_PROGRAMS
  buffer
  buffered_read_stream
  buffered_stream
  buffered_write_stream
  buffers_iterator
  coroutine
  error
  generic/datagram_protocol
  generic/raw_protocol
  generic/seq_packet_protocol
  generic/stream_protocol
  io_context
  ip/address
  ip/address_v4
  ip/address_v6
  ip/host_name
  ip/icmp
  ip/multicast
  ip/network_v4
  ip/network_v6
  ip/tcp
  ip/udp
  ip/unicast
  ip/v6_only
  is_read_buffered
  is_write_buffered
  local/connect_pair
  local/datagram_protocol
  local/stream_protocol
  posix/stream_descriptor
  read
  read_at
  read_until
  serial_port
  serial_port_base
  signal_set
  socket_base
  strand
  streambuf
  system_timer
  use_future
  windows/object_handle
  windows/overlapped_ptr
  windows/random_access_handle
  windows/stream_handle
  write
  write_at
)

if(OpenSSL_FOUND)
  set(null_test_PROGRAMS ${null_test_PROGRAMS}
    ssl/context
    ssl/context_base
    ssl/error
    ssl/rfc2818_verification
    ssl/stream_base
  )
  set(noinst_PROGRAMS ${noinst_PROGRAMS}
    ssl/stream
  )
endif()

foreach(program ${noinst_PROGRAMS} ${null_test_PROGRAMS})
  string(REPLACE "/" "_" target ${program})
  add_executable(${target_prefix}${target} ${program}.cpp unit_test.hpp)
  set_target_properties(${target_prefix}${target} PROPERTIES CXX_STANDARD 11)

  target_link_libraries(${target_prefix}${target} asio::standalone)
  if(EXTRA_LIBS)
    target_link_libraries(${target_prefix}${target} ${EXTRA_LIBS})
  endif()
endforeach()

if(MSVC)
  target_compile_features(${target_prefix}use_future PRIVATE
    cxx_std_11  # OK
    cxx_contextual_conversions  # OK
    cxx_final   # OK
    cxx_override    # OK
    cxx_variadic_templates  # OK
    #FIXME: missing features at MS VS12 2013:
    cxx_aggregate_default_initializers
    cxx_attribute_deprecated
    cxx_binary_literals
    cxx_constexpr
    cxx_decltype_auto
    cxx_decltype_incomplete_return_types
    cxx_defaulted_move_initializers
    cxx_digit_separators
    cxx_generic_lambdas
    cxx_lambda_init_captures
    cxx_noexcept
    cxx_relaxed_constexpr
    cxx_return_type_deduction
    cxx_variable_templates
  )
endif()

### enable_testing() with ctest
foreach(program ${noinst_PROGRAMS})
  string(REPLACE "/" "_" target ${program})
  add_test(NAME ${target_prefix}${target}
    COMMAND "${target_prefix}${target}"
  )
endforeach()

