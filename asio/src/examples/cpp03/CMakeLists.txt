set(target_prefix asio_cpp03_)

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
	http/client/async_client 
	http/client/sync_client 
	http/server/http_server 
	http/server2/http_server 
	http/server3/http_server 
	http/server4/http_server 
	icmp/ping 
	invocation/prioritised_handlers 
	iostreams/daytime_client 
	iostreams/daytime_server 
	iostreams/http_client 
	multicast/receiver 
	multicast/sender 
	nonblocking/third_party_lib 
	porthopper/client 
	porthopper/server 
	services/daytime_client 
	socks4/sync_client 
	timeouts/async_tcp_client 
	timeouts/blocking_tcp_client 
	timeouts/blocking_udp_client 
	timeouts/server 
	timers/time_t_timer 
	tutorial/timer1/timer 
	tutorial/timer2/timer 
	tutorial/timer3/timer 
	tutorial/timer4/timer 
	tutorial/timer5/timer 
	tutorial/daytime1/client 
	tutorial/daytime2/server 
	tutorial/daytime3/server 
	tutorial/daytime4/client 
	tutorial/daytime5/server 
	tutorial/daytime6/server 
	tutorial/daytime7/server
)

if(OpenSSL_FOUND)
  set(noinst_PROGRAMS ${noinst_PROGRAMS}
	ssl/client
	ssl/server
  )
endif()

set(allocation_server_SOURCES allocation/server)
set(buffers_reference_counted_SOURCES buffers/reference_counted)
set(chat_chat_client_SOURCES chat/chat_client)
set(chat_chat_server_SOURCES chat/chat_server)
set(echo_async_tcp_echo_server_SOURCES echo/async_tcp_echo_server)
set(echo_async_udp_echo_server_SOURCES echo/async_udp_echo_server)
set(echo_blocking_tcp_echo_client_SOURCES echo/blocking_tcp_echo_client)
set(echo_blocking_tcp_echo_server_SOURCES echo/blocking_tcp_echo_server)
set(echo_blocking_udp_echo_client_SOURCES echo/blocking_udp_echo_client)
set(echo_blocking_udp_echo_server_SOURCES echo/blocking_udp_echo_server)
set(http_client_async_client_SOURCES http/client/async_client)
set(http_client_sync_client_SOURCES http/client/sync_client)
set(http_server_http_server_SOURCES 
	http/server/connection 
	http/server/connection_manager 
	http/server/main 
	http/server/mime_types 
	http/server/reply 
	http/server/request_handler 
	http/server/request_parser 
	http/server/server)
set(http_server2_http_server_SOURCES 
	http/server2/connection 
	http/server2/io_context_pool 
	http/server2/main 
	http/server2/mime_types 
	http/server2/reply 
	http/server2/request_handler 
	http/server2/request_parser 
	http/server2/server)
set(http_server3_http_server_SOURCES 
	http/server3/connection 
	http/server3/main 
	http/server3/mime_types 
	http/server3/reply 
	http/server3/request_handler 
	http/server3/request_parser 
	http/server3/server)
set(http_server4_http_server_SOURCES 
	http/server4/file_handler 
	http/server4/main 
	http/server4/mime_types 
	http/server4/reply 
	http/server4/request_parser 
	http/server4/server)
set(icmp_ping_SOURCES icmp/ping)
set(invocation_prioritised_handlers_SOURCES invocation/prioritised_handlers)
set(iostreams_daytime_client_SOURCES iostreams/daytime_client)
set(iostreams_daytime_server_SOURCES iostreams/daytime_server)
set(iostreams_http_client_SOURCES iostreams/http_client)
set(multicast_receiver_SOURCES multicast/receiver)
set(multicast_sender_SOURCES multicast/sender)
set(nonblocking_third_party_lib_SOURCES nonblocking/third_party_lib)
set(porthopper_client_SOURCES porthopper/client)
set(porthopper_server_SOURCES porthopper/server)
set(services_daytime_client_SOURCES 
	services/daytime_client 
	services/logger_service)
set(socks4_sync_client_SOURCES socks4/sync_client)
set(timeouts_async_tcp_client_SOURCES timeouts/async_tcp_client)
set(timeouts_blocking_tcp_client_SOURCES timeouts/blocking_tcp_client)
set(timeouts_blocking_udp_client_SOURCES timeouts/blocking_udp_client)
set(timeouts_server_SOURCES timeouts/server)
set(timers_time_t_timer_SOURCES timers/time_t_timer)
set(tutorial_timer1_timer_SOURCES tutorial/timer1/timer)
set(tutorial_timer2_timer_SOURCES tutorial/timer2/timer)
set(tutorial_timer3_timer_SOURCES tutorial/timer3/timer)
set(tutorial_timer4_timer_SOURCES tutorial/timer4/timer)
set(tutorial_timer5_timer_SOURCES tutorial/timer5/timer)
set(tutorial_daytime1_client_SOURCES tutorial/daytime1/client)
set(tutorial_daytime2_server_SOURCES tutorial/daytime2/server)
set(tutorial_daytime3_server_SOURCES tutorial/daytime3/server)
set(tutorial_daytime4_client_SOURCES tutorial/daytime4/client)
set(tutorial_daytime5_server_SOURCES tutorial/daytime5/server)
set(tutorial_daytime6_server_SOURCES tutorial/daytime6/server)
set(tutorial_daytime7_server_SOURCES tutorial/daytime7/server)

set(ssl_client_SOURCES ssl/client)
set(ssl_server_SOURCES ssl/server)


foreach(program ${noinst_PROGRAMS})
  string(REPLACE "/" "_" target ${program})
  add_executable(${target_prefix}${target} ${${target}_SOURCES})
  set_target_properties(${target_prefix}${target} PROPERTIES CXX_STANDARD 98
    #FIXME CXX_STANDARD 03
    #cmake error: CXX_STANDARD is set to invalid value '03'
  )
  target_link_libraries(${target_prefix}${target} asio::standalone)

  if(EXTRA_LIBS)
    target_link_libraries(${target_prefix}${target} ${EXTRA_LIBS})
  endif()
endforeach()

