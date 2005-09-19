AUTOMAKE_OPTIONS = subdir-objects

SUBDIRS = doc

noinst_PROGRAMS = \
	tests/performance/client \
	tests/performance/server \
	tests/unit/basic_datagram_socket_test \
	tests/unit/basic_deadline_timer_test \
	tests/unit/basic_demuxer_test \
	tests/unit/basic_locking_dispatcher_test \
	tests/unit/basic_socket_acceptor_test \
	tests/unit/basic_stream_socket_test \
	tests/unit/buffered_read_stream_test \
	tests/unit/buffered_stream_test \
	tests/unit/buffered_write_stream_test \
	tests/unit/buffers_test \
	tests/unit/buffer_test \
	tests/unit/consuming_buffers_test \
	tests/unit/datagram_socket_service_test \
	tests/unit/datagram_socket_test \
	tests/unit/deadline_timer_service_test \
	tests/unit/deadline_timer_test \
	tests/unit/default_error_handler_test \
	tests/unit/demuxer_service_test \
	tests/unit/demuxer_test \
	tests/unit/error_handler_test \
	tests/unit/error_test \
	tests/unit/fixed_buffer_test \
	tests/unit/io_control_test \
	tests/unit/ipv4/address_test \
	tests/unit/ipv4/basic_host_resolver_test \
	tests/unit/ipv4/host_resolver_service_test \
	tests/unit/ipv4/host_resolver_test \
	tests/unit/ipv4/host_test \
	tests/unit/ipv4/multicast_test \
	tests/unit/ipv4/socket_option_test \
	tests/unit/ipv4/tcp_test \
	tests/unit/ipv4/udp_test \
	tests/unit/is_read_buffered_test \
	tests/unit/is_write_buffered_test \
	tests/unit/locking_dispatcher_service_test \
	tests/unit/locking_dispatcher_test \
	tests/unit/null_error_handler_test \
	tests/unit/placeholders_test \
	tests/unit/read_test \
	tests/unit/service_factory_test \
	tests/unit/socket_acceptor_service_test \
	tests/unit/socket_acceptor_test \
	tests/unit/socket_base_test \
	tests/unit/socket_option_test \
	tests/unit/stream_socket_service_test \
	tests/unit/stream_socket_test \
	tests/unit/thread_test \
	tests/unit/time_traits_test \
	tests/unit/wrapped_handler_test \
	tests/unit/write_test \
	examples/chat/chat_client \
	examples/chat/chat_server \
	examples/echo/async_tcp_echo_server \
	examples/echo/async_udp_echo_server \
	examples/echo/blocking_tcp_echo_client \
	examples/echo/blocking_tcp_echo_server \
	examples/echo/blocking_udp_echo_client \
	examples/echo/blocking_udp_echo_server \
	examples/iostreams/daytime_client \
	examples/multicast/receiver \
	examples/multicast/sender \
	examples/timeouts/accept_timeout \
	examples/timeouts/connect_timeout \
	examples/timeouts/datagram_receive_timeout \
	examples/timeouts/stream_receive_timeout \
	examples/tutorial/timer1/timer \
	examples/tutorial/timer2/timer \
	examples/tutorial/timer3/timer \
	examples/tutorial/timer4/timer \
	examples/tutorial/timer5/timer \
	examples/tutorial/daytime1/client \
	examples/tutorial/daytime2/server \
	examples/tutorial/daytime3/server \
	examples/tutorial/daytime4/client \
	examples/tutorial/daytime5/server \
	examples/tutorial/daytime6/server \
	examples/tutorial/daytime7/server

TESTS = \
	tests/unit/basic_datagram_socket_test \
	tests/unit/basic_deadline_timer_test \
	tests/unit/basic_demuxer_test \
	tests/unit/basic_locking_dispatcher_test \
	tests/unit/basic_socket_acceptor_test \
	tests/unit/basic_stream_socket_test \
	tests/unit/buffered_read_stream_test \
	tests/unit/buffered_stream_test \
	tests/unit/buffered_write_stream_test \
	tests/unit/buffers_test \
	tests/unit/buffer_test \
	tests/unit/consuming_buffers_test \
	tests/unit/datagram_socket_service_test \
	tests/unit/datagram_socket_test \
	tests/unit/deadline_timer_service_test \
	tests/unit/deadline_timer_test \
	tests/unit/default_error_handler_test \
	tests/unit/demuxer_service_test \
	tests/unit/demuxer_test \
	tests/unit/error_handler_test \
	tests/unit/error_test \
	tests/unit/fixed_buffer_test \
	tests/unit/io_control_test \
	tests/unit/ipv4/address_test \
	tests/unit/ipv4/basic_host_resolver_test \
	tests/unit/ipv4/host_resolver_service_test \
	tests/unit/ipv4/host_resolver_test \
	tests/unit/ipv4/host_test \
	tests/unit/ipv4/multicast_test \
	tests/unit/ipv4/socket_option_test \
	tests/unit/ipv4/tcp_test \
	tests/unit/ipv4/udp_test \
	tests/unit/is_read_buffered_test \
	tests/unit/is_write_buffered_test \
	tests/unit/locking_dispatcher_service_test \
	tests/unit/locking_dispatcher_test \
	tests/unit/null_error_handler_test \
	tests/unit/placeholders_test \
	tests/unit/read_test \
	tests/unit/service_factory_test \
	tests/unit/socket_acceptor_service_test \
	tests/unit/socket_acceptor_test \
	tests/unit/socket_base_test \
	tests/unit/socket_option_test \
	tests/unit/stream_socket_service_test \
	tests/unit/stream_socket_test \
	tests/unit/thread_test \
	tests/unit/time_traits_test \
	tests/unit/wrapped_handler_test \
	tests/unit/write_test

noinst_HEADERS = \
	tests/unit/unit_test.hpp \
	examples/chat/chat_message.hpp \
	examples/iostreams/socket_stream.hpp

AM_CXXFLAGS = -I$(srcdir)/../include

tests_performance_client_SOURCES = tests/performance/client.cpp
tests_performance_server_SOURCES = tests/performance/server.cpp
tests_unit_demuxer_test_SOURCES = tests/unit/demuxer_test.cpp
tests_unit_ipv4_address_test_SOURCES = tests/unit/ipv4/address_test.cpp
tests_unit_ipv4_basic_host_resolver_test_SOURCES = tests/unit/ipv4/basic_host_resolver_test.cpp
tests_unit_ipv4_host_resolver_service_test_SOURCES = tests/unit/ipv4/host_resolver_service_test.cpp
tests_unit_ipv4_host_resolver_test_SOURCES = tests/unit/ipv4/host_resolver_test.cpp
tests_unit_ipv4_host_test_SOURCES = tests/unit/ipv4/host_test.cpp
tests_unit_ipv4_multicast_test_SOURCES = tests/unit/ipv4/multicast_test.cpp
tests_unit_ipv4_socket_option_test_SOURCES = tests/unit/ipv4/socket_option_test.cpp
tests_unit_ipv4_tcp_test_SOURCES = tests/unit/ipv4/tcp_test.cpp
tests_unit_ipv4_udp_test_SOURCES = tests/unit/ipv4/udp_test.cpp
tests_unit_fixed_buffer_test_SOURCES = tests/unit/fixed_buffer_test.cpp
tests_unit_locking_dispatcher_test_SOURCES = tests/unit/locking_dispatcher_test.cpp
tests_unit_is_read_buffered_test_SOURCES = tests/unit/is_read_buffered_test.cpp
tests_unit_socket_acceptor_test_SOURCES = tests/unit/socket_acceptor_test.cpp
tests_unit_datagram_socket_test_SOURCES = tests/unit/datagram_socket_test.cpp
tests_unit_read_test_SOURCES = tests/unit/read_test.cpp
tests_unit_write_test_SOURCES = tests/unit/write_test.cpp
tests_unit_is_write_buffered_test_SOURCES = tests/unit/is_write_buffered_test.cpp
tests_unit_basic_datagram_socket_test_SOURCES = tests/unit/basic_datagram_socket_test.cpp
tests_unit_deadline_timer_test_SOURCES = tests/unit/deadline_timer_test.cpp
tests_unit_buffer_test_SOURCES = tests/unit/buffer_test.cpp
tests_unit_error_handler_test_SOURCES = tests/unit/error_handler_test.cpp
tests_unit_null_error_handler_test_SOURCES = tests/unit/null_error_handler_test.cpp
tests_unit_basic_deadline_timer_test_SOURCES = tests/unit/basic_deadline_timer_test.cpp
tests_unit_basic_demuxer_test_SOURCES = tests/unit/basic_demuxer_test.cpp
tests_unit_basic_locking_dispatcher_test_SOURCES = tests/unit/basic_locking_dispatcher_test.cpp
tests_unit_basic_socket_acceptor_test_SOURCES = tests/unit/basic_socket_acceptor_test.cpp
tests_unit_basic_stream_socket_test_SOURCES = tests/unit/basic_stream_socket_test.cpp
tests_unit_buffered_read_stream_test_SOURCES = tests/unit/buffered_read_stream_test.cpp
tests_unit_buffered_stream_test_SOURCES = tests/unit/buffered_stream_test.cpp
tests_unit_buffered_write_stream_test_SOURCES = tests/unit/buffered_write_stream_test.cpp
tests_unit_buffers_test_SOURCES = tests/unit/buffers_test.cpp
tests_unit_consuming_buffers_test_SOURCES = tests/unit/consuming_buffers_test.cpp
tests_unit_datagram_socket_service_test_SOURCES = tests/unit/datagram_socket_service_test.cpp
tests_unit_deadline_timer_service_test_SOURCES = tests/unit/deadline_timer_service_test.cpp
tests_unit_default_error_handler_test_SOURCES = tests/unit/default_error_handler_test.cpp
tests_unit_demuxer_service_test_SOURCES = tests/unit/demuxer_service_test.cpp
tests_unit_error_test_SOURCES = tests/unit/error_test.cpp
tests_unit_io_control_test_SOURCES = tests/unit/io_control_test.cpp
tests_unit_locking_dispatcher_service_test_SOURCES = tests/unit/locking_dispatcher_service_test.cpp
tests_unit_placeholders_test_SOURCES = tests/unit/placeholders_test.cpp
tests_unit_service_factory_test_SOURCES = tests/unit/service_factory_test.cpp
tests_unit_socket_acceptor_service_test_SOURCES = tests/unit/socket_acceptor_service_test.cpp
tests_unit_socket_base_test_SOURCES = tests/unit/socket_base_test.cpp
tests_unit_socket_option_test_SOURCES = tests/unit/socket_option_test.cpp
tests_unit_stream_socket_service_test_SOURCES = tests/unit/stream_socket_service_test.cpp
tests_unit_stream_socket_test_SOURCES = tests/unit/stream_socket_test.cpp
tests_unit_thread_test_SOURCES = tests/unit/thread_test.cpp
tests_unit_time_traits_test_SOURCES = tests/unit/time_traits_test.cpp
tests_unit_wrapped_handler_test_SOURCES = tests/unit/wrapped_handler_test.cpp
examples_chat_chat_client_SOURCES = examples/chat/chat_client.cpp
examples_chat_chat_server_SOURCES = examples/chat/chat_server.cpp
examples_echo_async_tcp_echo_server_SOURCES = examples/echo/async_tcp_echo_server.cpp
examples_echo_async_udp_echo_server_SOURCES = examples/echo/async_udp_echo_server.cpp
examples_echo_blocking_tcp_echo_client_SOURCES = examples/echo/blocking_tcp_echo_client.cpp
examples_echo_blocking_tcp_echo_server_SOURCES = examples/echo/blocking_tcp_echo_server.cpp
examples_echo_blocking_udp_echo_client_SOURCES = examples/echo/blocking_udp_echo_client.cpp
examples_echo_blocking_udp_echo_server_SOURCES = examples/echo/blocking_udp_echo_server.cpp
examples_iostreams_daytime_client_SOURCES = examples/iostreams/daytime_client.cpp
examples_multicast_receiver_SOURCES = examples/multicast/receiver.cpp
examples_multicast_sender_SOURCES = examples/multicast/sender.cpp
examples_timeouts_accept_timeout_SOURCES = examples/timeouts/accept_timeout.cpp
examples_timeouts_connect_timeout_SOURCES = examples/timeouts/connect_timeout.cpp
examples_timeouts_datagram_receive_timeout_SOURCES = examples/timeouts/datagram_receive_timeout.cpp
examples_timeouts_stream_receive_timeout_SOURCES = examples/timeouts/stream_receive_timeout.cpp
examples_tutorial_timer1_timer_SOURCES = examples/tutorial/timer1/timer.cpp
examples_tutorial_timer2_timer_SOURCES = examples/tutorial/timer2/timer.cpp
examples_tutorial_timer3_timer_SOURCES = examples/tutorial/timer3/timer.cpp
examples_tutorial_timer4_timer_SOURCES = examples/tutorial/timer4/timer.cpp
examples_tutorial_timer5_timer_SOURCES = examples/tutorial/timer5/timer.cpp
examples_tutorial_daytime1_client_SOURCES = examples/tutorial/daytime1/client.cpp
examples_tutorial_daytime2_server_SOURCES = examples/tutorial/daytime2/server.cpp
examples_tutorial_daytime3_server_SOURCES = examples/tutorial/daytime3/server.cpp
examples_tutorial_daytime4_client_SOURCES = examples/tutorial/daytime4/client.cpp
examples_tutorial_daytime5_server_SOURCES = examples/tutorial/daytime5/server.cpp
examples_tutorial_daytime6_server_SOURCES = examples/tutorial/daytime6/server.cpp
examples_tutorial_daytime7_server_SOURCES = examples/tutorial/daytime7/server.cpp

EXTRA_DIST = \
	Makefile.bor \
	Makefile.mgw \
	Makefile.msc

MAINTAINERCLEANFILES = \
	$(srcdir)/Makefile.in
