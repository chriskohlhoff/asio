AUTOMAKE_OPTIONS = subdir-objects

SUBDIRS = doc

noinst_PROGRAMS = \
	tests/performance/client \
	tests/performance/server \
	tests/unit/demuxer_test \
	tests/unit/dgram_socket_test \
	tests/unit/error_handler_test \
	tests/unit/fixed_buffer_test \
	tests/unit/locking_dispatcher_test \
	tests/unit/recv_test \
	tests/unit/send_test \
	tests/unit/socket_acceptor_test \
	tests/unit/timer_test \
	examples/chat/chat_client \
	examples/chat/chat_server \
	examples/echo/async_tcp_echo_server \
	examples/echo/async_udp_echo_server \
	examples/echo/blocking_tcp_echo_client \
	examples/echo/blocking_tcp_echo_server \
	examples/echo/blocking_udp_echo_client \
	examples/echo/blocking_udp_echo_server \
	examples/timeouts/accept_timeout \
	examples/timeouts/connect_timeout \
	examples/timeouts/dgram_recv_timeout \
	examples/tutorial/timer1/timer \
	examples/tutorial/timer2/timer \
	examples/tutorial/timer3/timer \
	examples/tutorial/timer4/timer \
	examples/tutorial/daytime1/client \
	examples/tutorial/daytime2/server \
	examples/tutorial/daytime3/server \
	examples/tutorial/daytime4/client \
	examples/tutorial/daytime5/server \
	examples/tutorial/daytime6/server \
	examples/tutorial/daytime7/server

TESTS = \
	tests/unit/demuxer_test \
	tests/unit/dgram_socket_test \
	tests/unit/error_handler_test \
	tests/unit/fixed_buffer_test \
	tests/unit/locking_dispatcher_test \
	tests/unit/recv_test \
	tests/unit/send_test \
	tests/unit/socket_acceptor_test \
	tests/unit/timer_test

noinst_HEADERS = \
	tests/unit/unit_test.hpp \
	examples/chat/chat_message.hpp

AM_CXXFLAGS = -I$(srcdir)/../include

tests_performance_client_SOURCES = tests/performance/client.cpp
tests_performance_server_SOURCES = tests/performance/server.cpp
tests_unit_demuxer_test_SOURCES = tests/unit/demuxer_test.cpp
tests_unit_dgram_socket_test_SOURCES = tests/unit/dgram_socket_test.cpp
tests_unit_error_handler_test_SOURCES = tests/unit/error_handler_test.cpp
tests_unit_fixed_buffer_test_SOURCES = tests/unit/fixed_buffer_test.cpp
tests_unit_locking_dispatcher_test_SOURCES = tests/unit/locking_dispatcher_test.cpp
tests_unit_recv_test_SOURCES = tests/unit/recv_test.cpp
tests_unit_send_test_SOURCES = tests/unit/send_test.cpp
tests_unit_socket_acceptor_test_SOURCES = tests/unit/socket_acceptor_test.cpp
tests_unit_timer_test_SOURCES = tests/unit/timer_test.cpp
examples_chat_chat_client_SOURCES = examples/chat/chat_client.cpp
examples_chat_chat_server_SOURCES = examples/chat/chat_server.cpp
examples_echo_async_tcp_echo_server_SOURCES = examples/echo/async_tcp_echo_server.cpp
examples_echo_async_udp_echo_server_SOURCES = examples/echo/async_udp_echo_server.cpp
examples_echo_blocking_tcp_echo_client_SOURCES = examples/echo/blocking_tcp_echo_client.cpp
examples_echo_blocking_tcp_echo_server_SOURCES = examples/echo/blocking_tcp_echo_server.cpp
examples_echo_blocking_udp_echo_client_SOURCES = examples/echo/blocking_udp_echo_client.cpp
examples_echo_blocking_udp_echo_server_SOURCES = examples/echo/blocking_udp_echo_server.cpp
examples_timeouts_accept_timeout_SOURCES = examples/timeouts/accept_timeout.cpp
examples_timeouts_connect_timeout_SOURCES = examples/timeouts/connect_timeout.cpp
examples_timeouts_dgram_recv_timeout_SOURCES = examples/timeouts/dgram_recv_timeout.cpp
examples_tutorial_timer1_timer_SOURCES = examples/tutorial/timer1/timer.cpp
examples_tutorial_timer2_timer_SOURCES = examples/tutorial/timer2/timer.cpp
examples_tutorial_timer3_timer_SOURCES = examples/tutorial/timer3/timer.cpp
examples_tutorial_timer4_timer_SOURCES = examples/tutorial/timer4/timer.cpp
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
	Makefile.in
