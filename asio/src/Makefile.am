AUTOMAKE_OPTIONS = subdir-objects

SUBDIRS = doc

noinst_PROGRAMS = \
	tests/dgram_echo_client_test \
	tests/dgram_echo_server_test \
	tests/echo_client_test \
	tests/echo_server_test \
	tests/socket_accept_test \
	tests/timed_accept_test \
	tests/timed_connect_test \
	tests/timed_dgram_recv_test \
	tests/timer_test \
	tests/tpc_echo_server_test \
	tests/performance/client \
	tests/performance/server \
	tests/unit/demuxer_test \
	tests/unit/error_handler_test \
	tests/unit/fixed_buffer_test \
	examples/chat/chat_client \
	examples/chat/chat_server \
	examples/echo/async_tcp_echo_server \
	examples/echo/async_udp_echo_server \
	examples/echo/blocking_tcp_echo_client \
	examples/echo/blocking_tcp_echo_server \
	examples/echo/blocking_udp_echo_client \
	examples/echo/blocking_udp_echo_server \
	examples/tutorial/timer1/timer \
	examples/tutorial/timer2/timer \
	examples/tutorial/timer3/timer \
	examples/tutorial/timer4/timer

TESTS = \
	tests/unit/demuxer_test \
	tests/unit/error_handler_test \
	tests/unit/fixed_buffer_test

noinst_HEADERS = \
	tests/unit/unit_test.hpp \
	examples/chat/chat_message.hpp

AM_CXXFLAGS = -I../include

tests_dgram_echo_server_test_SOURCES = tests/dgram_echo_server_test.cpp
tests_dgram_echo_client_test_SOURCES = tests/dgram_echo_client_test.cpp
tests_echo_client_test_SOURCES = tests/echo_client_test.cpp
tests_echo_server_test_SOURCES = tests/echo_server_test.cpp
tests_socket_accept_test_SOURCES = tests/socket_accept_test.cpp
tests_timed_accept_test_SOURCES = tests/timed_accept_test.cpp
tests_timed_connect_test_SOURCES = tests/timed_connect_test.cpp
tests_timed_dgram_recv_test_SOURCES = tests/timed_dgram_recv_test.cpp
tests_timer_test_SOURCES = tests/timer_test.cpp
tests_tpc_echo_server_test_SOURCES = tests/tpc_echo_server_test.cpp
tests_performance_client_SOURCES = tests/performance/client.cpp
tests_performance_server_SOURCES = tests/performance/server.cpp
tests_unit_demuxer_test_SOURCES = tests/unit/demuxer_test.cpp
tests_unit_error_handler_test_SOURCES = tests/unit/error_handler_test.cpp
tests_unit_fixed_buffer_test_SOURCES = tests/unit/fixed_buffer_test.cpp
examples_chat_chat_client_SOURCES = examples/chat/chat_client.cpp
examples_chat_chat_server_SOURCES = examples/chat/chat_server.cpp
examples_echo_async_tcp_echo_server_SOURCES = examples/echo/async_tcp_echo_server.cpp
examples_echo_async_udp_echo_server_SOURCES = examples/echo/async_udp_echo_server.cpp
examples_echo_blocking_tcp_echo_client_SOURCES = examples/echo/blocking_tcp_echo_client.cpp
examples_echo_blocking_tcp_echo_server_SOURCES = examples/echo/blocking_tcp_echo_server.cpp
examples_echo_blocking_udp_echo_client_SOURCES = examples/echo/blocking_udp_echo_client.cpp
examples_echo_blocking_udp_echo_server_SOURCES = examples/echo/blocking_udp_echo_server.cpp
examples_tutorial_timer1_timer_SOURCES = examples/tutorial/timer1/timer.cpp
examples_tutorial_timer2_timer_SOURCES = examples/tutorial/timer2/timer.cpp
examples_tutorial_timer3_timer_SOURCES = examples/tutorial/timer3/timer.cpp
examples_tutorial_timer4_timer_SOURCES = examples/tutorial/timer4/timer.cpp

EXTRA_DIST = \
	Makefile.bor \
	Makefile.mgw \
	Makefile.msc

MAINTAINERCLEANFILES = \
	Makefile.in
