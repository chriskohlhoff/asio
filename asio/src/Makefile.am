AUTOMAKE_OPTIONS = subdir-objects

SUBDIRS = doc

noinst_PROGRAMS = \
	tests/dispatcher_test \
	tests/dgram_echo_client_test \
	tests/dgram_echo_server_test \
	tests/echo_client_test \
	tests/echo_server_test \
	tests/socket_accept_test \
	tests/timed_accept_test \
	tests/timed_connect_test \
	tests/timed_dgram_recv_test \
	tests/timer_test \
	tests/tpc_echo_server_test

AM_CXXFLAGS = -I../include

tests_dispatcher_test_SOURCES = tests/dispatcher_test.cpp

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

EXTRA_DIST = \
	Makefile.bor \
	Makefile.mgw \
	Makefile.msc

MAINTAINERCLEANFILES = \
	Makefile.in
