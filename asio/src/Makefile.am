AUTOMAKE_OPTIONS = subdir-objects

SUBDIRS = doc

noinst_PROGRAMS = \
	../bin/dispatcher_test \
	../bin/dgram_echo_client_test \
	../bin/dgram_echo_server_test \
	../bin/echo_client_test \
	../bin/echo_server_test \
	../bin/socket_accept_test \
	../bin/timed_accept_test \
	../bin/timed_connect_test \
	../bin/timed_dgram_recv_test \
	../bin/timer_test \
	../bin/tpc_echo_server_test

AM_CXXFLAGS = -I../include

___bin_dispatcher_test_SOURCES = tests/dispatcher_test.cpp
___bin_dispatcher_test_LDADD = -lpthread

___bin_dgram_echo_server_test_SOURCES = tests/dgram_echo_server_test.cpp
___bin_dgram_echo_server_test_LDADD = -lpthread

___bin_dgram_echo_client_test_SOURCES = tests/dgram_echo_client_test.cpp
___bin_dgram_echo_client_test_LDADD = -lpthread

___bin_echo_client_test_SOURCES = tests/echo_client_test.cpp
___bin_echo_client_test_LDADD = -lpthread

___bin_echo_server_test_SOURCES = tests/echo_server_test.cpp
___bin_echo_server_test_LDADD = -lpthread

___bin_socket_accept_test_SOURCES = tests/socket_accept_test.cpp
___bin_socket_accept_test_LDADD = -lpthread

___bin_timed_accept_test_SOURCES = tests/timed_accept_test.cpp
___bin_timed_accept_test_LDADD = -lpthread

___bin_timed_connect_test_SOURCES = tests/timed_connect_test.cpp
___bin_timed_connect_test_LDADD = -lpthread

___bin_timed_dgram_recv_test_SOURCES = tests/timed_dgram_recv_test.cpp
___bin_timed_dgram_recv_test_LDADD = -lpthread

___bin_timer_test_SOURCES = tests/timer_test.cpp
___bin_timer_test_LDADD = -lpthread

___bin_tpc_echo_server_test_SOURCES = tests/tpc_echo_server_test.cpp
___bin_tpc_echo_server_test_LDADD = -lpthread

EXTRA_DIST = \
	Makefile.bor \
	Makefile.msc

MAINTAINERCLEANFILES = \
	Makefile.in
