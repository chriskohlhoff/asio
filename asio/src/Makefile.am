AUTOMAKE_OPTIONS = subdir-objects

SUBDIRS = doc

lib_LIBRARIES = ../lib/libasio.a

___lib_libasio_a_SOURCES = \
	asio/completion_context.cpp \
	asio/completion_context_locker.cpp \
	asio/counting_completion_context.cpp \
	asio/demuxer_task.cpp \
	asio/socket_error.cpp \
	asio/detail/demuxer_task_thread.cpp \
	asio/detail/select_interrupter.cpp \
	asio/detail/shared_thread_demuxer_service.cpp

nobase_include_HEADERS = \
	asio.hpp \
	asio/basic_demuxer.hpp \
	asio/basic_dgram_socket.hpp \
	asio/basic_socket_acceptor.hpp \
	asio/basic_socket_connector.hpp \
	asio/basic_stream_socket.hpp \
	asio/basic_timer_queue.hpp \
	asio/buffered_recv_stream.hpp \
	asio/buffered_send_stream.hpp \
	asio/buffered_stream.hpp \
	asio/completion_context.hpp \
	asio/completion_context_locker.hpp \
	asio/counting_completion_context.hpp \
	asio/demuxer.hpp \
	asio/demuxer_task.hpp \
	asio/detail/demuxer_task_thread.hpp \
	asio/detail/pop_options.hpp \
	asio/detail/push_options.hpp \
	asio/detail/reactive_dgram_socket_service.hpp \
	asio/detail/reactive_socket_acceptor_service.hpp \
	asio/detail/reactive_socket_connector_service.hpp \
	asio/detail/reactive_stream_socket_service.hpp \
	asio/detail/reactor_op_queue.hpp \
	asio/detail/select_interrupter.hpp \
	asio/detail/select_reactor.hpp \
	asio/detail/service_registry.hpp \
	asio/detail/shared_thread_demuxer_service.hpp \
	asio/detail/socket_holder.hpp \
	asio/detail/socket_ops.hpp \
	asio/detail/socket_types.hpp \
	asio/detail/timer_queue_service.hpp \
	asio/detail/tss_bool.hpp \
	asio/detail/win_iocp_provider.hpp \
	asio/detail/winsock_init.hpp \
	asio/dgram_socket.hpp \
	asio/inet_address_v4.hpp \
	asio/service_factory.hpp \
	asio/socket_acceptor.hpp \
	asio/socket_connector.hpp \
	asio/socket_error.hpp \
	asio/stream_socket.hpp \
	asio/timer_queue.hpp

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
	../bin/timer_queue_test \
	../bin/tpc_echo_server_test

___bin_dispatcher_test_SOURCES = tests/dispatcher_test.cpp
___bin_dispatcher_test_LDADD = ../lib/libasio.a -lboost_thread

___bin_dgram_echo_server_test_SOURCES = tests/dgram_echo_server_test.cpp
___bin_dgram_echo_server_test_LDADD = ../lib/libasio.a -lboost_thread

___bin_dgram_echo_client_test_SOURCES = tests/dgram_echo_client_test.cpp
___bin_dgram_echo_client_test_LDADD = ../lib/libasio.a -lboost_thread

___bin_echo_client_test_SOURCES = tests/echo_client_test.cpp
___bin_echo_client_test_LDADD = ../lib/libasio.a -lboost_thread

___bin_echo_server_test_SOURCES = tests/echo_server_test.cpp
___bin_echo_server_test_LDADD = ../lib/libasio.a -lboost_thread

___bin_socket_accept_test_SOURCES = tests/socket_accept_test.cpp
___bin_socket_accept_test_LDADD = ../lib/libasio.a -lboost_thread

___bin_timed_accept_test_SOURCES = tests/timed_accept_test.cpp
___bin_timed_accept_test_LDADD = ../lib/libasio.a -lboost_thread

___bin_timed_connect_test_SOURCES = tests/timed_connect_test.cpp
___bin_timed_connect_test_LDADD = ../lib/libasio.a -lboost_thread

___bin_timed_dgram_recv_test_SOURCES = tests/timed_dgram_recv_test.cpp
___bin_timed_dgram_recv_test_LDADD = ../lib/libasio.a -lboost_thread

___bin_timer_queue_test_SOURCES = tests/timer_queue_test.cpp
___bin_timer_queue_test_LDADD = ../lib/libasio.a -lboost_thread

___bin_tpc_echo_server_test_SOURCES = tests/tpc_echo_server_test.cpp
___bin_tpc_echo_server_test_LDADD = ../lib/libasio.a -lboost_thread

EXTRA_DIST = \
	Makefile.bor \
	Makefile.msc

MAINTAINERCLEANFILES = \
	Makefile.in
