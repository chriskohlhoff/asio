AUTOMAKE_OPTIONS = subdir-objects

SUBDIRS = doc

lib_LIBRARIES = ../lib/libasio.a

___lib_libasio_a_SOURCES = \
	asio/completion_context.cpp \
	asio/completion_context_locker.cpp \
	asio/counting_completion_context.cpp \
	asio/demuxer.cpp \
	asio/demuxer_service.cpp \
	asio/demuxer_task.cpp \
	asio/generic_address.cpp \
	asio/inet_address_v4.cpp \
	asio/service.cpp \
	asio/service_provider.cpp \
	asio/service_provider_factory.cpp \
	asio/service_unavailable.cpp \
	asio/socket_acceptor.cpp \
	asio/socket_address.cpp \
	asio/socket_connector.cpp \
	asio/socket_error.cpp \
	asio/stream_socket.cpp \
	asio/detail/default_service_provider_factory.cpp \
	asio/detail/demuxer_thread_pool.cpp \
	asio/detail/dgram_socket_service.cpp \
	asio/detail/selector.cpp \
	asio/detail/select_interrupter.cpp \
	asio/detail/select_op.cpp \
	asio/detail/select_op_queue.cpp \
	asio/detail/select_provider.cpp \
	asio/detail/service_registry.cpp \
	asio/detail/shared_thread_demuxer_provider.cpp \
	asio/detail/socket_acceptor_service.cpp \
	asio/detail/socket_connector_impl.cpp \
	asio/detail/socket_connector_service.cpp \
	asio/detail/stream_socket_service.cpp \
	asio/detail/timer_queue_provider.cpp \
	asio/detail/timer_queue_service.cpp \
	asio/detail/win_iocp_provider.cpp

nobase_include_HEADERS = \
	asio.hpp \
	asio/basic_dgram_socket.hpp \
	asio/basic_timer_queue.hpp \
	asio/buffered_recv_stream.hpp \
	asio/buffered_send_stream.hpp \
	asio/buffered_stream.hpp \
	asio/completion_context.hpp \
	asio/completion_context_locker.hpp \
	asio/counting_completion_context.hpp \
	asio/demuxer.hpp \
	asio/demuxer_service.hpp \
	asio/demuxer_task.hpp \
	asio/dgram_socket.hpp \
	asio/generic_address.hpp \
	asio/inet_address_v4.hpp \
	asio/service.hpp \
	asio/service_provider.hpp \
	asio/service_provider_factory.hpp \
	asio/service_type_id.hpp \
	asio/service_unavailable.hpp \
	asio/socket_acceptor.hpp \
	asio/socket_address.hpp \
	asio/socket_connector.hpp \
	asio/socket_error.hpp \
	asio/stream_socket.hpp \
	asio/timer_queue.hpp \
	asio/detail/dgram_socket_service.hpp \
	asio/detail/pop_options.hpp \
	asio/detail/push_options.hpp \
	asio/detail/socket_acceptor_service.hpp \
	asio/detail/socket_connector_service.hpp \
	asio/detail/stream_socket_service.hpp \
	asio/detail/socket_types.hpp \
	asio/detail/timer_queue_service.hpp

noinst_HEADERS = \
	asio/detail/default_service_provider_factory.hpp \
	asio/detail/demuxer_thread_pool.hpp \
	asio/detail/selector.hpp \
	asio/detail/select_interrupter.hpp \
	asio/detail/select_op.hpp \
	asio/detail/select_op_queue.hpp \
	asio/detail/select_provider.hpp \
	asio/detail/service_registry.hpp \
	asio/detail/shared_thread_demuxer_provider.hpp \
	asio/detail/socket_connector_impl.hpp \
	asio/detail/socket_holder.hpp \
	asio/detail/socket_ops.hpp \
	asio/detail/timer_queue_provider.hpp \
	asio/detail/win_iocp_provider.hpp

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
