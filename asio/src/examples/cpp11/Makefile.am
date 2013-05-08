AUTOMAKE_OPTIONS = subdir-objects

if SEPARATE_COMPILATION
noinst_LIBRARIES = libasio.a
libasio_a_SOURCES = ../../asio.cpp
if HAVE_OPENSSL
libasio_a_SOURCES += ../../asio_ssl.cpp
endif
LDADD = libasio.a
endif

noinst_PROGRAMS = \
	allocation/server \
	buffers/reference_counted \
	chat/chat_client \
	chat/chat_server \
	echo/async_tcp_echo_server \
	echo/async_udp_echo_server \
	echo/blocking_tcp_echo_client \
	echo/blocking_tcp_echo_server \
	echo/blocking_udp_echo_client \
	echo/blocking_udp_echo_server

noinst_HEADERS = \
	chat/chat_message.hpp

AM_CXXFLAGS = -I$(srcdir)/../../../include

allocation_server_SOURCES = allocation/server.cpp
buffers_reference_counted_SOURCES = buffers/reference_counted.cpp
chat_chat_client_SOURCES = chat/chat_client.cpp
chat_chat_server_SOURCES = chat/chat_server.cpp
echo_async_tcp_echo_server_SOURCES = echo/async_tcp_echo_server.cpp
echo_async_udp_echo_server_SOURCES = echo/async_udp_echo_server.cpp
echo_blocking_tcp_echo_client_SOURCES = echo/blocking_tcp_echo_client.cpp
echo_blocking_tcp_echo_server_SOURCES = echo/blocking_tcp_echo_server.cpp
echo_blocking_udp_echo_client_SOURCES = echo/blocking_udp_echo_client.cpp
echo_blocking_udp_echo_server_SOURCES = echo/blocking_udp_echo_server.cpp

MAINTAINERCLEANFILES = \
	$(srcdir)/Makefile.in
