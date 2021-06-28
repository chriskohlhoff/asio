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
  parallel_group/parallel_sort \
  parallel_group/wait_for_all \
  parallel_group/wait_for_one \
  parallel_group/wait_for_one_error \
  parallel_group/wait_for_one_success

if HAVE_COROUTINES
noinst_PROGRAMS += \
	coroutines_ts/chat_server \
	coroutines_ts/echo_server \
	coroutines_ts/echo_server_with_default \
	coroutines_ts/echo_server_with_as_single_default \
	coroutines_ts/echo_server_with_as_tuple_default \
	coroutines_ts/refactored_echo_server
endif

AM_CXXFLAGS = -I$(srcdir)/../../../include

parallel_group_parallel_sort_SOURCES = parallel_group/parallel_sort.cpp
parallel_group_wait_for_all_SOURCES = parallel_group/wait_for_all.cpp
parallel_group_wait_for_one_SOURCES = parallel_group/wait_for_one.cpp
parallel_group_wait_for_one_error_SOURCES = parallel_group/wait_for_one_error.cpp
parallel_group_wait_for_one_success_SOURCES = parallel_group/wait_for_one_success.cpp

if HAVE_COROUTINES
coroutines_ts_chat_server_SOURCES = coroutines_ts/chat_server.cpp
coroutines_ts_echo_server_SOURCES = coroutines_ts/echo_server.cpp
coroutines_ts_echo_server_with_default_SOURCES = coroutines_ts/echo_server_with_default.cpp
coroutines_ts_echo_server_with_as_single_default_SOURCES = coroutines_ts/echo_server_with_as_single_default.cpp
coroutines_ts_echo_server_with_as_tuple_default_SOURCES = coroutines_ts/echo_server_with_as_tuple_default.cpp
coroutines_ts_refactored_echo_server_SOURCES = coroutines_ts/refactored_echo_server.cpp
endif

EXTRA_DIST = \
	coroutines_ts/range_based_for.cpp

MAINTAINERCLEANFILES = \
	$(srcdir)/Makefile.in
