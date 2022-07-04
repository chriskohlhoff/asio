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
	operations/c_callback_wrapper \
	operations/callback_wrapper \
	operations/composed_1 \
	operations/composed_1 \
	operations/composed_2 \
	operations/composed_3 \
	operations/composed_4 \
	operations/composed_5 \
	operations/composed_6 \
	operations/composed_7 \
	operations/composed_8

if HAVE_COROUTINES
noinst_PROGRAMS += \
	channels/throttling_proxy
endif

AM_CXXFLAGS = -I$(srcdir)/../../../include

operations_c_callback_wrapper_SOURCES = operations/c_callback_wrapper.cpp
operations_callback_wrapper_SOURCES = operations/callback_wrapper.cpp
operations_composed_1_SOURCES = operations/composed_1.cpp
operations_composed_2_SOURCES = operations/composed_2.cpp
operations_composed_3_SOURCES = operations/composed_3.cpp
operations_composed_4_SOURCES = operations/composed_4.cpp
operations_composed_5_SOURCES = operations/composed_5.cpp
operations_composed_6_SOURCES = operations/composed_6.cpp
operations_composed_7_SOURCES = operations/composed_7.cpp
operations_composed_8_SOURCES = operations/composed_8.cpp

if HAVE_COROUTINES
channels_throttling_proxy_SOURCES = channels/throttling_proxy.cpp
endif

MAINTAINERCLEANFILES = \
	$(srcdir)/Makefile.in
