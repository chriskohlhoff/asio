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
	executors/bank_account_1 \
	executors/bank_account_2 \
	executors/bank_account_3 \
	executors/bank_account_4 \
	executors/bank_account_5

AM_CXXFLAGS = -I$(srcdir)/../../../include

executors_bank_account_1_SOURCES = executors/bank_account_1.cpp
executors_bank_account_2_SOURCES = executors/bank_account_2.cpp
executors_bank_account_3_SOURCES = executors/bank_account_3.cpp
executors_bank_account_4_SOURCES = executors/bank_account_4.cpp
executors_bank_account_5_SOURCES = executors/bank_account_5.cpp

MAINTAINERCLEANFILES = \
	$(srcdir)/Makefile.in
