if ENABLE_EXAMPLES
EXAMPLES_CPP03 = examples/cpp03
EXAMPLES_CPP11 = examples/cpp11
EXAMPLES_CPP14 = examples/cpp14
endif

if ENABLE_TESTS
TESTS = tests
endif

if STANDALONE
SUBDIRS = $(EXAMPLES_CPP11) $(TESTS)
else
SUBDIRS = $(EXAMPLES_CPP03) $(TESTS)
endif

DIST_SUBDIRS = $(EXAMPLES_CPP03) $(EXAMPLES_CPP11) $(EXAMPLES_CPP14) $(TESTS)

EXTRA_DIST = \
	Makefile.mgw \
	Makefile.msc \
	tools/handlerviz.pl

MAINTAINERCLEANFILES = \
	$(srcdir)/Makefile.in
