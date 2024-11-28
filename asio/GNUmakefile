# Standard stuff

.SUFFIXES:
$(VERBOSE).SILENT:

MAKEFLAGS+= --no-builtin-rules
MAKEFLAGS+= --warn-undefined-variables

export hostSystemName=$(shell uname)

.PHONY: all check test clean distclean
all: .init
	cmake --workflow --preset dev --fresh

check:
	-iwyu_tool -p build/dev \
		build/dev/asio_verify_interface_header_sets/asio/basic_readable_pipe.hpp.cxx \
		build/dev/asio_verify_interface_header_sets/asio/detail/reactive_socket_send_op.hpp.cxx \
		-- -Xiwyu --cxx17ns #XXX -Xiwyu --transitive_includes_only
	run-clang-tidy -p build/dev -checks='-*,misc-header-*,misc-include-*' src/tests
	# ninja -C build/dev spell-check
	# ninja -C build/dev format-check

test:
	cmake --preset ci-${hostSystemName}
	cmake --build build
	cmake --install build --prefix $(CURDIR)/stagedir
	cmake -G Ninja -B build/tests -S src/tests -D CMAKE_PREFIX_PATH=$(CURDIR)/stagedir/usr/local
	cmake --build build/tests
	ctest --test-dir build/tests

.init: requirements.txt .CMakeUserPresets.json
	perl -p -e 's/<hostSystemName>/${hostSystemName}/;' .CMakeUserPresets.json > CMakeUserPresets.json
	-pip3 install --user --upgrade -r requirements.txt
	touch .init

clean:
	rm -rf build

distclean: clean
	rm -rf stagedir .init tags *.bak

GNUmakefile :: ;
*.txt :: ;
*.json :: ;

# Anything we don't know how to build will use this rule.
# The command is a do-nothing command.
#
% :: ;
