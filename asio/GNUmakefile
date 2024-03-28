export hostSystemName=$(shell uname)

.PHONY: all check test clean distclean
all: .init
	cmake --workflow --preset dev

check:
	run-clang-tidy -p build/dev -checks='-*,misc-header-*' src/tests
	# ninja -C build/dev spell-check
	# ninja -C build/dev format-check

test:
	cmake --preset ci-${hostSystemName}
	DESTDIR=$(CURDIR)/stagedir cmake --build build --target install
	cmake -G Ninja -B build/tests -S src/tests -D CMAKE_PREFIX_PATH=$(CURDIR)/stagedir/usr/local
	cmake --build build/tests
	ctest --test-dir build/tests

.init: requirements.txt CMakeUserPresets.json
	perl -p -i.bak -e 's/<hostSystemName>/${hostSystemName}/;' CMakeUserPresets.json
	-pip3 install --user --upgrade -r requirements.txt
	touch .init

clean:
	rm -rf build

distclean: clean
	rm -rf stagedir .init tags *.bak
