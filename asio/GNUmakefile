.PHONY: all check test clean
all: .init
	cmake --workflow --preset dev

check:
	run-clang-tidy -p build/dev -checks='-*,misc-header-*' src/tests
	# ninja -C build/dev spell-check
	# ninja -C build/dev format-check

test:
	cmake --preset ci-macos
	DESTDIR=$(CURDIR)/stagedir cmake --build build --target install
	cmake -G Ninja -B build/tests -S src/tests -D CMAKE_PREFIX_PATH=$(CURDIR)/stagedir/usr/local
	cmake --build build/tests
	ctest --test-dir build/tests

.init: requirements.txt
	-pip3 install --user --upgrade -r requirements.txt
	touch .init

clean:
	rm -rf build
