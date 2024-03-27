.PHONY: all check clean
all: .init
	cmake --workflow --preset dev

check:
	run-clang-tidy -p build/dev -checks='-*,misc-header-*' src/tests
	# ninja -C build/dev spell-check
	# ninja -C build/dev format-check

.init: requirements.txt
	-pip3 install --user --upgrade -r requirements.txt
	touch .init

clean:
	rm -rf build
