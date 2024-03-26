.PHONY: all check clean
all: .init
	cmake --preset dev
	cmake --build --preset dev
	ninja -C build/dev all_verify_interface_header_sets

check:
	ninja -C build/dev spell-check
	ninja -C build/dev format-check

.init: requirements.txt
	pip3 install -U -r requirements.txt
	touch .init

clean:
	ninja -C build/dev clean
