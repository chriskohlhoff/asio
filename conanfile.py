from conans import ConanFile, tools
import os


class AsioConan(ConanFile):
    name = "asio"
    version = "1.11.0"
    description = "Asio C++ Library"
    license = "MIT"
    url = "https://github.com/chriskohlhoff/asio"
    exports_sources = "asio/include/*"
    no_copy_source = True

    def package(self):
        self.copy(pattern="*.hpp", src="asio/include", dst="include")
        self.copy(pattern="*.ipp", src="asio/include", dst="include")

    def package_info(self):
        self.cpp_info.defines = ["ASIO_STANDALONE"]

    def package_id(self):
        self.info.header_only()
