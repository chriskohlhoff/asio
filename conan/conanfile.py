from conans import ConanFile, CMake, tools

# from coanan exampe for "How to package header-only libraries"


class AsioConan(ConanFile):
    name = "asio"
    version = "1.12.1"
    license = "Boost Software License - Version 1.0"
    url = "https://github.com/chriskohlhoff/asio/issues"
    description = "Asio C++ Library"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "header_only": [True, False],
        "force_openssl": [True, False],
        "cxx_14": [True, False]
    }
    default_options = '''
shared=False
header_only=True
force_openssl=True
cxx_14=True
'''
    generators = "cmake"
    exports_sources = "../asio/*"
    no_copy_source = True

    def requirements(self):
        if not self.options.header_only:
            if self.options.force_openssl:
                self.requires.add("OpenSSL/1.0.2n@conan/stable", private=False)

    def build(self):
        cmake = CMake(self)
        if self.options.header_only:
            definitions = {
                "ASIO_SEPARATE_COMPILATION": "OFF"
            }
            cmake.configure(defs=definitions, source_folder=".")
        else:
            cmake.configure(source_folder=".")
        cmake.build()
        if not tools.cross_building(self.settings):
            if tools.get_env("CONAN_RUN_TESTS", True):
                cmake.test()
        cmake.install()

    def package(self):
        pass
        # NOTE: done by cmake.install()
        # self.copy("*.hpp", dst="include", src="include")
        # self.copy("*.ipp", dst="include", src="include")
        # self.copy("*.lib", dst="lib", keep_path=False)
        # self.copy("*.dll", dst="bin", keep_path=False)
        # self.copy("*.dylib*", dst="lib", keep_path=False)
        # self.copy("*.so", dst="lib", keep_path=False)
        # self.copy("*.a", dst="lib", keep_path=False)

    def package_id(self):
        if self.options.header_only:
            self.info.header_only()

    # NOTE: If header only is used, there is no need for a custom
    # package_info() method!
    def package_info(self):
        self.cpp_info.defines.append("ASIO_STANDALONE")
        self.cpp_info.defines.append("ASIO_NO_DEPRECATED")
        if not self.options.header_only:
            self.cpp_info.libs = ["asio"]
