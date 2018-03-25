from conans import ConanFile, CMake, tools


class AsioConan(ConanFile):
    name = "asio"
    version = "1.11.0"
    license = "Boost Software License - Version 1.0"
    url = "https://github.com/chriskohlhoff/asio/issues"
    description = "Asio C++ Library"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "force_openssl": [True, False],
        "cxx_14": [True, False]
    }
    default_options = '''
shared=False
force_openssl=True
cxx_14=True
'''
    generators = "cmake"
    exports_sources = "asio/*"

    def requirements(self):
        if self.options.force_openssl:
            self.requires.add("OpenSSL/1.0.2n@conan/stable", private=False)

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_folder="asio")
        cmake.build()
        if tools.get_env("CONAN_RUN_TESTS", True):
            cmake.test()
        cmake.install()

    def package(self):
        self.copy("*.hpp", dst="include", src="asio/include")
        self.copy("*.ipp", dst="include", src="asio/include")
        self.copy("*.lib", dst="lib", keep_path=False)
        self.copy("*.dll", dst="bin", keep_path=False)
        self.copy("*.dylib*", dst="lib", keep_path=False)
        self.copy("*.so", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["asio"]
