Claus-MBP:test_package clausklein$ cp ../../asio/src/examples/cpp14/executors/priority_scheduler.cpp example.cpp

Claus-MBP:test_package clausklein$ ls -1
CMakeLists.txt
build
conanfile.py
example.cpp

Claus-MBP:test_package clausklein$ mkdir build && cd build

Claus-MBP:build clausklein$ conan install .. --build=missing
PROJECT: Installing /Users/clausklein/Workspace/cpp/asio/conan/test_package/conanfile.py
Requirements
    OpenSSL/1.0.2n@conan/stable from 'conan-center'
    asio/1.12.0@demo/testing from local cache
    zlib/1.2.11@conan/stable from 'conan-center'
Packages
    OpenSSL/1.0.2n@conan/stable:811d822905b54fc167634e916129401c4f86d1e5
    asio/1.12.0@demo/testing:7ee503fc205a7c0f4d223770041ad21111402674
    zlib/1.2.11@conan/stable:dfaeed675a0f450dbc88fe8262d9d89b3e8509b0

zlib/1.2.11@conan/stable: Already installed!
OpenSSL/1.0.2n@conan/stable: Already installed!
asio/1.12.0@demo/testing: Already installed!
PROJECT: Generator cmake created conanbuildinfo.cmake
PROJECT: Generator txt created conanbuildinfo.txt
PROJECT: Generated conaninfo.txt

Claus-MBP:build clausklein$ conan build ..
Project: Running build()

----Running------
> cd '/Users/clausklein/Workspace/cpp/asio/conan/test_package/build' && cmake -G "Ninja" -DCMAKE_BUILD_TYPE="Release" -DCONAN_EXPORTED="1" -DCONAN_COMPILER="apple-clang" -DCONAN_COMPILER_VERSION="8.0" -DCONAN_CXX_FLAGS="-m64" -DCONAN_SHARED_LINKER_FLAGS="-m64" -DCONAN_C_FLAGS="-m64" -DCONAN_LIBCXX="libc++" -DCMAKE_INSTALL_PREFIX="/Users/clausklein/Workspace/cpp/asio/conan/test_package/build/package" -Wno-dev '/Users/clausklein/Workspace/cpp/asio/conan/test_package'
-----------------
Logging command output to file '/Users/clausklein/Workspace/cpp/asio/conan/test_package/build/conan_run.log'
-- The CXX compiler identification is AppleClang 8.0.0.8000042
-- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
-- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Conan: called by CMake conan helper
-- Conan: Using cmake targets configuration
-- Library asio found /Users/clausklein/.conan/data/asio/1.12.0/demo/testing/package/7ee503fc205a7c0f4d223770041ad21111402674/lib/libasio.a
-- Library ssl found /Users/clausklein/.conan/data/OpenSSL/1.0.2n/conan/stable/package/811d822905b54fc167634e916129401c4f86d1e5/lib/libssl.a
-- Library crypto found /Users/clausklein/.conan/data/OpenSSL/1.0.2n/conan/stable/package/811d822905b54fc167634e916129401c4f86d1e5/lib/libcrypto.a
-- Library z found /Users/clausklein/.conan/data/zlib/1.2.11/conan/stable/package/dfaeed675a0f450dbc88fe8262d9d89b3e8509b0/lib/libz.a
-- Conan: Adjusting default RPATHs Conan policies
-- Conan: Adjusting language standard
-- Conan: C++ stdlib: libc++
-- Configuring done
-- Generating done
-- Build files have been written to: /Users/clausklein/Workspace/cpp/asio/conan/test_package/build

----Running------
> cmake --build '/Users/clausklein/Workspace/cpp/asio/conan/test_package/build'
-----------------
Logging command output to file '/Users/clausklein/Workspace/cpp/asio/conan/test_package/build/conan_run.log'
[0/1] Re-running CMake...
-- Conan: called by CMake conan helper
-- Conan: Using cmake targets configuration
-- Library asio found /Users/clausklein/.conan/data/asio/1.12.0/demo/testing/package/7ee503fc205a7c0f4d223770041ad21111402674/lib/libasio.a
-- Library ssl found /Users/clausklein/.conan/data/OpenSSL/1.0.2n/conan/stable/package/811d822905b54fc167634e916129401c4f86d1e5/lib/libssl.a
-- Library crypto found /Users/clausklein/.conan/data/OpenSSL/1.0.2n/conan/stable/package/811d822905b54fc167634e916129401c4f86d1e5/lib/libcrypto.a
-- Library z found /Users/clausklein/.conan/data/zlib/1.2.11/conan/stable/package/dfaeed675a0f450dbc88fe8262d9d89b3e8509b0/lib/libz.a
-- Conan: Adjusting default RPATHs Conan policies
-- Conan: Adjusting language standard
-- Conan: C++ stdlib: libc++
-- Configuring done
-- Generating done
-- Build files have been written to: /Users/clausklein/Workspace/cpp/asio/conan/test_package/build
[1/2] Building CXX object CMakeFiles/example.dir/example.cpp.o
[2/2] Linking CXX executable bin/example

----Running------
> cmake --build '/Users/clausklein/Workspace/cpp/asio/conan/test_package/build' '--target' 'test'
-----------------
Logging command output to file '/Users/clausklein/Workspace/cpp/asio/conan/test_package/build/conan_run.log'
[0/1] Running tests...
Test project /Users/clausklein/Workspace/cpp/asio/conan/test_package/build
    Start 1: example
1/1 Test #1: example ..........................   Passed    0.01 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =   0.02 sec


Claus-MBP:build clausklein$ ls -1
CMakeCache.txt
CMakeFiles
CTestTestfile.cmake
Testing
bin
build.ninja
cmake_install.cmake
conan_run.log
conanbuildinfo.cmake
conanbuildinfo.txt
conaninfo.txt
rules.ninja
Claus-MBP:build clausklein$ 

