=======================================================
This example is created to test the conan asio package:
=======================================================

This conan package of asio is a header only standalone distribution build with cmake.

    $cp ../asio/src/examples/cpp14/executors/priority_scheduler.cpp test_package/example.cpp


How to use:
===========

see too https://bincrafters.github.io/2018/02/27/Updated-Conan-Package-Flow-1.1/

::

  $conan create . bincrafters/testing
  
  ...
  
  asio/1.12.1@bincrafters/testing: Package '5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9' built
  asio/1.12.1@bincrafters/testing: Build folder /Users/clausklein/.conan/data/asio/1.12.1/bincrafters/testing/build/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9
  asio/1.12.1@bincrafters/testing: Generated conaninfo.txt
  asio/1.12.1@bincrafters/testing: Generated conanbuildinfo.txt
  asio/1.12.1@bincrafters/testing: Generating the package
  asio/1.12.1@bincrafters/testing: Package folder /Users/clausklein/.conan/data/asio/1.12.1/bincrafters/testing/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9
  asio/1.12.1@bincrafters/testing: Calling package()
  asio/1.12.1@bincrafters/testing package(): WARN: No files copied!
  asio/1.12.1@bincrafters/testing package(): WARN: No files copied!
  asio/1.12.1@bincrafters/testing: Package '5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9' created
  asio/1.12.1@bincrafters/testing (test package): Generator cmake created conanbuildinfo.cmake
  asio/1.12.1@bincrafters/testing (test package): Generator txt created conanbuildinfo.txt
  asio/1.12.1@bincrafters/testing (test package): Generated conaninfo.txt
  asio/1.12.1@bincrafters/testing (test package): Running build()
  
  ----Running------
  > cd '/var/folders/wb/ckvxxgls5db7qyhqq4y5_l1c0000gq/T/conansQqfoiE' && cmake -G "Ninja" -DCMAKE_BUILD_TYPE="Release" -DCONAN_EXPORTED="1" -DCONAN_COMPILER="apple-clang" -DCONAN_COMPILER_VERSION="8.0" -DCONAN_CXX_FLAGS="-m64" -DCONAN_SHARED_LINKER_FLAGS="-m64" -DCONAN_C_FLAGS="-m64" -DCONAN_LIBCXX="libc++" -Wno-dev '/Users/clausklein/Workspace/cpp/asio/conan/test_package'
  -----------------
  -- The CXX compiler identification is AppleClang 8.0.0.8000042
  -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
  -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ -- works
  -- Detecting CXX compiler ABI info
  -- Detecting CXX compiler ABI info - done
  -- Detecting CXX compile features
  -- Detecting CXX compile features - done
  -- Conan: called by CMake conan helper
  -- Conan: Using cmake targets configuration
  -- Conan: Adjusting default RPATHs Conan policies
  -- Conan: Adjusting language standard
  -- Conan: C++ stdlib: libc++
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /var/folders/wb/ckvxxgls5db7qyhqq4y5_l1c0000gq/T/conansQqfoiE
  
  ----Running------
  > cmake --build '/var/folders/wb/ckvxxgls5db7qyhqq4y5_l1c0000gq/T/conansQqfoiE'
  -----------------
  [1/2] Building CXX object CMakeFiles/example.dir/example.cpp.o
  [2/2] Linking CXX executable bin/example
  
  ----Running------
  > cmake --build '/var/folders/wb/ckvxxgls5db7qyhqq4y5_l1c0000gq/T/conansQqfoiE' '--target' 'test'
  -----------------
  [0/1] Running tests...
  Test project /var/folders/wb/ckvxxgls5db7qyhqq4y5_l1c0000gq/T/conansQqfoiE
      Start 1: example
  1/1 Test #1: example ..........................   Passed    0.01 sec
  
  100% tests passed, 0 tests failed out of 1
  
  Total Test time (real) =   0.02 sec
  asio/1.12.1@bincrafters/testing (test package): Running test()
  
  ----Running------
  > ./example
  -----------------
  3
  33
  333
  2
  22
  11
  1
  bash-4.4$ 
