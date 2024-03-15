# Testing asio lib on QNX

**NOTE**: QNX ports are only supported from a **Linux host** operating system

asio lib normally wants to be tested on the same machine it was built on. This obviously doesn't work when cross-compiling for QNX. The gist is to build, then copy the whole asio lib source tree on a target. This will include all the relevant files and directory structure which asio lib expects when running its test suite.

# Running the Test Suite

### Install dependencies

`sudo apt install automake`

`sudo apt install pkg-config`

### Switch to asio main folder

`cd asio`

### Generate GNU build tool ./configure and all needed Makefiles

`./autogen.sh`

### Setup QNX SDP environment

`source <path-to-sdp>/qnxsdp-env.sh`

### Build and install all asio tests into SDP

`JLEVEL=$(nproc) CPULIST=x86_64 make -C qnx/build check`

### Then build your QNX image using mkqnximage and the following options:

`export ASIO_ROOT=$PWD`

`mkdir test_image && cd test_image`

`mkqnximage --extra-dirs=$ASIO_ROOT/qnx/test/mkqnximage --clean --run --force --test-asio=$QNX_TARGET/x86_64/usr/bin/asio_tests`

### Once the target has booted, the asio tests will be located in /data/asio:

`cd /data/asio`

`./run_testsuites.sh`

### Test execution summary
`...`

`=========================================================================`

`Testsuite summary for asio 1.29.0`

`=========================================================================`

`# TOTAL: 346`

`# PASS: 346`

`# FAIL: 0`

`=========================================================================`