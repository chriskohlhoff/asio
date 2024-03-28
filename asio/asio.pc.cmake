prefix=@CMAKE_INSTALL_PREFIX@
exec_prefix=@CMAKE_INSTALL_PREFIX@/bin
includedir=@CMAKE_INSTALL_FULL_INCLUDEDIR@

Name: @PROJECT_NAME@
Description: A cross-platform C++ library for network and low-level I/O programming that provides developers with a consistent asynchronous model using a modern C++ approach.
Version: @PROJECT_VERSION@
Cflags: -I${CMAKE_INSTALL_FULL_INCLUDEDIR} -DASIO_NO_DEPRECATED
Lflags:
Requires:
Requires.private:
