include(CMakeFindDependencyMacro)
find_dependency(Threads)
find_dependency(OpenSSL)

include("${CMAKE_CURRENT_LIST_DIR}/asioTargets.cmake")
