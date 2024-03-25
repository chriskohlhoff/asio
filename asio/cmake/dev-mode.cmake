include(cmake/folders.cmake)

option(BUILD_TESTING "Build the testing tree." ON)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(src/tests)
endif()

option(ENABLE_COVERAGE "Enable coverage support separate from CTest's" OFF)
if(ENABLE_COVERAGE)
  include(cmake/coverage.cmake)
endif()

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
  include(cmake/open-cpp-coverage.cmake OPTIONAL)
endif()

include(cmake/lint-targets.cmake)
include(cmake/spell-targets.cmake)

add_folders(Project)
