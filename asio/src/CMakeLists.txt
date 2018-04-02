# build unit tests
if(ASIO_BUILD_TESTS)
  add_subdirectory(tests/unit)
endif()

# Examples
if(ASIO_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

