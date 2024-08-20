if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_INSTALL_INCLUDEDIR "include/asio-${PROJECT_VERSION}" CACHE PATH "")
endif()

# Project is configured with no languages, so tell GNUInstallDirs the lib dir
set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "")

include(cmake/AddUninstallTarget.cmake)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package asio)

#NO! install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT asio_Development)

if(TARGET asio_header)
  install(TARGETS asio_header EXPORT asioTargets FILE_SET HEADERS)
endif()
if(TARGET asio)
  install(TARGETS asio EXPORT asioTargets FILE_SET public_headers)
endif()

write_basic_package_version_file("${package}ConfigVersion.cmake" COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)

# Allow package maintainers to freely override the path for the configs
set(asio_INSTALL_CMAKEDIR "${CMAKE_INSTALL_DATADIR}/${package}"
    CACHE PATH "CMake package config location relative to the install prefix"
)
mark_as_advanced(asio_INSTALL_CMAKEDIR)

install(FILES cmake/install-config.cmake DESTINATION "${asio_INSTALL_CMAKEDIR}" RENAME "${package}Config.cmake"
        COMPONENT asio_Development
)

install(FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake" DESTINATION "${asio_INSTALL_CMAKEDIR}"
        COMPONENT asio_Development
)

install(EXPORT asioTargets NAMESPACE asio:: DESTINATION "${asio_INSTALL_CMAKEDIR}" COMPONENT asio_Development)

if(PROJECT_IS_TOP_LEVEL)
  set(CPACK_GENERATOR TGZ)
  include(CPack)
endif()
