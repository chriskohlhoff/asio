cmake_minimum_required(VERSION 3.5...3.16)

project(asio VERSION 1.28.2 LANGUAGES CXX)

set(AsioRequiredFile ${PROJECT_SOURCE_DIR}/include/asio.hpp)

# Check if the file exists
if(NOT EXISTS ${AsioRequiredFile})
    message(FATAL_ERROR "Error: The required file does not exist: ${AsioRequiredFile}")
endif()

add_library(asio_library INTERFACE)
add_library(Asio ALIAS asio_library)

target_include_directories(asio_library INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

#options
set(BOOST CACHE STRING "location of boost distribution")
if(BOOST STREQUAL "no")
    message(STATUS "STANDALONE is set to true")
    add_compile_definitions(ASIO_STANDALONE)
else()
    if(NOT BOOST STREQUAL "system")
        include_directories(${BOOST})
        link_libraries(${BOOST}/stage/lib)
    endif()
    add_compile_definitions(ASIO_ENABLE_BOOST BOOST_CHRONO_HEADER_ONLY BOOST_CHRONO_DONT_PROVIDE_HYBRID_ERROR_HANDLING)
endif()

option(SEPARATE_COMPILATION "separate compilation of asio source" OFF)
if(SEPARATE_COMPILATION)
    add_compile_definitions(ASIO_SEPARATE_COMPILATION)
endif()


if(BUILD_TESTING)
add_subdirectory(tonyTests)
endif()

# Allow the library to be installed via "make install" and found with "find_package"
set(CMAKE_INSTALL_PREFIX "/Users/tonybishara/cmake")
include(GNUInstallDirs)
#1
install(TARGETS asio_library
        EXPORT ${PROJECT_NAME}Targets)
#2
install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} 
        COMPONENT headers
        FILES_MATCHING
        REGEX ".*\\.(hpp|ipp)$")
#3
install(EXPORT ${PROJECT_NAME}Targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME} NAMESPACE Asio::)

include(CMakePackageConfigHelpers)
#4
write_basic_package_version_file(
    cmake/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)
#5
configure_package_config_file(
    cmake/${PROJECT_NAME}Config.cmake.in
    cmake/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
#6
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
#####################################################################################################################################
