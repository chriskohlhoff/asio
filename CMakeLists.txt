# Copyright (c) 2026 Narta Xaymar Dirks (info at xaymar dot com)
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Require the latest CMake version.
cmake_minimum_required(VERSION 4.2.1)

project(asio)

# Is this the new or old layout?
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
	# New Layout
	set(_src "${CMAKE_CURRENT_SOURCE_DIR}")
else()
	# Old Layout
	set(_src "${CMAKE_CURRENT_SOURCE_DIR}/asio")
endif()

# Figure out the version from the provided version.hpp.
if(EXISTS "${_src}/include/asio/version.hpp")
	file(READ "${_src}/include/asio/version.hpp" ASIO_VERSION_HPP)
	string(REGEX MATCH "#define ASIO_VERSION ([0-9]+) // ([0-9]+)\.([0-9]+)\.([0-9]+)" ASIO_VERSION "${ASIO_VERSION_HPP}")
	set(ASIO_VERSION_INT ${CMAKE_MATCH_1})
	set(ASIO_VERSION_MAJOR ${CMAKE_MATCH_2}) # Trust the comment for now.
	set(ASIO_VERSION_MINOR ${CMAKE_MATCH_3})
	set(ASIO_VERSION_PATCH ${CMAKE_MATCH_4})
	project(${PROJECT_NAME} VERSION "${ASIO_VERSION_MAJOR}.${ASIO_VERSION_MINOR}.${ASIO_VERSION_PATCH}")
endif()


#--------------------------------------------------------------------------------#
# Options
#--------------------------------------------------------------------------------#

# What options?

#--------------------------------------------------------------------------------#
# Library
#--------------------------------------------------------------------------------#

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME}
	INTERFACE
		$<BUILD_INTERFACE:${_src}/include/>
		$<INSTALL_INTERFACE:include/>
)

target_compile_definitions(${PROJECT_NAME}
	INTERFACE
		ASIO_STANDALONE
)

set_target_properties(${PROJECT_NAME} PROPERTIES
	# Only necessary for source builds, which only work on MSVC anyway.
	CXX_STANDARD 11
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
)

#--------------------------------------------------------------------------------#
# Install
#--------------------------------------------------------------------------------#

install(TARGETS ${PROJECT_NAME}
	EXPORT ${PROJECT_NAME}-targets
	LIBRARY RUNTIME FRAMEWORK
	PERMISSIONS OWNER_WRITE GROUP_WRITE OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
)
install(
	DIRECTORY "${_src}/include/"
	TYPE INCLUDE
	FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ GROUP_WRITE WORLD_READ
	DIRECTORY_PERMISSIONS OWNER_WRITE GROUP_WRITE OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
	FILES_MATCHING
		REGEX ".*\.hpp$"
		REGEX ".*\.ipp$"
)
install(
	FILES
		"${CMAKE_CURRENT_BINARY_DIR}/asio-config.cmake"
		"${CMAKE_CURRENT_BINARY_DIR}/asio-config-version.cmake"
	DESTINATION "share/cmake/asio/"
)

#--------------------------------------------------------------------------------#
# CMake Config
#--------------------------------------------------------------------------------#

include(CMakePackageConfigHelpers)
configure_package_config_file(
	"cmake/config.cmake.in"
	"${PROJECT_NAME}-config.cmake"
	INSTALL_DESTINATION "share/cmake/asio/"
)
write_basic_package_version_file(
	"${PROJECT_NAME}-config-version.cmake"
	COMPATIBILITY SameMinorVersion
	ARCH_INDEPENDENT
)

# Generate asio-targets.cmake and asio-targets${CMAKE_BUILD_TYPE}.cmake files.
export(EXPORT ${PROJECT_NAME}-targets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-targets.cmake"
	NAMESPACE ${PROJECT_NAME}::
)
install(EXPORT ${PROJECT_NAME}-targets
	DESTINATION "share/cmake/asio/"
	NAMESPACE ${PROJECT_NAME}::
)

