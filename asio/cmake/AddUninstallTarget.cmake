# SPDX-FileCopyrightText: 2012-2021 Istituto Italiano di Tecnologia (IIT)
# SPDX-FileCopyrightText: 2008-2013 Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

#[=======================================================================[.rst:
AddUninstallTarget
------------------

Add the "uninstall" target for your project::

  include(AddUninstallTarget)


will create a file ``cmake_uninstall.cmake`` in the build directory and add a
custom target ``uninstall`` (or ``UNINSTALL`` on Visual Studio and Xcode) that
will remove the files installed by your package (using
``install_manifest.txt``).
See also
https://gitlab.kitware.com/cmake/community/wikis/FAQ#can-i-do-make-uninstall-with-cmake

The :module:`AddUninstallTarget` module must be included in your main
``CMakeLists.txt``. If included in a subdirectory it does nothing.
This allows you to use it safely in your main ``CMakeLists.txt`` and include
your project using ``add_subdirectory`` (for example when using it with
:cmake:module:`FetchContent`).

If the ``uninstall`` target already exists, the module does nothing.
#]=======================================================================]

# AddUninstallTarget works only when included in the main CMakeLists.txt
if(NOT "${CMAKE_CURRENT_BINARY_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  return()
endif()

# The name of the target is uppercase in MSVC and Xcode (for coherence with the
# other standard targets)
if("${CMAKE_GENERATOR}" MATCHES "^(Visual Studio|Xcode)")
  set(_uninstall "UNINSTALL")
else()
  set(_uninstall "uninstall")
endif()

# If target is already defined don't do anything
if(TARGET ${_uninstall})
  return()
endif()

set(_filename cmake_uninstall.cmake)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${_filename}"
     "if(NOT EXISTS \"${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt\")
  message(WARNING \"Cannot find install manifest: \\\"${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt\\\"\")
  return()
endif()

file(READ \"${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt\" files)
string(STRIP \"\${files}\" files)
string(REGEX REPLACE \"\\n\" \";\" files \"\${files}\")
list(REVERSE files)
foreach(file \${files})
  if(IS_SYMLINK \"\$ENV{DESTDIR}\${file}\" OR EXISTS \"\$ENV{DESTDIR}\${file}\")
    message(STATUS \"Uninstalling: \$ENV{DESTDIR}\${file}\")
    execute_process(
      COMMAND \${CMAKE_COMMAND} -E remove \"\$ENV{DESTDIR}\${file}\"
      OUTPUT_VARIABLE rm_out
      RESULT_VARIABLE rm_retval)
    if(NOT \"\${rm_retval}\" EQUAL 0)
      message(FATAL_ERROR \"Problem when removing \\\"\$ENV{DESTDIR}\${file}\\\"\")
    endif()
  else()
    message(STATUS \"Not-found: \$ENV{DESTDIR}\${file}\")
  endif()
endforeach(file)
"
)

set(_desc "Uninstall the project...")
if(CMAKE_GENERATOR STREQUAL "Unix Makefiles")
  set(_comment COMMAND \$\(CMAKE_COMMAND\) -E cmake_echo_color --switch=$\(COLOR\) --cyan "${_desc}")
else()
  set(_comment COMMENT "${_desc}")
endif()
add_custom_target(
  ${_uninstall}
  ${_comment}
  COMMAND ${CMAKE_COMMAND} -P ${_filename}
  USES_TERMINAL
  BYPRODUCTS uninstall_byproduct
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
set_property(SOURCE uninstall_byproduct PROPERTY SYMBOLIC 1)

set_property(TARGET ${_uninstall} PROPERTY FOLDER "CMakePredefinedTargets")
