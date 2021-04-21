# Find UMPIRE linear solver library
#
# Set UMPIRE_DIR to the base directory where the package is installed
#
# Sets two variables
#   - UMPIRE_INCLUDE_DIRS
#   - UMPIRE_LIBRARIES
#

find_path(UMPIRE_INCLUDE_DIRS
  umpire.h
  HINTS ${UMPIRE_DIR} ${CMAKE_INSTALL_PREFIX}
  PATH_SUFFIXES include/umpire/interface)

find_library(UMPIRE_LIBRARIES
  NAMES umpire umpire
  HINTS ${UMPIRE_DIR} ${CMAKE_INSTALL_PREFIX}
  PATH_SUFFIXES lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Umpire DEFAULT_MSG UMPIRE_INCLUDE_DIRS UMPIRE_LIBRARIES)
mark_as_advanced(UMPIRE_INCLUDE_DIRS UMPIRE_LIBRARIES)

set(UMPIRE_INCLUDE_DIRS  "${UMPIRE_INCLUDE_DIRS}/../../")
