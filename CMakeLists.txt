cmake_minimum_required(VERSION 3.12)
project(eagle_mpc VERSION 0.0
                  DESCRIPTION "EagleMPC: MPC library for unmanned aerial manipulators"
                  LANGUAGES CXX)

set(CMAKE_CONFIGURATION_TYPES
    ${CMAKE_BUILD_TYPE}
    CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -DPINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR -DPINOCCHIO_URDFDOM_USE_STD_SHARED_PTR -DBOOST_MPL_LIMIT_LIST_SIZE=30 -DPINOCCHIO_WITH_URDFDOM"
)

set(CMAKE_INSTALL_PREFIX /usr/local)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LIBRARY_DESTINATION "lib")
set(INCLUDE_DESTINATION "include")
set(SHARE_DESTINATION "share/eagle_mpc")

enable_testing()

option(WITH_BINDINGS "Compile bindings" ON)
option(WITH_EXAMPLES "Compile examples" ON)
option(WITH_BENCHMARK "Compile benchmarks" OFF)

find_package(Eigen3 3.3 REQUIRED)
find_package(eigenpy REQUIRED)
find_package(pinocchio REQUIRED)
find_package(crocoddyl REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(example-robot-data QUIET)


# # Make use of python.cmake
# include(cmake/python.cmake)

# Specifically include python dirs
set(PYTHON_INCLUDE_DIRS /usr/local/python/3.8.20/lib/python3.8/site-packages/numpy/core/include/)
include_directories(${PYTHON_INCLUDE_DIRS})


set(Python_ADDITIONAL_VERSIONS 3.6 3.7 3.8)

# Find Python3 interpreter, NumPy package, and development files
find_package(Python3 REQUIRED Development Interpreter) 
# find_package(Python3 REQUIRED Development Interpreter NumPy)

set(WHICH_PYTHON3 "python3${Python3_VERSION_MINOR}" CACHE STRING "Python3 version")

# # Check if NumPy was found
# if (Python3_NumPy_FOUND)
#     message(STATUS "NumPy include dirs: ${Python3_NumPy_INCLUDE_DIRS}")
#     include_directories(${Python3_NumPy_INCLUDE_DIRS})  # Add NumPy include directories
# else()
#     message(FATAL_ERROR "NumPy not found. Please ensure that it is installed.")
# endif()

find_package(
  Boost
  COMPONENTS ${WHICH_PYTHON3} filesystem unit_test_framework
  REQUIRED
  )

set(EAGLE_MPC_YAML_DIR "${eagle_mpc_SOURCE_DIR}/yaml")

if(example-robot-data_FOUND)
  # [Boks] set EXAMPLE_ROBOT_DATA_MODEL_DIR
  add_compile_definitions(EXAMPLE_ROBOT_DATA_MODEL_DIR=\"/usr/local/share/example-robot-data/robots\")
  add_compile_definitions(EAGLE_MPC_USE_EXAMPLE_ROBOT_DATA)
else()
  message(STATUS "EXAMPLE ROBOT DATA NOT FOUND")
  # if you don't want to use example-robot-data, place your path to the urdf
  # base folder inside the ""
  set(EAGLE_MPC_ROBOT_DATA_DIR "")
endif()

configure_file(${eagle_mpc_SOURCE_DIR}/config/path.py.in path.py)

# Define level of verbosity
# VERBOSE_LEVEL
# 0: errors messages
# 1: errors & warnings
# 2: errors, warnings & info
#
# VERBOSE_DEBUG: to activate debug messages
set(VERBOSE_LEVEL 2)
set(VERBOSE_DEBUG 1)


add_compile_definitions(VERBOSE_DEBUG=${VERBOSE_DEBUG})
add_compile_definitions(VERBOSE_LEVEL=${VERBOSE_LEVEL})

add_subdirectory(src)
if(WITH_EXAMPLES)
  add_subdirectory(examples)
endif()
if(WITH_BINDINGS)
  add_subdirectory(bindings)
endif()
if(WITH_BENCHMARK)
  add_subdirectory(benchmark)
endif()

# uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${eagle_mpc_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
