# Specify the minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Project name and the language used
project(SNNMotion LANGUAGES CXX)

# Set the C++ compiler explicitly (optional if using clang++)
set(CMAKE_CXX_COMPILER clang++)

# Define build types and set default flags
set(CMAKE_BUILD_TYPE Debug) # Options: Debug, Release, RelWithDebInfo, MinSizeRel
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# Directory paths
set(SRC_DIR src)
set(INCLUDE_DIR include)
set(BUILD_DIR ${CMAKE_BINARY_DIR})
set(OBJ_DIR ${BUILD_DIR}/obj)
set(BIN_DIR ${BUILD_DIR}/bin)
set(LIB_DIR ${BUILD_DIR}/lib)

# Include directories
include_directories(${INCLUDE_DIR})

# Find all source files in the SRC_DIR
file(GLOB SRC_FILES "${SRC_DIR}/*.cpp")

# Define the output for the static library
set(STATIC_LIB ${LIB_DIR}/libshared.a)

# Define the main executable target
set(SNN_MOTION_MAIN_SRC ${SRC_DIR}/snn-motion-main.cpp)
set(SNN_MOTION_MAIN_TARGET ${BIN_DIR}/snn-motion-main)

# Create directories for outputs
file(MAKE_DIRECTORY ${LIB_DIR} ${BIN_DIR})

# Add a static library target
add_library(shared STATIC ${SRC_FILES})
set_target_properties(shared PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${LIB_DIR}
)

# Add the executable target
add_executable(snn-motion-main ${SNN_MOTION_MAIN_SRC})
target_link_libraries(snn-motion-main PRIVATE shared)

# Set the output directory for the executable
set_target_properties(snn-motion-main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
)

# Custom clean rule
add_custom_target(custom_clean
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${BUILD_DIR}
    COMMENT "Cleaning build directory"
)