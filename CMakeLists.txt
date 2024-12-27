# Specify the minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Project name and the language used
project(SNNLIB LANGUAGES CXX)


# Set the C++ compiler explicitly (optional if using clang++)
set(CMAKE_CXX_COMPILER clang++)

# Define build types and set default flags
set(CMAKE_BUILD_TYPE Debug) # Options: Debug, Release, RelWithDebInfo, MinSizeRel
set(SANITIZE_ADDRESS OFF)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Directory paths
set(SRC_DIR src)
set(INCLUDE_DIR include)
set(BUILD_DIR ${CMAKE_BINARY_DIR})
set(OBJ_DIR ${BUILD_DIR}/obj)
set(BIN_DIR ${BUILD_DIR}/bin)
set(LIB_DIR ${BUILD_DIR}/lib)

file(GLOB_RECURSE SOURCES "src/**/*.cpp")

# Include directories
include_directories(${INCLUDE_DIR})

# Find all source files in the SRC_DIR

# Define the output for the static library
set(STATIC_LIB ${LIB_DIR}/libshared.a)

# Define the main executable target
set(SNN_MAIN_SRC ${SRC_DIR}/snn-main.cpp)
set(SNN_MAIN_TARGET ${BIN_DIR}/snn-main)

# Create directories for outputs
file(MAKE_DIRECTORY ${LIB_DIR} ${BIN_DIR})

find_package(yaml-cpp REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add a static library target
add_library(shared STATIC ${SOURCES})
set_target_properties(shared PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${LIB_DIR}
)

# Add OpenMP to the target if available
if(OpenMP_CXX_FOUND)
    target_compile_options(shared PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(shared PRIVATE OpenMP::OpenMP_CXX)
endif()

# Add the executable target
add_executable(snn-main src/snn-main.cpp ${SOURCES})
target_link_libraries(snn-main PRIVATE shared)
target_link_libraries(snn-main PRIVATE yaml-cpp)
if (SANITIZE_ADDRESS)
    target_compile_options(snn-main PRIVATE -fsanitize=address)
    target_link_options(snn-main PRIVATE -fsanitize=address)
endif ()


# Set the output directory for the executable
set_target_properties(snn-main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
)

# Custom clean rule
add_custom_target(custom_clean
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${BUILD_DIR}
    COMMENT "Cleaning build directory"
)