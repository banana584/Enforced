cmake_minimum_required(VERSION 3.15)

project(Enforced
    VERSION 1.0.0
    LANGUAGES CXX
)

# Option to build shared or static library
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)  # static libs (.a/.lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)  # shared libs (.so/.dll/.dylib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)  # executables (if any)

# Library target
add_library(${CMAKE_PROJECT_NAME}
    # Source files
    src/utils/utils.cpp
    src/model/model.cpp
    src/reinforcement/reinforcement.cpp
)

# Public include directory
target_include_directories(${CMAKE_PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Optional: warnings
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Optional: set library versioning (useful for shared libs)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

