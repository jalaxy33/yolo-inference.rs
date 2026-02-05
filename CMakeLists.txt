# =============================================================================
# Preamble
# =============================================================================

cmake_minimum_required(VERSION 4.0.0)

# generate `compile_commands.json` for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# Project Configuration
# =============================================================================

project(yolo-inference VERSION 0.1.1 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)


# =============================================================================
# Rust Library Configuration
# =============================================================================

function(setup_rust_library targetname rustlibname)
    # Fix Windows MSVC runtime library compatibility
    # Always use /MD (Release CRT) to match Rust and vcpkg Release libraries
    if(WIN32 AND MSVC)
        target_compile_options(${targetname} PRIVATE /MD)
        # Disable iterator debug level in Debug builds to match Release libraries
        # Use generator expression to work with multi-config generators like Visual Studio
        target_compile_definitions(${targetname} PRIVATE 
            $<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=0>
        )
    endif()

    # Ensure the Rust library is built
    set(RUST_TARGET_DIR ${CMAKE_SOURCE_DIR}/target)
    set(rust_lib_dir ${RUST_TARGET_DIR}/release)   # always use release build for compatibility
    
    if(NOT EXISTS ${rust_lib_dir})
        message(WARNING "Rust library not found. Building Rust library now...")
        execute_process(
            COMMAND cargo build --release --features cpp
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
    endif()

    # Link Rust library
    set(rust_lib_path "${rust_lib_dir}/lib${rustlibname}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if (WIN32)
        set(rust_lib_path "${rust_lib_path}.lib")
    endif()
    target_link_libraries(${targetname} PRIVATE "${rust_lib_path}")

    # Set include directories for cxx generated headers
    file(GLOB cxxbridge_include_dirs
        "${RUST_TARGET_DIR}/cxxbridge/*/src"
        "${RUST_TARGET_DIR}/cxxbridge/*/src/*"
    )
    target_include_directories(${targetname} PRIVATE
        ${RUST_TARGET_DIR}/cxxbridge
        ${RUST_TARGET_DIR}/cxxbridge/rust
        ${cxxbridge_include_dirs}
    )

    # Gather cxxbridge generated source files
    file(GLOB cxxbridge_src
        "${RUST_TARGET_DIR}/cxxbridge/*/src/*.rs.cpp"
        "${RUST_TARGET_DIR}/cxxbridge/*/src/*.rs.cc"
        "${RUST_TARGET_DIR}/cxxbridge/*/src/*/*.rs.cpp"
        "${RUST_TARGET_DIR}/cxxbridge/*/src/*/*.rs.cc"
    )

    if (cxxbridge_src)
        target_sources(${targetname} PRIVATE ${cxxbridge_src})
    endif()

    # link cxx library
    file(GLOB cxx_lib_files
        "${rust_lib_dir}/build/cxx-*/out/*${CMAKE_STATIC_LIBRARY_SUFFIX}"
    )
    foreach(cxx_lib IN LISTS cxx_lib_files)
        target_link_libraries(${targetname} PRIVATE ${cxx_lib})
    endforeach()

    # Copy DLL on Windows
    if(WIN32)
        add_custom_command(TARGET ${targetname} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${rust_lib_files}"
            $<TARGET_FILE_DIR:${targetname}>
            COMMENT "Copying Rust DLL for $<CONFIG>."
        )
    endif()

endfunction()


# =============================================================================
# Target Configuration
# =============================================================================

set(targetname cpp_bin)
add_executable(${targetname} cpp_src/main.cpp)
target_include_directories(${targetname}  PRIVATE cpp_headers/)

setup_rust_library(${targetname} "yolo_inference")


# =============================================================================
# Global Variables
# =============================================================================

# Current working directory
set(PROJECT_ROOT ${CMAKE_SOURCE_DIR})

# Make PROJECT_ROOT visible in C++ code
target_compile_definitions(${targetname} PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")


