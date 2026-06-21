cmake_minimum_required(VERSION 3.22.1)
project(FEM LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)

# --- Optional MKL support ––––––––––––––––––––––––––––––––––––––––––––––––––––––
set(MKL_INTERFACE_FULL "intel_lp64" CACHE STRING "MKL Interface" FORCE)
find_package(MKL CONFIG)
if (NOT MKL_FOUND)
    message(WARNING "MKL not found. MKL targets will be skipped.")
endif()

# --- Python and nanobind –––––––––––––––––––––––––––––––––––––––––––––––––––––––
if (CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()
find_package(Python 3.9 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# --- Eigen3 and nanobind –––––––––––––––––––––––––––––––––––––––––––––––––––––––
find_package(Eigen3 QUIET)

if (NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found system-wide, fetching from GitLab...")
    
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG          3.4.0
        SYSTEM # Optional: Treats Eigen headers as system headers to suppress warnings
    )
    FetchContent_MakeAvailable(eigen)
    
    # 3. Create an alias so your target_link_libraries remains identical 
    # whether Eigen was found locally or fetched.
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen ALIAS eigen)
    endif()
endif()
message(STATUS "Found Eigen3 at: ${Eigen3_DIR}")

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
if (nanobind_ROOT)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${nanobind_ROOT})
else()
    message(FATAL_ERROR "nanobind not found.")
endif()
find_package(nanobind CONFIG REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# --- Configure a target with Eigen and optionally MKL –––––––––––––––––––––––––––
# Usage: configure_eigen_target(<target> [MKL])
function(configure_eigen_target TARGET)
    target_link_libraries(${TARGET} PRIVATE Eigen3::Eigen)
    if ("MKL" IN_LIST ARGN)
        if (MKL_FOUND)
            target_compile_definitions(${TARGET} PRIVATE USE_MKL)
            target_link_libraries(${TARGET} PRIVATE MKL::MKL)
        else()
            message(WARNING "MKL requested for '${TARGET}' but not found.")
        endif()
    endif()
endfunction()

# --- Build targets; No MKL ---
add_executable(fem_1d_app src_cpp/main.cpp   src_cpp/FEM_1D.cpp)
add_executable(fem_2d_app src_cpp/main2d.cpp src_cpp/FEM_2D.cpp)
nanobind_add_module(
  fem_cpp 
  src_cpp/nano_binding.cpp 
  src_cpp/FEM_1D.cpp 
  src_cpp/FEM_2D.cpp
)

configure_eigen_target(fem_1d_app)
configure_eigen_target(fem_2d_app)
configure_eigen_target(fem_cpp)

install(TARGETS fem_cpp DESTINATION ".")
nanobind_add_stub(
  fem_cpp_stub 
  MODULE fem_cpp 
  OUTPUT fem_cpp.pyi
  PYTHON_PATH "."
  DEPENDS fem_cpp
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fem_cpp.pyi DESTINATION .)

# --- Build targets; MKL variants ---
option(BUILD_MKL_TARGETS "Build the MKL-accelerated variants" ON)

if (BUILD_MKL_TARGETS)
    add_executable(fem_1d_app_mkl src_cpp/main.cpp   src_cpp/FEM_1D.cpp)
    add_executable(fem_2d_app_mkl src_cpp/main2d.cpp src_cpp/FEM_2D.cpp)
    nanobind_add_module(fem_cpp_mkl src_cpp/nano_binding.cpp src_cpp/FEM_1D.cpp src_cpp/FEM_2D.cpp)

    configure_eigen_target(fem_1d_app_mkl MKL)
    configure_eigen_target(fem_2d_app_mkl MKL)
    configure_eigen_target(fem_cpp_mkl    MKL)
    target_compile_definitions(fem_cpp_mkl PRIVATE MODULE_NAME=fem_cpp_mkl)

    install(TARGETS fem_cpp_mkl DESTINATION "." OPTIONAL)
    nanobind_add_stub(
      fem_cpp_mkl_stub 
      MODULE fem_cpp_mkl 
      OUTPUT fem_cpp_mkl.pyi
      PYTHON_PATH "."
      DEPENDS fem_cpp_mkl
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fem_cpp_mkl.pyi DESTINATION . OPTIONAL)
endif()

