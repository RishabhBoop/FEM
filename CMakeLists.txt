cmake_minimum_required(VERSION 3.22.1)
project(FEM LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set MKL directory
# set(MKL_DIR /opt/intel/oneapi/mkl/2026.0/lib/cmake/mkl)

# Find the Eigen library
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

# Find Intel MKL
# find_package(MKL CONFIG REQUIRED)

# Set the output directory for the executables and shared libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(fem_1d_app src_cpp/main.cpp src_cpp/FEM_1D.cpp)
add_executable(fem_2d_app src_cpp/main2d.cpp src_cpp/FEM_2D.cpp)

pybind11_add_module(fem_cpp src_cpp/binding.cpp src_cpp/FEM_1D.cpp src_cpp/FEM_2D.cpp)

# Link Eigen to your project
target_link_libraries(fem_1d_app PRIVATE Eigen3::Eigen)
target_link_libraries(fem_2d_app PRIVATE Eigen3::Eigen)
target_link_libraries(fem_cpp PRIVATE Eigen3::Eigen)

# # Enable Eigen to use MKL
# target_compile_definitions(fem_app PRIVATE EIGEN_USE_MKL_ALL)
# target_compile_definitions(fem_cpp PRIVATE EIGEN_USE_MKL_ALL)

# install for python
install(TARGETS fem_cpp DESTINATION .)