cmake_minimum_required(VERSION 3.22.1)
project(FEM_1D_Project LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Eigen library
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

# Set the output directory for the executables and shared libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(fem_app src_cpp/main.cpp src_cpp/FEM_1D.cpp)

pybind11_add_module(fem_cpp src_cpp/binding.cpp src_cpp/FEM_1D.cpp)

# Link Eigen to your project
target_link_libraries(fem_app PRIVATE Eigen3::Eigen)
target_link_libraries(fem_cpp PRIVATE Eigen3::Eigen)