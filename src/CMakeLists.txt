cmake_minimum_required (VERSION 2.8)
find_package(CUDA REQUIRED)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(CUDA_VERSION_STRING VERSION_EQUAL 8.0 OR CUDA_VERSION_STRING VERSION_GREATER 8.0)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 6.0)
      message(FATAL_ERROR "CUDA 8.0 does not support gcc6+. Please choose an older gcc version like:\n
cmake -DCMAKE_C_COMPILER=`which gcc-5` -DCMAKE_CXX_COMPILER=`which g++-5` ..")
    endif()
  elseif(CUDA_VERSION_STRING VERSION_EQUAL 7.5 OR CUDA_VERSION_STRING VERSION_GREATER 7.5)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
      message(FATAL_ERROR "CUDA 7.5 does not support gcc5+. Please choose an older gcc version like:\n
cmake -DCMAKE_C_COMPILER=`which gcc-4` -DCMAKE_CXX_COMPILER=`which g++-4` ..")
    endif()
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.0)
    message(FATAL_ERROR "gcc4.0 or newer required.")
  endif()
endif()

cuda_add_executable(tlb-bench tlb_benchmark.cu)
cuda_add_executable(tlb-sharing tlb_sharing.cu)
