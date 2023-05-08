CMAKE_MINIMUM_REQUIRED( VERSION 3.3.0 )

project (GGNN)
set(CMAKE_CXX_STANDARD 14)


find_package(CUDA REQUIRED)

set(CUDA_ARCH_LIST Auto)
# set(CUDA_ARCH_LIST "7.0")
# set(CUDA_ARCH_LIST "7.5 6.1 8.6")
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})

set(CUB_INCLUDE_PATH "3rdparty/cub")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -Xptxas=-v --expt-relaxed-constexpr --default-stream per-thread" )
# debug
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -g")
# profile
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")

add_subdirectory("3rdparty/pybind11")
set(GFLAGS_NAMESPACE "google")
add_subdirectory("3rdparty/glog")

message("CUDA_ARCH_LIST: ${CUDA_ARCH_LIST}")
message("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

# configurations for specific datasets
# TODO: generate from template using CMake?

cuda_add_executable(heavy src/heavy.cu)
target_include_directories(heavy
  PRIVATE
  ${CUDA_NVCC_FLAGS}
  ${CUB_INCLUDE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(heavy ${CUDA_curand_LIBRARY} glog::glog)

cuda_add_executable(light src/light.cu)
target_include_directories(light
  PRIVATE
  ${CUDA_NVCC_FLAGS}
  ${CUB_INCLUDE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(light ${CUDA_curand_LIBRARY} glog::glog)

