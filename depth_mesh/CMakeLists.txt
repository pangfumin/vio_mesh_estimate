cmake_minimum_required(VERSION 2.8.11)
project(depth_mesh)

if(NOT DO_TIMING)
  add_definitions(-DDEACTIVATE_TIMERS )
  message(STATUS "Deactivating timers.")
endif()

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

# Build main library.
set(FLAME_SRCS
        src/mesh_estimator.cpp
        src/flame.cc
        src/optimizers/nltgv2_l1_graph_regularizer.cc
        src/stereo/inverse_depth_filter.cc
        src/stereo/inverse_depth_meas_model.cc
        src/external/triangle/triangle.cpp
        src/utils/delaunay.cc
        src/utils/frame.cc
        src/utils/image_utils.cc
        src/utils/pyramids.cc
        src/utils/rasterization.cc
        )

cs_add_library(depth_mesh  ${FLAME_SRCS})


include_directories(sample)
add_executable(flame_offline_asl
        sample/flame_offline_asl.cc
        sample/utils.h
        sample/utils.cc
        sample/ros_sensor_streams/asl_rgbd_offline_stream.h
        sample/ros_sensor_streams/asl_rgbd_offline_stream.cc
        sample/dataset_utils/asl/dataset.h
        sample/dataset_utils/asl/types.h
        sample/dataset_utils/utils.h
        sample/dataset_utils/utils.cc)

target_link_libraries(flame_offline_asl
        depth_mesh)

cs_install()
cs_export()



