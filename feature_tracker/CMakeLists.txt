cmake_minimum_required(VERSION 2.8.3)
project(feature_tracker)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME} 
  src/feature-detection-extraction.cc
  src/feature-track-extractor.cc
  src/feature-tracking-pipeline.cc
  src/feature-tracking-types.cc
  src/vo-feature-tracking-pipeline.cc)


cs_install()
cs_export()
