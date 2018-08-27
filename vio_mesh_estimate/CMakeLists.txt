cmake_minimum_required(VERSION 2.8.3)
project(vio_mesh_estimate)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############
cs_add_executable(estimator_node
 src/estimator_node.cpp)

#######
# APP #
#######
#cs_add_executable(rovioli
#  app/rovioli-app.cc
#)
#target_link_libraries(rovioli ${PROJECT_NAME}_lib)
#



############
## EXPORT ##
############
cs_install()
cs_export()
