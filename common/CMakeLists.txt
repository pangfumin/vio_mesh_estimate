cmake_minimum_required(VERSION 2.8.3)
project(common)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

include(cmake/maplab_common-extras.cmake.in)


#############
# LIBRARIES #
#############
find_package (Threads)
cs_add_library(${PROJECT_NAME} src/condition.cc
                               src/progress-bar.cc
                               src/threading-helpers.cc
                               src/ImuFrameSynchronizer.cpp
                                src/Time/Duration.cpp
                                src/Time/Time.cpp
                              )
target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT}  readline)

############
## EXPORT ##
############
cs_install()
cs_export(CFG_EXTRAS maplab_common-extras.cmake)
