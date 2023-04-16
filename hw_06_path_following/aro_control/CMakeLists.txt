cmake_minimum_required(VERSION 2.8.3)
project(aro_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  actionlib_msgs
  actionlib
  message_generation
  aro_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
    scripts/path_follower.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
