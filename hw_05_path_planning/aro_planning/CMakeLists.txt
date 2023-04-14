cmake_minimum_required(VERSION 2.8.3)
project(aro_planning)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  message_generation
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
    scripts/planner.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
