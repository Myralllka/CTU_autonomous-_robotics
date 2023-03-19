cmake_minimum_required(VERSION 2.8.3)
project(aro_robocop_goto_as)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  actionlib_msgs
  actionlib
  message_generation
)

add_action_files(
  DIRECTORY action
  FILES GoToPosition.action
)

generate_messages(
  DEPENDENCIES geometry_msgs actionlib_msgs std_msgs  
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    message_runtime
    actionlib_msgs
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
    scripts/robocop.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
