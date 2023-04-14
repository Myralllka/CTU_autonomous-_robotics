cmake_minimum_required(VERSION 2.8.3)
project(aro_msgs)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  actionlib_msgs
  message_generation
)

add_message_files(
  FILES
  Path.msg
)

add_service_files(
  FILES
  GenerateFrontier.srv
  PlanPath.srv
)

add_action_files(
    FILES
    FollowPath.action
)

generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
  geometry_msgs
  actionlib_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

