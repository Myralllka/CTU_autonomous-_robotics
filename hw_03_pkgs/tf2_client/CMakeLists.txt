cmake_minimum_required(VERSION 2.8.3)
project(tf2_client)

set(CATKIN_DEPS
    roscpp
    rospy
    tf2
    tf2_ros
)

find_package(catkin REQUIRED COMPONENTS ${CATKIN_DEPS})

catkin_python_setup()

catkin_package(
    CATKIN_DEPENDS ${CATKIN_DEPS}
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
)

include_directories(include ${catkin_INCLUDE_DIRS})

add_library(${PROJECT_NAME} src/tf2_client/tf2_client.cpp)
link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
install(FILES
   launch/server.launch
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
