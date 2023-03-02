cmake_minimum_required(VERSION 2.8.3)
project(aro_sim)

find_package(catkin REQUIRED)

catkin_package(
    CATKIN_DEPENDS
        joystick_drivers
        teleop_twist_joy
        turtlebot3
        turtlebot3_simulations
)
