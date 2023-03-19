cmake_minimum_required(VERSION 3.0)
project(aro_slam VERSION 2023.0.0)

find_package(catkin REQUIRED)

# For aro_slam package
catkin_python_setup()

catkin_package(
    CATKIN_DEPENDS
        geometry_msgs
        message_runtime
        nav_msgs
        ros_numpy
        rospy
        scan_to_point_cloud
        sensor_msgs
        tf2_ros
        voxel_map
)

catkin_install_python(
    PROGRAMS
        scripts/cloud_filter
        scripts/icp_slam
        scripts/occupancy_mapper
        scripts/tf_metrics
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
