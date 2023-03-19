cmake_minimum_required(VERSION 3.0.2)
project(aro_localization VERSION 2023.0.0)

find_package(catkin REQUIRED)

# For aro_localization package
catkin_python_setup()

catkin_package()

catkin_install_python(
    PROGRAMS
        scripts/aro_localization
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
