cmake_minimum_required(VERSION 3.0.2)
project(tiago_grasping_on_the_move)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  tf2
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  interactive_markers
)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES tiago_grasping_on_the_move
  CATKIN_DEPENDS
    rospy
    geometry_msgs
    tf2
    moveit_core
    moveit_ros_planning
    moveit_ros_planning_interface
    interactive_markers
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/tiago_grasping_on_the_move
  scripts/tiago_ik
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
