cmake_minimum_required(VERSION 3.8)
project(inspection_control_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AutofocusData.msg"
  "msg/FocusFeature.msg"
  "msg/FocusFeatures.msg"
  "msg/FocusValue.msg"
  "msg/OrientationControlData.msg"
  "msg/VirtualSphere.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

ament_package()
