# Inspection Control: Perception-Driven Admittance-Based Orientation Control

## Overview
This repository presents a perception-driven orientation control framework for robotic inspection tasks. The system integrates real-time point cloud perception, orientation error estimation, and an admittance-based control strategy to enable compliant and stable end-effector alignment with target surfaces.

The framework is designed for human-in-the-loop robotic inspection, where the controller assists operators by autonomously regulating orientation while respecting physical constraints such as torque and velocity limits.

## Key Features
- Perception-driven control using surface normals from point clouds
- Admittance-based outer-loop controller for compliant behavior
- PD-based orientation regulation with physically interpretable dynamics
- Compatible with ROS 2 and MoveIt Servo pipelines
- Real-time integration of perception, control, and execution
- Validated on UR5e manipulator hardware

## System Architecture
The control pipeline consists of the following stages:

```text
Depth Image -> Point Cloud -> Surface Normal Estimation
            -> Orientation Error Computation
            -> PD Controller (Torque Output)
            -> Admittance Dynamics (Virtual Sphere Model)
            -> Velocity Commands -> Robot Execution
