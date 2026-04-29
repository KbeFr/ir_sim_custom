# Heterogeneous Digital Twin (HDT) Mission Planning Architecture

Before implementing complex, multi-agent robotic systems with real-world components, the viability and advantages of mission planning and Heterogeneous Digital Twin (HDT) architectures must be tested. Simulation is a highly valuable tool for this purpose. 

This project wraps around and extends [ir-sim](https://github.com/hanruihua/ir-sim) (MIT License, Copyright (c) 2022 Ruihua Han), a relatively lightweight and simple Python-based "YAML-driven robot simulator for navigation, control, and learning". Because `ir-sim` is built on object-oriented Python architecture, altering minor functionalities to implement the HDT was straightforward and highly extensible.

The entire environment and all its components can be defined in a prewritten YAML file and loaded on startup. This allows for easy configurability and rapid testing of different scenarios and configurations.

## Architecture & Components

To implement the HDT, several new components were created inside the `ir-sim` project itself, while the overarching logic was defined externally. 

### 1. Simulated Physical Layer (`ir-sim` modifications)
* **UAV & UGV Twins:** Custom digital twin wrappers that bridge the physical simulation states for aerial and ground vehicles.
* **Custom Sensors:** We implemented two new sensors representing monocular cameras:
  * **UGV Camera:** A pie-shaped forward detection zone.
  * **UAV Camera:** A rectangular, top-down coverage window around the drone.

### 2. The Overarching Twin Layer
Defined outside the core simulator, the `OverArchingTwin` layer orchestrates the global logic:
* **Global Grid Map & Rasterizer:** Dynamically builds a static occupancy grid from `ir-sim` obstacle geometry and computes per-cell traversal costs based on risk, energy, time, and uncertainty.
* **Mission Planner:** Handles task assignments (e.g., GOTO, TRACK, PATROL) optimally to available UGVs, respecting battery budgets and posture weights.
* **UAV Fleet DT:** Aggregates obstacle detections and coverage footprints from all active UAVs to dynamically reduce map uncertainty for ground units.
* **A* Global Planner:** A modified A* algorithm adapted from previous projects to accept our custom, dynamically updating cost map.

### 3. Local Control Layer
We reused and integrated several local controllers for the ground robots:
* **Collision Cone CBF (C3BF)**
* **CBF QP Controller**
* **Pure Pursuit Controller**

By integrating all of these components, we can directly observe the variance and performance advantages of using a full HDT architecture (cooperative UAV + UGV) over a standard UGV-only sensor approach.

## Project Setup

1. **Install Base Simulator:** Ensure you have the required dependencies for `ir-sim` v2.9.2.
2. **Apply Core Modifications:** For the simulator to recognize the HDT components, you must migrate the files located in `project_kobe_frateur/custom_components` into the core `irsim` directory. 
   * 👉 **Please follow the step-by-step instructions in [changes_irsim.md](project_kobe_frateur/custom_components/changes_irsim.md)**.

## Usage

You can easily configure environments, obstacles, and the robot fleet via YAML files (e.g., `custom_world.yaml` .

To run the main simulation:
```bash
python custom_world.py
```

### Configuration Options
Inside `custom_world.py`, you can tweak the following global parameters before running:
* `CONTROLLER`: Choose between `"c3bf"`, `"cbf"`, or `"pure_pursuit"`.
* `USE_GLOBAL_PLAN`: Set to `True` to use the Overarching Twin's mission planner, or `False` to let the UGV navigate directly to its YAML-defined goal.
* `PERCEPTION_MODE`: Toggle between `"all"`, `"uav"`, `"ugv"`, or `"merged"`.
* `MAX_STEPS`: Total simulation steps to run.

## Metrics & Logging

This framework is designed for architectural testing and thesis validation, and includes a robust logging pipeline (`MetricsLogger` and `MissionLogger`).

As the simulation runs, it records per-step data for every UGV. Once the simulation concludes, it automatically generates publication-ready matplotlib figures (saved as PDFs) to visualize:
* Cumulative Distance & Velocity over time
* Battery State of Charge & Energy Consumption
* Position Uncertainty (illustrating the benefit of UAV coverage)
* Cost Function Breakdown
* Final trajectory maps colored by speed