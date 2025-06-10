# DMP-based Robot Control with Robosuite

This repository contains a Dynamic Motor Primitive (DMP) based policy for robot manipulation tasks, specifically designed for the Nut Assembly Square task in the Robosuite simulation environment.

## Overview

The system uses Dynamic Motor Primitives combined with PID control to enable robots to learn and execute manipulation tasks from demonstrations. The policy can adapt to different object positions by re-targeting the learned trajectories to new goal locations.

### Key Components

- **DMP (Dynamic Motor Primitives)**: Learns movement patterns from demonstrations and can generalize to new situations
- **PID Control**: Provides precise position control for trajectory following
- **Segment-based Execution**: Splits demonstrations into segments based on grasp state changes
- **Adaptive Re-targeting**: Adjusts learned trajectories for different object positions

## Project Structure

```
.
├── README.md           # This file
├── dmp.py             # Core DMP implementation with canonical system
├── dmp_policy.py      # Main policy class combining DMP with PID control
├── load_data.py       # Utility for loading demonstration data from NPZ files
├── pid.py             # PID controller implementation
└── test_final.py      # Main test script for running the robot simulation
```

## Installation

### Prerequisites

1. **Install Robosuite**: Follow the installation guide at https://robosuite.ai/docs/installation.html

2. **Required Python packages**:
   ```bash
   pip install numpy scipy matplotlib
   ```

### Setup

1. Clone this repository

2. Ensure you have demonstration data in NPZ format named `demos.npz` in the project root directory.

## Usage

### Running the Simulation

Execute the main test script to run the robot simulation:

```bash
python test_final.py
```

This will:
- Create a Robosuite NutAssemblySquare environment with a Panda robot
- Reset the environment and initialize the DMP policy
- Execute the learned manipulation task for 10 runs
- Display the success rate

### Key Parameters

In `test_final.py`, you can modify:
- `runs`: Number of test episodes (default: 10)
- Environment parameters in `suite.make()`

In `dmp_policy.py`, you can adjust:
- `dt`: Control timestep (default: 0.01)
- `n_bfs`: Number of basis functions per DMP (default: 20)
- PID gains: `kp`, `ki`, `kd` values

## How It Works

### 1. Demonstration Loading
The system loads pre-recorded demonstrations from an NPZ file containing robot trajectories and actions.

### 2. Trajectory Segmentation
Demonstrations are automatically segmented based on grasp state changes (open/close gripper transitions).

### 3. DMP Learning
Each segment is learned as a separate DMP, capturing the movement dynamics while allowing for goal re-targeting.

### 4. Adaptive Execution
- **First segment**: Re-targeted to the current object position
- **Middle segments**: Execute original learned patterns
- **Final segment**: Directed to a predefined rod position for insertion
- **Completion**: Performs exploratory nudging motions

### 5. PID Control
A PID controller ensures precise trajectory following by computing position corrections based on the desired DMP trajectory.

### 6. Selecting best expert
We choose the expert situation that matches as close to our current simulation for maximum performance.

## File Descriptions

### `dmp.py`
Contains the core DMP implementation:
- `CanonicalSystem`: Generates the phase variable for DMP execution
- `DMP`: Main DMP class with imitation learning and trajectory generation

### `dmp_policy.py`
The main policy class that:
- Loads and processes demonstration data
- Segments trajectories based on grasp changes
- Combines multiple DMPs with PID control for task execution

### `pid.py`
A multi-dimensional PID controller for precise position control.

### `load_data.py`
Utility functions for loading structured demonstration data from NPZ files.

### `test_final.py`
Main script that sets up the Robosuite environment and tests the DMP policy.

### Modifying Robot Behavior
Adjust the DMP parameters in `dmp_policy.py`:
- Change basis function count (`n_bfs`) for smoother/more detailed movements
- Modify PID gains for different responsiveness
- Adjust offset calculations for different object pickup strategies

### Different Tasks
The system can be adapted to other Robosuite tasks by:
1. Changing the environment name in `test_final.py`
2. Updating observation keys in `dmp_policy.py`
3. Modifying the reward success condition

## Expected Output

The system will display:
- Real-time robot simulation with rendering
- Console output showing object positions and policy initialization
- Final success rate across all test runs

A successful run should show the robot picking up the square nut and inserting it into the corresponding slot.

## Troubleshooting

Data
- demos.npz

## References

This implementation is based on Dynamic Motor Primitives research and demonstrates their application in robotic manipulation tasks using the Robosuite simulation framework.