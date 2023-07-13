# torque_constrained_motion_planning


### Usage:
#### Set up ikfast
from the command line run:
```
python3 ik_setup.py
```

#### Planning Problem Setup
The Problem utility class (in util.py) is used to provide environment information. It takes 4 arguments:
- `Robot` - the body ID of the robot in your pybullet sim
- `Fixed` - a list of objects/ obstacle body IDs in your sim the robot must avoid (not including the payload)
- `Payload` - body ID of object representing robot payload
- `payload mass`- mass of the payload 
- `execution_time` - the maximum time that the trajectory should take to execute
- `torque_test` - the method you wish to use for the torque evaluation of the joints the options are as follows:
    - `base` - no torque evaluation
    - `dyn` - the rigid body dynamics equation for inverse dynamics with the payload estimated as a point mass at the gripper
    - `nov` - recursive newton euler where the velocity and accelerations are assumed to 0
    - `rne` - full recursive newton euler where the velocities and accelerations from min jerk optimization are considered

### Using the Planner
Once you set up your problem planner_fn_force_aware To generate a trajectory the planner takes the following as input:
- `start_conf` - The initial joint configuration of the robot
- `pose` - the goal location for the payload
- `problem` - the defined problem

If a solution trajectory is found the returned object has the following form:
- path (array of:)
    - value - the joint configuration at the given time step
    - velocities - the joint velocities at a given timestep
    - accelerations - the joint accelerations at a given timestep
    - dt - the length of time from the first state in the trajectory to the end of execution for this state (state.dt = prev_state.dt + DT)
        where DT is the length of execution for this state in seconds
