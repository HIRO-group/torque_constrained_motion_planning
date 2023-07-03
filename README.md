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

### Using the Planner
Once you set up your problem to use the planner you need to set up the planner function using the `get_planner_fn_force_aware`
method which takes the planning problem setup as input.
```
planner = get_planner_fn_force_aware(problem)
```
To generate a trajectory the planner takes the following as input:

- `start_conf` - The initial joint configuration of the robot
- `pose` - the goal location for the payload
If a solution trajectory is found the returned object has the following form:
- path (array of:)
    - value - the joint configuration at the given time step
    - velocities - the joint velocities at a given timestep
    - accelerations - the joint accelerations at a given timestep
    - dt - the length of time from the first state in the trajectory to the end of execution for this state (state.dt = prev_state.dt + DT)
        where DT is the length of execution for this state in seconds