# torque_constrained_motion_planning


### Usage:
#### Set up ikfast
from the command line run:
```
python3 ik_setup.py
```

#### Planning Problem Setup
The Problem class is used to provide environment information. It takes 4 arguments:
- Robot - the body ID of the robot in your pybullet sim
- Fixed - a list of objects/ obstacle body IDs in your sim
