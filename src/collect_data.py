from random import uniform

import argparse
from utils import *
from panda_primitives import *
import ikfast


def packed_force_aware_transfer_HIRO(show_sols=True, arm='right', num=1, dist=0.5, high_angle=math.pi/4, low_angle = -math.pi/4, mass=MASS, initial_conf=TOP_HOLDING_LEFT_ARM):
    # TODO: packing problem where you have to place in one direction
    print('in packed')
    base_extent = 5.0
    X_DIST = dist
    base_limits = (-base_extent/2.*np.ones(2), base_extent/2.*np.ones(2))
    block_width = 0.04
    block_height = 0.1
    #block_height = 2*block_width
    block_area = block_width*block_width

    plate_width = 0.2

    print('Width:', plate_width)
    plate_width = min(plate_width, 0.04)
    plate_height = 0.005

    initial_conf = TOP_HOLDING_LEFT_ARM
    add_data_path()
    floor = load_pybullet("plane.urdf")
    set_point(floor, (0,0,-.954))
    panda = create_panda()
    # set_point(panda,point=Point(0,0, 0.1))
    set_joint_force_limits(panda)
    set_arm_conf(panda, arm, initial_conf)
    open_arm(panda, arm)
    # set_point(panda, (0,0,0.4))
    table = load_pybullet(HIRO_TABLE_1, rel_path=True)
    set_point(table, (-0.2994,0,-0.5131))
    table2 = load_pybullet(HIRO_TABLE_2, rel_path=True)
    set_point(table2, (0.6218, 0,-0.5131))
    set_mass(table2, 1000000)
    set_mass(table, 1000000)

    wall = load_pybullet(WALL_URDF, rel_path=True)
    set_pose(wall, ((-0.7366, 0,0),quat_from_euler((0,0,0))))
    add_fixed_constraint(wall, floor)

    start_plate = create_box(.5, .9, .01, color=GREEN)
    plate_z = stable_z(start_plate, table2)
    set_point(start_plate, (.5, 0, plate_z))
    plate = create_box(plate_width, plate_width, plate_height, color=GREEN)
    plate_z = stable_z(plate, table)
    set_point(plate, Point(x=0, y=-.45, z=plate_z ))
    add_fixed_constraint(plate, table)

    blocks = [load_pybullet(COKE_URDF, rel_path=True) for _ in range(num)]
    for block in blocks:
        set_mass(block, mass)
    initial_surfaces = {block: start_plate for block in blocks}

    sample_placements(initial_surfaces)
    start_dist = get_pose(blocks[0])
    theta = uniform(low_angle, high_angle)
    new_x = X_DIST * math.cos(theta)
    new_y = X_DIST * math.sin(theta)
    obj_z = stable_z(blocks[0], start_plate)
    set_point(blocks[0], (new_x, new_y, obj_z))
    enable_gravity()
    problem = Problem(panda, [table, table2, wall, plate], blocks[-1])
    planner = get_planner_fn_force_aware(problem)
    saver = WorldSaver()
    traj = planner(initial_conf, get_pose(blocks[-1]))
    saver.restore()
    set_real_time(True)
    prevT = 0
    if traj is None:
        return None
    path = list(traj.path)
    path_rev = deepcopy(path)
    path_rev.reverse()
    full_path = path + path_rev
    print(path[-1].values)
    if show_sols:
        for conf in full_path:
            set_joint_positions_torque(panda, get_arm_joints(panda), conf.values, conf.velocities)
            wait_for_duration(.001)
    return path

def save_traj_data(traj):
    if traj is None:
        return


def main():
    parser = argparse.ArgumentParser()
    ts = str(datetime.now()).replace(" ", "_")
    parser.add_argument('-sets', default=10, type=int, help='The number of itterations to run experiment')
    parser.add_argument('-random-start', action='store_true', help='Randomizes start position')
    parser.add_argument('-show-solutions', action='store_true', help='Randomizes start position')
    parser.add_argument('-data-path', default="data/", type=str, help='The number of itterations to run experiment')
    parser.add_argument('-file-name', default=f"data_collection_{ts}.csv")
    connect(use_gui=True)
    args = parser.parse_args()
    for i in range(args.sets):
        traj = packed_force_aware_transfer_HIRO()
    save_traj_data(traj)
    
    disconnect()