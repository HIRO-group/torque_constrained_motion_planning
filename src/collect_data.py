from random import uniform
import time as Time
import argparse
from utils import *
from panda_primitives import *
import ikfast
import os
import csv

def packed_force_aware_transfer_HIRO(show_sols=True, arm='right', num=1, dist=0.5, high_angle=math.pi/4, low_angle = -math.pi/4, mass=MASS, initial_conf=TOP_HOLDING_LEFT_ARM):
    # TODO: packing problem where you have to place in one direction
    connect(use_gui=show_sols)
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
    set_point(floor, (0,0,-1))
    panda = create_panda()
    # set_point(panda,point=Point(0,0, 0.1))
    set_joint_force_limits(panda)
    set_arm_conf(panda, arm, initial_conf)
    open_arm(panda, arm)
    # set_point(panda, (0,0,0.4))
    table = load_pybullet(HIRO_TABLE_1, rel_path=True)
    set_point(table, (-0.39905, -0.04297, -0.48))
    table2 = load_pybullet(HIRO_TABLE_2, rel_path=True)
    set_point(table2, (0.4614, -0.0502, -0.48))
    set_mass(table2, 1000000)
    set_mass(table, 1000000)

    wall = load_pybullet(WALL_URDF, rel_path=True)
    set_pose(wall, ((-0.7366, 0, 0),quat_from_euler((0,0,0))))
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
    problem = Problem(panda, [table, table2, wall, plate], blocks[-1], mass, 5, torque_test="rne")
    planner = planner_fn_force_aware
    saver = WorldSaver()
    start = Time.time()
    
    grasp_pose = get_pose(blocks[-1])
    approach_pose = ((grasp_pose[0][0],grasp_pose[0][1], grasp_pose[0][2] + .05), grasp_pose[1])
    place_pose = ((0, -0.45, plate_z + .05), grasp_pose[1])
    approach_path = planner(initial_conf, approach_pose, problem)
    problem.execution_time = 1
    grasp_path = planner(approach_path.path[-1].values, grasp_pose,  problem)
    problem.execution_time = 5
    place_path = planner(grasp_path.path[-1].values, place_pose,  problem)
    planning_time = Time.time() - start
    saver.restore()
    set_real_time(True)
    prevT = 0
    if approach_path is None or grasp_path is None or place_path is None:
        disconnect()
        return None, None
    path = list(approach_path.path) + list(grasp_path.path) + list(place_path.path)
    full_path = path
    print(path[-1].values)
    if show_sols:
        for conf in full_path:
            set_joint_positions_torque(panda, get_arm_joints(panda), conf.values, conf.velocities)
            wait_for_duration(.001)
    disconnect()
    return path, planning_time



def save_traj_data(traj, args, filename):
    if traj is None:
        return
    velocities = []
    confs = []
    torques = []
    ts = []
    accelerations = []
    for conf in traj:
        velocities.append(conf.velocities)
        confs.append(conf.values)
        accelerations.append(conf.accelerations)
        ts.append(conf.dt)
        torques.append(conf.torques)
    
    np.savez(
        args.data_path + "/" + filename,
        q = confs,
        qd = velocities,
        qdd = accelerations,
        torques = torques,
        ts = ts
    )

def main():
    parser = argparse.ArgumentParser()
    ts = str(datetime.datetime.now()).replace(" ", "_")
    parser.add_argument('-sets', default=10, type=int, help='The number of itterations to run experiment')
    parser.add_argument('-random-start', action='store_true', help='Randomizes start position')
    parser.add_argument('-mass', default=MASS, type=int, help='mass of the payload for this set of experiments')
    parser.add_argument('-dist', default=0.5, type=float, help='distance of the payload from the base of the robot for this set of experiments (0, .8)')

    parser.add_argument('-show-solutions', default=False, action='store_true', help='Randomizes start position')
    parser.add_argument('-data-path', default="data/", type=str, help='The number of itterations to run experiment')
    parser.add_argument('-file-name', default=f"data_collection_{ts}")
    args = parser.parse_args()
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

    meta_file = os.path.join(args.data_path, args.file_name)
    with open(meta_file, 'w', newline='') as csvfile:
        metaWriter = csv.writer(csvfile, delimiter=',')
        metaWriter.writerow(["planning_times, mass, distance, filename"])

    meta_file = args.data_path + "/" + args.file_name + "_meta.csv"
    
    with open(meta_file, 'w', newline='') as csvfile:
        metaWriter = csv.writer(csvfile, delimiter=',')
        
        for i in range(args.sets):
            filename = args.file_name + f"{i}.npz"
            traj, planning_time = packed_force_aware_transfer_HIRO(show_sols=args.show_solutions, mass=args.mass, dist=args.dist)
            save_traj_data(traj, args, filename)
            metaWriter.writerow([planning_time, args.mass, args.dist, filename])

    
if __name__ == '__main__':
    main()