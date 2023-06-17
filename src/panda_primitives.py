from rne import *
from min_jerk_v2 import *
METHOD = "arne"

def get_torque_limits_not_exceded_test_v4(problem, arm, mass=None):
    robot = problem.robot
    max_limits = []
    baseLink = 1
    joints = get_arm_joints(robot, arm)
    a = arm_from_arm(arm)
    
    for joint in joints:
            max_limits.append(get_max_force(problem.robot, joint))

    EPS =  1
    totalMass = mass
    if totalMass is None:
        totalMass = get_mass(problem.movable[-1])
    comR = []
    totalMass = 0
    def test(poses = None, ptotalMass = None, velocities=None, accelerations=None):
        totalMass = ptotalMass
        if totalMass is None:
            totalMass = get_mass(problem.movable[-1])
        if velocities is None or accelerations is None:
            velocities = [0]*len(poses)
            accelerations = [0]*len(poses)
        if totalMass > 0.01:
            r = [0,0,0.03]
            add_payload(r, totalMass)
        torques = RNE(poses, velocities, accelerations)
        for i in range(len(max_limits)-1):
            if (abs(torques[i]) >= max_limits[i]*EPS):
                print("torque test: FAILED", i, torques[i])
                # print("Velocities: ", velocities)
                # print("Accelerations: ", accelerations)
                remove_payload()
                return False
        # print("torque test: PASSED")
        remove_payload()
        return True

    return test


def get_ik_fn_force_aware(problem, custom_limits={}, collisions=True, teleport=True, max_attempts = 100):
    robot = problem.robot
    obstacles = problem.fixed + problem.surfaces if collisions else []
    # torque_test_left = get_torque_limits_not_exceded_test_v2(problem, 'left')
    torque_test_right = None
    if METHOD == "rne":
        torque_test_right = get_torque_limits_not_exceded_test_v3(problem, 'right')
    elif METHOD == "arne":
        torque_test_right = get_torque_limits_not_exceded_test_v4(problem, 'right')
    elif METHOD == "dyn":
        torque_test_right = get_torque_limits_not_exceded_test_v2(problem, 'right')
    elif METHOD == "base":
        torque_test_right = get_torque_limits_not_exceded_test_base(problem, 'right')
    elif METHOD == "nov":
        torque_test_right = get_torque_limits_not_exceded_test_v3_nov(problem, 'right')
    timestamp = str(datetime.datetime.now())
    timestamp = "{}_{}".format(timestamp.split(' ')[0], timestamp.split(' ')[1])

    def fn(arm, obj, pose, grasp, reconfig=None):
        torque_test = torque_test_left if arm == 'left' else torque_test_right
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}

        gripper_pose = multiply(pose.value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
        approach_pose = multiply(pose.value, invert(grasp.approach))
        arm_link = get_gripper_link(robot, arm)
        pick_grasp = get_pick_grasp()
        # arm_link = link_from_name(robot, 'r_panda_link8')
        arm_joints = get_arm_joints(robot, arm)
        max_velocities = get_max_velocities(problem.robot, arm_joints)
        resolutions = 0.2**np.ones(len(arm_joints))
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(resolutions)
        dynam_fn = get_dynamics_fn_v5(problem, resolutions)
        objMass = get_mass(obj)
        objPose = get_pose(obj)[0]
        default_conf = arm_conf(arm, grasp.carry)
        pose.assign()
        open_arm(robot, arm)
        set_joint_positions(robot, arm_joints, default_conf) # default_conf | sample_fn()
        ikfaskt_info = PANDA_RIGHT_INFO
        gripper_link = link_from_name(robot, PANDA_GRIPPER_ROOTS[arm])
        grasp_conf = None
        # custom_limits[-2] = (default_conf[-2]-.1, default_conf[-2]+.1)
        # custom_limits[-1] = (default_conf[-1]-.001, default_conf[-1]+.001)
        grasp_conf = bi_panda_inverse_kinematics(robot, arm, arm_link, gripper_pose, max_attempts=25, max_time=3.5, obstacles=obstacles)
        if (grasp_conf is None) or any(pairwise_collision(robot, b) for b in obstacles): # [obj]
            print('Grasp IK failure', grasp_conf)
            return None
        if not torque_test(grasp_conf):
            print('grasp conf torques exceded')
            return None
        # if grasp_conf is None:
        print("found grasp")
        set_joint_positions(robot, arm_joints, default_conf)

        attachment = grasp.get_attachment(problem.robot, arm)
        attachments = {attachment.child: attachment}
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            # grasp_path = plan_direct_joint_motion_force_aware(robot, arm_joints, grasp_conf, torque_test, dynam_fn, attachments=attachments.values(),
            #                                       obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
            #                                       custom_limits={}, resolutions=resolutions/2.)
            # if grasp_path is None:
            #     print('Grasp path failure')
            #     return None
            set_joint_positions(robot, arm_joints, default_conf)
            # approach_path, approach_vels, _ = plan_joint_motion_force_aware(robot, arm_joints, approach_conf, torque_test, dynam_fn, attachments=attachments.values(),
            #                                   obstacles=obstacles, self_collisions=SELF_COLLISIONS,
            #                                   custom_limits=custom_limits, resolutions=resolutions,
            #                                   restarts=4, iterations=25, smooth=25)
            approach_path, approach_vels, approach_accels, approach_dts = plan_joint_motion_force_aware(robot, arm_joints, grasp_conf, torque_test, dynam_fn, attachments=attachments.values(),
                                              obstacles=obstacles, self_collisions=SELF_COLLISIONS, max_time=50,
                                              custom_limits=custom_limits, radius=resolutions/2,
                                              max_iterations=50)
            # approach_path, approach_vels, approach_accels = ruckig_path_planner(default_conf, grasp_conf, torque_test, arm_joints, max_velocities)
            if approach_path is None:
                print('Approach path failure')
                return None

            path = approach_path #+ grasp_path
        mt = create_trajectory(robot, arm_joints, path, bodies = problem.movable, velocities=approach_vels, accelerations=approach_accels, dts = approach_dts, ts=timestamp)
        if reconfig is not None:
            cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[reconfig, mt])
        else:
            cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])

        return (cmd,)
    return fn

def test_path_torque_constraint(robot, arm, joints, path, mass, r, test_fn):
    reset = get_joint_positions(robot, joints)
    for conf in path:
        set_joint_positions(robot, joints, conf)
        if not test_fn(arm, ptotalMass=mass, pcomR=r):
            print('conf torques exceded in path')
            set_joint_positions(robot, joints, reset)
            return True
    set_joint_positions(robot, joints, reset)
    return False