from rne import *
from min_jerk_v2 import *
from utils import *
from franka_ik_fast import *
from rrt_star import *
import panda_dynamics_model as pdm

METHOD = "arne"
MASS = 5
def arm_conf(_,__):
    return [0, -PI/4, 0.0, -6*PI/8, 0, PI/2, PI/4]

def get_torque_limits_not_exceded_test_base(problem, mass=None):
    def test(poses = None, ptotalMass = None, velocities=None, accelerations=None):
        return True
    return test

def get_torque_limits_not_exceded_test_v3_nov(problem, arm, mass=None):
    robot = problem.robot
    max_limits = []
    baseLink = 1
    joints = get_arm_joints(robot, arm)
    for joint in joints:
            max_limits.append(get_max_force(problem.robot, joint))

    EPS =  1
    totalMass = mass
    if totalMass is None:
        totalMass = get_mass(problem.payload)
    comR = []
    totalMass = 0
    def test(poses = None, ptotalMass = None, velocities=None, accelerations=None):
        totalMass = problem.payload_mass
        if totalMass is None and problem.payload is not None:
            totalMass = get_mass(problem.payload)
        elif problem.payload is None:
            totalMass = 0
        velocities = [0]*len(poses)
        accelerations = [0]*len(poses)
        hold = get_joint_positions(robot, joints)
        set_joint_positions(robot, joints, poses)

        set_joint_positions(robot, joints, hold)
        if totalMass > 0.01:
            r = [0,0,0.05]
            add_payload(r, totalMass)
        torques = rne(poses, velocities, accelerations)
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

def get_torque_limits_not_exceded_test_v2(problem, mass=None):
    robot = problem.robot
    max_limits = []
    baseLink = 1
    joints = get_arm_joints(robot)
    for joint in joints:
            max_limits.append(get_max_force(problem.robot, joint))
    ee_link = get_gripper_link(robot)
    EPS =  1
    def test(poses = None, ptotalMass = None, velocities=None, accelerations=None):
        # return True
        totalMass = problem.payload_mass
        print("in torque test")
        if totalMass is None and problem.payload is not None:
            totalMass = get_mass(problem.payload)
        elif problem.payload is None:
            totalMass = 0
        if velocities is None or accelerations is None:
            velocities = [0]*len(poses)
            accelerations = [0]*len(poses)
    
        hold = get_joint_positions(robot, joints)
        set_joint_positions(robot, joints, poses)
        accelerations += [0.0] * 2
        velocities += [0.0] * 2
        Jl, Ja = compute_jacobian(problem.robot, ee_link, velocities=velocities, accelerations=accelerations)
        M = pdm.get_mass_matrix(poses)
        p = np.ndarray(())
        C = pdm.get_coriolis_matrix(np.asarray([poses[:7]], dtype=np.float64).transpose(), np.asarray([velocities[:7]], dtype=np.float64).transpose())
        torquesInert = np.matmul(M, accelerations[:7])
        torquesC = np.matmul(C, velocities[:7])
        torquesG = pdm.get_gravity_vector(poses)

        set_joint_positions(robot, joints, hold)
        Jl = np.array(Jl).transpose()
        Ja = np.array(Ja).transpose()

        J = np.concatenate((Jl, Ja))

        print(J.shape)
        Jt = np.transpose(J)
        print(J.shape)
        force = totalMass * 9.81
        toolV = np.matmul(J, velocities)

        forceReal = np.array([0, 0, force,
                                0, 0, 0])
        force3d = forceReal
        # print(force3d)
        # print(len(J), len(J[0]))
        torquesExt = np.matmul(Jt, force3d)
        torques = torquesExt[:7] + torquesInert + torquesC + torquesG
        for i in range(len(max_limits)-1):
            if (abs(torques[i]) >= max_limits[i]*EPS):
                return False
        return True
    return test

def get_torque_limits_not_exceded_test_v3_nov(problem, mass=None):
    robot = problem.robot
    max_limits = []
    joints = get_arm_joints(robot)
    for joint in joints:
            max_limits.append(get_max_force(problem.robot, joint))

    EPS =  1
    totalMass = mass
    if totalMass is None:
        totalMass = get_mass(problem.payload)
    totalMass = 0
    def test(poses = None, ptotalMass = None, velocities=None, accelerations=None):
        totalMass = problem.payload_mass
        if totalMass is None and problem.payload is not None:
            totalMass = get_mass(problem.payload)
        elif problem.payload is None:
            totalMass = 0
        velocities = [0]*len(poses)
        accelerations = [0]*len(poses)
        hold = get_joint_positions(robot, joints)
        set_joint_positions(robot, joints, poses)

        set_joint_positions(robot, joints, hold)
        if totalMass > 0.01:
            r = [0,0,0.05]
            add_payload(r, totalMass)
        torques = rne(poses, velocities, accelerations)
        for i in range(len(max_limits)-1):
            if (abs(torques[i]) >= max_limits[i]*EPS):
                remove_payload()
                return False
        remove_payload()
        return True

    return test

def get_torque_limits_not_exceded_test_v4(problem, mass=None):
    robot = problem.robot
    max_limits = []
    baseLink = 1
    joints = get_arm_joints(robot)
    a = 1
    
    for joint in joints:
            max_limits.append(get_max_force(problem.robot, joint))

    EPS =  1
    totalMass = mass
    if totalMass is None:
        totalMass = get_mass(problem.payload)
    comR = []
    totalMass = 0
    def test(poses = None, ptotalMass = problem.payload_mass, velocities=None, accelerations=None):
        totalMass = ptotalMass
        if totalMass is None and problem.payload is not None:
            totalMass = get_mass(problem.payload)
        if velocities is None or accelerations is None:
            velocities = [0]*len(poses)
            accelerations = [0]*len(poses)
        if totalMass > 0.01:
            r = [0,0,0.03]
            add_payload(r, totalMass)
        torques = rne(poses, velocities, accelerations)
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
MAX_GRASP_WIDTH = 0.07
GRASP_LENGTH = 0.15

def get_top_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                   max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH):
    # TODO: rename the box grasps
    center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose)
    reflect_z = Pose(euler=[0, math.pi, 0])
    translate_z = Pose(point=[0, 0, h  - grasp_length])
    translate_center = Pose(point=point_from_pose(body_pose)-center)
    grasps = []
    if w <= max_width:
        for i in range(1 + under):
            rotate_z = Pose(euler=[0, 0, math.pi / 2 + i * math.pi])
            grasps += [multiply(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, body_pose)]
    if l <= max_width:
        for i in range(1 + under):
            rotate_z = Pose(euler=[0, 0, i * math.pi])
            grasps += [multiply(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, body_pose)]
    return grasps

def get_top_grasp(body):
    approach_vector = get_unit_vector([1, 0, 0])
    grasps = get_top_grasps(body)
    grasp = grasps[0]
    return Grasp('top', body, grasp, multiply((approach_vector, unit_quat()), grasp), TOP_HOLDING_LEFT_ARM)

def planner_fn_force_aware(start_conf, pose, problem):
    timestamp = str(datetime.datetime.now())
    timestamp = "{}_{}".format(timestamp.split(' ')[0], timestamp.split(' ')[1])
    robot = problem.robot
    obstacles = problem.fixed
    METHOD = problem.torque_test
    if METHOD == "rne":
        torque_test_right = get_torque_limits_not_exceded_test_v4(problem)
    elif METHOD == "dyn":
        torque_test_right = get_torque_limits_not_exceded_test_v2(problem)
    elif METHOD == "base":
        torque_test_right = get_torque_limits_not_exceded_test_base(problem)
    elif METHOD == "nov":
        torque_test_right = get_torque_limits_not_exceded_test_v3_nov(problem)
    arm = "right"
    obj = problem.payload
    custom_limits = {}
    grasp = get_top_grasp(obj)
    # pose = world_pose_to_robot_frame(robot, pose)
    torque_test = torque_test_right
    gripper_pose = multiply(list(pose), invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
    arm_link = get_gripper_link(robot)
    # arm_link = link_from_name(robot, 'r_panda_link8')
    arm_joints = get_arm_joints(robot)
    max_velocities = get_max_velocities(problem.robot, arm_joints)
    resolutions = 0.2**np.ones(len(arm_joints))
    dynam_fn = get_dynamics_fn_v5(problem, resolutions)
    default_conf = start_conf
    open_arm(robot, arm)
    set_joint_positions(robot, arm_joints, default_conf) # default_conf | sample_fn()
    ikfaskt_info = PANDA_INFO
    gripper_link = link_from_name(robot, PANDA_GRIPPER_ROOT)
    grasp_conf = None
    for i in range(25):
        grasp_conf = bi_panda_inverse_kinematics(robot, arm, arm_link, gripper_pose, max_attempts=25, max_time=3.5, obstacles=obstacles)
        if grasp_conf is not None:
            break
    if (grasp_conf is None) or any(pairwise_collision(robot, b) for b in obstacles): # [obj]
        print('Grasp IK failure', grasp_conf)
        return None
    if not torque_test(grasp_conf):
        print('grasp conf torques exceded')
        return None
    # if grasp_conf is None:
    print("found grasp")
    set_joint_positions(robot, arm_joints, default_conf)

    attachments = {}
    set_joint_positions(robot, arm_joints, default_conf)
    approach_path, approach_vels, approach_accels, approach_dts = plan_joint_motion_force_aware(robot, arm_joints, grasp_conf, torque_test, dynam_fn, attachments=attachments.values(),
                                        obstacles=obstacles, self_collisions=SELF_COLLISIONS, max_time=50,
                                        custom_limits=custom_limits, radius=resolutions/2,
                                        max_iterations=50, start_conf=start_conf)
    if approach_path is None:
        print('Approach path failure')
        return None

    path = approach_path #+ grasp_path
    mt = create_trajectory(robot, arm_joints, path, bodies = [problem.payload], velocities=approach_vels, accelerations=approach_accels, dts = approach_dts, ts=timestamp, dynam_fn=rne)
    return mt

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

def get_dynamics_fn_v5(problem, resolutions):
    arm_joints = get_arm_joints(problem.robot)
    num_joints = len(arm_joints)
    max_velocities = get_max_velocities(problem.robot, arm_joints)
    def dynam_fn(path, dur = None, vel0 = [0.0]*num_joints, acc0=[0.0]*num_joints):
        print("run min jerk")
        m_coeff = minjerk_coefficients(np.array(path))

        #### CHANGE THIS ####
        move_time = problem.execution_time  # seconds
        #####################

        panda_command_freq = 1000  # Hz
        num_intervals = move_time * panda_command_freq / len(path)

        minjerk_traj = minjerk_trajectory(m_coeff, num_intervals=int(num_intervals))
        # outputs timestamp, q, qd, qdd
        q = [list(q[0]) for q in minjerk_traj]
        qd = [list(q[1]) for q in minjerk_traj]
        qdd  = [list(q[2]) for q in minjerk_traj]
        psg = [move_time * n/len(minjerk_traj) for n in range(0, len(minjerk_traj))]
        return q,psg, qd,qdd
    
    return dynam_fn

def open_arm(robot, arm): # These are mirrored on the pr2
    for joint in get_gripper_joints(robot, arm):
        set_joint_position(robot, joint, get_max_limit(robot, joint))

def get_gripper_joints(robot, arm):
    return joints_from_names(robot, GRIPPER_JOINT_NAMES)

def plan_joint_motion_force_aware(body, joints, end_conf, torque_fn, dynam_fn, obstacles=[], attachments=[],
                      self_collisions=True, disabled_collisions=set(),
                      weights=None, radius=None, max_distance=MAX_DISTANCE,
                      use_aabb=False, cache=True, custom_limits={}, start_conf = None,**kwargs):

    assert len(joints) == len(end_conf)
    if (weights is None) and (radius is not None):
        weights = np.reciprocal(radius)
    sample_fn = get_sample_fn(body, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(body, joints, weights=weights)
    extend_fn = get_extend_fn(body, joints, resolutions=radius)
    collision_fn = get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance,
                                    use_aabb=use_aabb, cache=cache)
    if start_conf is None:
        start_conf = get_joint_positions(body, joints)

    if not check_initial_end_force_aware(start_conf, end_conf, collision_fn, torque_fn):
        return None, None, None
    return rrt_star_force_aware(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, torque_fn, dynam_fn, radius=[0.01], **kwargs)