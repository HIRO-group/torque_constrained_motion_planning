from __future__ import print_function

import collections
import colorsys
import inspect
import json
import math
import os
import pickle
import platform
import signal
import numpy as np
import pybullet as p
import random
import sys
import time
import datetime
import shutil
import cProfile
import pstats
from copy import deepcopy

from collections import defaultdict, deque, namedtuple
from collections import defaultdict, deque, namedtuple
from itertools import product, combinations, count, cycle, islice
from multiprocessing import TimeoutError
from contextlib import contextmanager
from tf import *
ARM_NAMES = ['panda_joint1', 'panda_joint2', 'panda_joint3','panda_joint4',
                            'panda_joint5', 'panda_joint6', 'panda_joint7']

GRIPPER_NAMES = ['panda_hand_joint', 'panda_finger_joint1',
                                 'panda_finger_joint2', 'panda_grasptarget_hand']

PANDA_GRIPPER_ROOT = 'panda_hand'


def Point(x=0., y=0., z=0.):
    return np.array([x, y, z])

def Euler(roll=0., pitch=0., yaw=0.):
    return np.array([roll, pitch, yaw])

def Pose(point=None, euler=None):
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return point, quat_from_euler(euler)

TOOL_POSE = Pose(point=Point(0, 0.0, 0.1),euler=Euler(roll= 0,pitch=0, yaw=0))

BASE_LINK = -1
CLIENT = 0
INF = float("inf")
PI = np.pi
Interval = namedtuple('Interval', ['lower', 'upper']) # AABB
UNIT_LIMITS = Interval(0., 1.)
CIRCULAR_LIMITS = Interval(-PI, PI)
UNBOUNDED_LIMITS = Interval(-INF, INF)
# Colors
SELF_COLLISIONS = False
RGB = namedtuple('RGB', ['red', 'green', 'blue'])
RGBA = namedtuple('RGBA', ['red', 'green', 'blue', 'alpha'])
MAX_RGB = 2**8 - 1

RED = RGBA(1, 0, 0, 1)
GREEN = RGBA(0, 1, 0, 1)
BLUE = RGBA(0, 0, 1, 1)
BLACK = RGBA(0, 0, 0, 1)
WHITE = RGBA(1, 1, 1, 1)
BROWN = RGBA(0.396, 0.263, 0.129, 1)
TAN = RGBA(0.824, 0.706, 0.549, 1)
GREY = RGBA(0.5, 0.5, 0.5, 1)
YELLOW = RGBA(1, 1, 0, 1)
TRANSPARENT = RGBA(0, 0, 0, 0)

ACHROMATIC_COLORS = {
    'white': WHITE,
    'grey': GREY,
    'black': BLACK,
}

CHROMATIC_COLORS = {
    'red': RED,
    'green': GREEN,
    'blue': BLUE,
}


def merge_dicts(*args):
    result = {}
    for d in args:
        result.update(d)
    return result
    # return dict(reduce(operator.add, [d.items() for d in args]))

COLOR_FROM_NAME = merge_dicts(ACHROMATIC_COLORS, CHROMATIC_COLORS)
def read(filename):
    with open(filename, 'r') as f:
        return f.read()

def write(filename, string):
    with open(filename, 'w') as f:
        f.write(string)

def remove_alpha(color):
    return RGB(*color[:3])

def apply_alpha(color, alpha=1.):
    if color is None:
        return None
    red, green, blue = color[:3]
    return RGBA(red, green, blue, alpha)

def spaced_colors(n, s=1, v=1):
    return [RGB(*colorsys.hsv_to_rgb(h, s, v)) for h in np.linspace(0, 1, n, endpoint=False)]

def safe_zip(sequence1, sequence2): # TODO: *args
    sequence1, sequence2 = list(sequence1), list(sequence2)
    assert len(sequence1) == len(sequence2)
    return list(zip(sequence1, sequence2))
class HideOutput(object):
    # https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
    # https://stackoverflow.com/questions/4178614/suppressing-output-of-module-calling-outside-library
    # https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
    '''
    A context manager that block stdout for its scope, usage:

    with HideOutput():
        os.system('ls -l')
    '''
    DEFAULT_ENABLE = True
    def __init__(self, enable=None):
        if enable is None:
            enable = self.DEFAULT_ENABLE
        self.enable = enable
        if not self.enable:
            return
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        if not self.enable:
            return
        self.fd = 1
        #self.fd = sys.stdout.fileno()
        self._newstdout = os.dup(self.fd)
        os.dup2(self._devnull, self.fd)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return
        sys.stdout.close()
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, self.fd)
        os.close(self._oldstdout_fno) # Added
CLIENTS = {}

def joint_from_name(body, name):
    for joint in get_joints(body):
        if get_joint_name(body, joint) == name:
            return joint
    raise ValueError(body, name)

def get_client(client):
    return CLIENTS[client]

def set_renderer(enable):
    client = CLIENT
    if not has_gui(client):
        return
    CLIENTS[client] = enable
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, int(enable), physicsClientId=client)

class LockRenderer():
    # disabling rendering temporary makes adding objects faster
    def __init__(self, lock=True):
        self.client = CLIENT
        self.state = CLIENTS[self.client]
        # skip if the visualizer isn't active
        if has_gui(self.client) and lock:
            set_renderer(enable=False)

    def restore(self):
        if not has_gui(self.client):
            return
        assert self.state is not None
        if self.state != CLIENTS[self.client]:
           set_renderer(enable=self.state)
def tform_point(affine, point):
    return point_from_pose(multiply(affine, Pose(point=point)))

PANDA_ARM = ['panda_joint1', 'panda_joint2', 'panda_joint3','panda_joint4',
                            'panda_joint5', 'panda_joint6', 'panda_joint7']

def get_arm_joints(arm):
    return joints_from_names(PANDA_ARM)

PANDA_GROUPS = {
    "base": ["panda_joint1"],
    "arm": ['panda_joint1', 'panda_joint2', 'panda_joint3','panda_joint4',
                            'panda_joint5', 'panda_joint6', 'panda_joint7'],
    # gripper_from_arm(LEFT_ARM): ['l_panda_hand_joint', 'l_panda_finger_joint1',
    #                              'l_panda_finger_joint2', 'l_panda_grasptarget_hand'],
    "hand": ['panda_hand_joint', 'panda_finger_joint1',
                                 'panda_finger_joint2', 'panda_grasptarget_hand'],
}
JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                       'jointReactionForces', 'appliedJointMotorTorque'])

DynamicsInfo = namedtuple('DynamicsInfo', [
    'mass', 'lateral_friction', 'local_inertia_diagonal', 'local_inertial_pos',  'local_inertial_orn',
    'restitution', 'rolling_friction', 'spinning_friction', 'contact_damping', 'contact_stiffness']) #, 'body_type'])

ModelInfo = namedtuple('URDFInfo', ['name', 'path', 'fixed_base', 'scale'])

INFO_FROM_BODY = {}

def get_model_info(body):
    key = (CLIENT, body)
    return INFO_FROM_BODY.get(key, None)

def get_urdf_flags(cache=False, cylinder=False):
    # by default, Bullet disables self-collision
    # URDF_INITIALIZE_SAT_FEATURES
    # URDF_ENABLE_CACHED_GRAPHICS_SHAPES seems to help
    # but URDF_INITIALIZE_SAT_FEATURES does not (might need to be provided a mesh)
    # flags = p.URDF_INITIALIZE_SAT_FEATURES | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    flags = 0
    if cache:
        flags |= p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    if cylinder:
        flags |= p.URDF_USE_IMPLICIT_CYLINDER
    return flags | p.URDF_USE_INERTIA_FROM_FILE

def load_pybullet(filename, fixed_base=False, scale=1., **kwargs):
    # fixed_base=False implies infinite base mass
    with LockRenderer():
        if filename.endswith('.urdf'):
            flags = get_urdf_flags(**kwargs)
            body = p.loadURDF(filename, useFixedBase=fixed_base, flags=flags,
                              globalScaling=scale, physicsClientId=CLIENT)
        elif filename.endswith('.sdf'):
            body = p.loadSDF(filename, physicsClientId=CLIENT)
        elif filename.endswith('.xml'):
            body = p.loadMJCF(filename, physicsClientId=CLIENT)
        elif filename.endswith('.bullet'):
            body = p.loadBullet(filename, physicsClientId=CLIENT)
        elif filename.endswith('.obj'):
            # TODO: fixed_base => mass = 0?
            body = create_obj(filename, scale=scale, **kwargs)
        else:
            raise ValueError(filename)
    INFO_FROM_BODY[CLIENT, body] = ModelInfo(None, filename, fixed_base, scale)
    return body

def set_caching(cache):
    p.setPhysicsEngineParameter(enableFileCaching=int(cache), physicsClientId=CLIENT)

def load_model_info(info):
    # TODO: disable file caching to reuse old filenames
    # p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=CLIENT)
    if info.path.endswith('.urdf'):
        return load_pybullet(info.path, fixed_base=info.fixed_base, scale=info.scale)
    if info.path.endswith('.obj'):
        mass = STATIC_MASS if info.fixed_base else 1.
        return create_obj(info.path, mass=mass, scale=info.scale)
    raise NotImplementedError(info.path)

URDF_FLAGS = [p.URDF_USE_INERTIA_FROM_FILE,
              p.URDF_USE_SELF_COLLISION,
              p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
              p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS]

def get_model_path(rel_path): # TODO: add to search path
    directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directory, '..', rel_path)

def load_model(rel_path, pose=None, **kwargs):
    # TODO: error with loadURDF when loading MESH visual and CYLINDER collision
    abs_path = get_model_path(rel_path)
    add_data_path()
    #with LockRenderer():
    body = load_pybullet(abs_path, **kwargs)
    if pose is not None:
        set_pose(body, pose)
    return body

#TOOLS_VERSION = date.date()

def get_version(): # year-month-0-day format
    s = str(p.getAPIVersion(physicsClientId=CLIENT))
    return datetime.date(year=int(s[:4]), month=int(s[4:6]), day=int(s[7:9]))

#####################################

# class World(object):
#     def __init__(self, client):
#         self.client = client
#         self.bodies = {}
#     def activate(self):
#         set_client(self.client)
#     def load(self, path, name=None, fixed_base=False, scale=1.):
#         body = p.loadURDF(path, useFixedBase=fixed_base, physicsClientId=self.client)
#         self.bodies[body] = URDFInfo(name, path, fixed_base, scale)
#         return body
#     def remove(self, body):
#         del self.bodies[body]
#         return p.removeBody(body, physicsClientId=self.client)
#     def reset(self):
#         p.resetSimulation(physicsClientId=self.client)
#         self.bodies = {}
#     # TODO: with statement
#     def copy(self):
#         raise NotImplementedError()
#     def __repr__(self):
#         return '{}({})'.format(self.__class__.__name__, len(self.bodies))

#####################################

now = time.time

def elapsed_time(start_time):
    return time.time() - start_time

MouseEvent = namedtuple('MouseEvent', ['eventType', 'mousePosX', 'mousePosY', 'buttonIndex', 'buttonState'])

def get_mouse_events():
    return list(MouseEvent(*event) for event in p.getMouseEvents(physicsClientId=CLIENT))

def update_viewer():
    # https://docs.python.org/2/library/select.html
    # events = p.getKeyboardEvents() # TODO: only works when the viewer is in focus
    get_mouse_events()
    # for k, v in keys.items():
    #    #p.KEY_IS_DOWN, p.KEY_WAS_RELEASED, p.KEY_WAS_TRIGGERED
    #    if (k == p.B3G_RETURN) and (v & p.KEY_WAS_TRIGGERED):
    #        return
    # time.sleep(1e-3) # Doesn't work
    # disable_gravity()

def wait_for_duration(duration): #, dt=0):
    t0 = time.time()
    while elapsed_time(t0) <= duration:
        update_viewer()

def simulate_for_duration(duration):
    dt = get_time_step()
    for i in range(int(duration / dt)):
        step_simulation()

def get_time_step():
    # {'gravityAccelerationX', 'useRealTimeSimulation', 'gravityAccelerationZ', 'numSolverIterations',
    # 'gravityAccelerationY', 'numSubSteps', 'fixedTimeStep'}
    return p.getPhysicsEngineParameters(physicsClientId=CLIENT)['fixedTimeStep']

def enable_separating_axis_test():
    p.setPhysicsEngineParameter(enableSAT=1, physicsClientId=CLIENT)
    #p.setCollisionFilterPair()
    #p.setCollisionFilterGroupMask()
    #p.setInternalSimFlags()
    # enableFileCaching: Set to 0 to disable file caching, such as .obj wavefront file loading
    #p.getAPIVersion() # TODO: check that API is up-to-date

def simulate_for_sim_duration(sim_duration, real_dt=0, frequency=INF):
    t0 = time.time()
    sim_dt = get_time_step()
    sim_time = 0
    last_print = 0
    while sim_time < sim_duration:
        if frequency < (sim_time - last_print):
            print('Sim time: {:.3f} | Real time: {:.3f}'.format(sim_time, elapsed_time(t0)))
            last_print = sim_time
        step_simulation()
        sim_time += sim_dt
        time.sleep(real_dt)

def wait_for_user(message='Press enter to continue'):
    if has_gui():
        # OS X doesn't multi-thread the OpenGL visualizer
        #wait_for_interrupt()
        return threaded_input(message)
    return input(message)

def wait_if_gui(*args, **kwargs):
    if has_gui():
        wait_for_user(*args, **kwargs)

def is_unlocked():
    return CLIENTS[CLIENT] is True

def wait_if_unlocked(*args, **kwargs):
    if is_unlocked():
        wait_for_user(*args, **kwargs)

def wait_for_interrupt(max_time=np.inf):
    """
    Hold Ctrl to move the camera as well as zoom
    """
    print('Press Ctrl-C to continue')
    try:
        wait_for_duration(max_time)
    except KeyboardInterrupt:
        pass
    finally:
        print()

def set_preview(enable):
    # lightPosition, shadowMapResolution, shadowMapWorldSize
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, enable, physicsClientId=CLIENT)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, enable, physicsClientId=CLIENT)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, enable, physicsClientId=CLIENT)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, enable, physicsClientId=CLIENT)
    #p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, True, physicsClientId=CLIENT)
    #p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, True, physicsClientId=CLIENT)

def enable_preview():
    set_preview(enable=True)

def disable_preview():
    set_preview(enable=False)

def set_renderer(enable):
    client = CLIENT
    if not has_gui(client):
        return
    CLIENTS[client] = enable
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, int(enable), physicsClientId=client)

class LockRenderer():
    # disabling rendering temporary makes adding objects faster
    def __init__(self, lock=True):
        self.client = CLIENT
        self.state = CLIENTS[self.client]
        # skip if the visualizer isn't active
        if has_gui(self.client) and lock:
            set_renderer(enable=False)

    def restore(self):
        if not has_gui(self.client):
            return
        assert self.state is not None
        if self.state != CLIENTS[self.client]:
           set_renderer(enable=self.state)

def connect(use_gui=True, shadows=True, color=None, width=None, height=None):
    # Shared Memory: execute the physics simulation and rendering in a separate process
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/vrminitaur.py#L7
    # make sure to compile pybullet with PYBULLET_USE_NUMPY enabled
    if use_gui and ('DISPLAY' not in os.environ):
        use_gui = False
        print('No display detected!')
    method = p.GUI if use_gui else p.DIRECT
    with HideOutput():
        #  --window_backend=2 --render_device=0'
        # options="--mp4=\"test.mp4\' --mp4fps=240"
        options = ''
        if color is not None:
            options += '--background_color_red={} --background_color_green={} --background_color_blue={}'.format(*color)
        if width is not None:
            options += '--width={}'.format(width)
        if height is not None:
            options += '--height={}'.format(height)
        sim_id = p.connect(method, options=options) # key=None,
        #sim_id = p.connect(p.GUI, options='--opengl2') if use_gui else p.connect(p.DIRECT)

    assert 0 <= sim_id
    #sim_id2 = p.connect(p.SHARED_MEMORY)
    #print(sim_id, sim_id2)
    CLIENTS[sim_id] = True if use_gui else None
    if use_gui:
        # p.COV_ENABLE_PLANAR_REFLECTION
        # p.COV_ENABLE_SINGLE_STEP_RENDERING
        disable_preview()
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, False, physicsClientId=sim_id) # TODO: does this matter?
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, shadows, physicsClientId=sim_id)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, False, physicsClientId=sim_id) # mouse moves meshes
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, False, physicsClientId=sim_id)

    # you can also use GUI mode, for faster OpenGL rendering (instead of TinyRender CPU)
    #visualizer_options = {
    #    p.COV_ENABLE_WIREFRAME: 1,
    #    p.COV_ENABLE_SHADOWS: 0,
    #    p.COV_ENABLE_RENDERING: 0,
    #    p.COV_ENABLE_TINY_RENDERER: 1,
    #    p.COV_ENABLE_RGB_BUFFER_PREVIEW: 0,
    #    p.COV_ENABLE_DEPTH_BUFFER_PREVIEW: 0,
    #    p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW: 0,
    #    p.COV_ENABLE_VR_RENDER_CONTROLLERS: 0,
    #    p.COV_ENABLE_VR_PICKING: 0,
    #    p.COV_ENABLE_VR_TELEPORTING: 0,
    #}
    #for pair in visualizer_options.items():
    #    p.configureDebugVisualizer(*pair)
    return sim_id

def clip(value, min_value=-INF, max_value=+INF):
    return min(max(min_value, value), max_value)

def threaded_input(*args, **kwargs):
    # OS X doesn't multi-thread the OpenGL visualizer
    # http://openrave.org/docs/0.8.2/_modules/openravepy/misc/#SetViewerUserThread
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/userData.py
    # https://github.com/bulletphysics/bullet3/tree/master/examples/ExampleBrowser
    #from pybullet_utils import bullet_client
    #from pybullet_utils.bullet_client import BulletClient
    #server = bullet_client.BulletClient(connection_mode=p.SHARED_MEMORY_SERVER) # GUI_SERVER
    #sim_id = p.connect(p.GUI)
    #print(dir(server))
    #client = bullet_client.BulletClient(connection_mode=p.SHARED_MEMORY)
    #sim_id = p.connect(p.SHARED_MEMORY)

    #threading = __import__('threading')
    import threading
    data = []
    thread = threading.Thread(target=lambda: data.append(input(*args, **kwargs)), args=[])
    thread.start()
    #threading.enumerate()
    #thread_id = 0
    #for tid, tobj in threading._active.items():
    #    if tobj is thread:
    #        thread_id = tid
    #        break
    try:
        while thread.is_alive():
            update_viewer()
    finally:
        thread.join()
    return data[-1]

def disconnect():
    # TODO: change CLIENT?
    if CLIENT in CLIENTS:
        del CLIENTS[CLIENT]
    with HideOutput():
        return p.disconnect(physicsClientId=CLIENT)

def is_connected():
    #return p.isConnected(physicsClientId=CLIENT)
    return p.getConnectionInfo(physicsClientId=CLIENT)['isConnected']

def get_connection(client=None):
    return p.getConnectionInfo(physicsClientId=get_client(client))['connectionMethod']

def has_gui(client=None):
    return get_connection(get_client(client)) == p.GUI

def get_data_path():
    import pybullet_data
    return pybullet_data.getDataPath()

def add_data_path(data_path=None):
    if data_path is None:
        data_path = get_data_path()
    p.setAdditionalSearchPath(data_path)
    return data_path

GRAVITY = 9.8

def enable_gravity():
    p.setGravity(0, 0, -GRAVITY, physicsClientId=CLIENT)

def disable_gravity():
    p.setGravity(0, 0, 0, physicsClientId=CLIENT)

def step_simulation():
    p.stepSimulation(physicsClientId=CLIENT)

def update_scene():
    # TODO: https://github.com/bulletphysics/bullet3/pull/3331
    p.performCollisionDetection(physicsClientId=CLIENT)

def set_real_time(real_time):
    p.setRealTimeSimulation(int(real_time), physicsClientId=CLIENT)

def enable_real_time():
    set_real_time(True)

def disable_real_time():
    set_real_time(False)

def update_state():
    # TODO: this doesn't seem to automatically update still
    disable_gravity()
    #step_simulation()
    #for body in get_bodies():
    #    for link in get_links(body):
    #        # if set to 1 (or True), the Cartesian world position/orientation
    #        # will be recomputed using forward kinematics.
    #        get_link_state(body, link)
    #for body in get_bodies():
    #    get_pose(body)
    #    for joint in get_joints(body):
    #        get_joint_position(body, joint)
    #p.getKeyboardEvents()
    #p.getMouseEvents()

def reset_simulation():
    # RESET_USE_SIMPLE_BROADPHASE
    # RESET_USE_DEFORMABLE_WORLD
    # RESET_USE_DISCRETE_DYNAMICS_WORLD
    p.resetSimulation(physicsClientId=CLIENT)

#####################################

Pixel = namedtuple('Pixel', ['row', 'column'])

def get_camera_matrix(width, height, fx, fy=None):
    if fy is None:
        fy = fx
    #cx, cy = width / 2., height / 2.
    cx, cy = (width - 1) / 2., (height - 1) / 2.
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])

def clip_pixel(pixel, width, height):
    x, y = pixel # TODO: row, column instead?
    return clip(x, 0, width-1), clip(y, 0, height-1)

def ray_from_pixel(camera_matrix, pixel):
    return np.linalg.inv(camera_matrix).dot(np.append(pixel, 1))

def pixel_from_ray(camera_matrix, ray):
    return camera_matrix.dot(np.array(ray) / ray[2])[:2]

def dimensions_from_camera_matrix(camera_matrix):
    cx, cy = np.array(camera_matrix)[:2, 2]
    width, height = (2*cx + 1), (2*cy + 1)
    return width, height

def get_field_of_view(camera_matrix):
    dimensions = np.array(dimensions_from_camera_matrix(camera_matrix))
    focal_lengths = np.array([camera_matrix[i, i] for i in range(2)])
    return 2*np.arctan(np.divide(dimensions, 2*focal_lengths))

def get_focal_lengths(dims, fovs):
    return np.divide(dims, np.tan(fovs / 2)) / 2

def pixel_from_point(camera_matrix, point_camera):
    px, py = pixel_from_ray(camera_matrix, point_camera)
    width, height = dimensions_from_camera_matrix(camera_matrix)
    if (0 <= px < width) and (0 <= py < height):
        r, c = np.floor([py, px]).astype(int)
        return Pixel(r, c)
    return None

def get_image_aabb(camera_matrix):
    upper = np.array(dimensions_from_camera_matrix(camera_matrix)) - 1
    lower = np.zeros(upper.shape)
    return AABB(lower, upper)

def get_visible_aabb(camera_matrix, rays):
    image_aabb = get_image_aabb(camera_matrix)
    rays_aabb = aabb_from_points([pixel_from_ray(camera_matrix, ray) for ray in rays])
    intersection = aabb_intersection(image_aabb, rays_aabb)
    if intersection is None:
        return intersection
    return AABB(*np.array(intersection).astype(int))

def draw_lines_on_image(img_array, points, color='red', **kwargs):
    from PIL import Image, ImageDraw
    source_img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(source_img)
    draw.line(list(map(tuple, points)), fill=color, **kwargs)
    return np.array(source_img)

def draw_box_on_image(img_array, aabb, color='red', **kwargs):
    # https://github.com/caelan/ROS-Labeler/blob/master/main.py
    # https://github.mit.edu/caelan/rl-plan/blob/master/planar_ml/rect_cnn.py
    # https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
    # TODO: annotate boxes with text
    from PIL import Image, ImageDraw
    source_img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(source_img)
    #box = list(np.array(aabb).astype(int).flatten())
    box = list(map(tuple, aabb))
    draw.rectangle(box, fill=None, outline=color, **kwargs)
    return np.array(source_img)

def extract_box_from_image(img_array, box):
    (x1, y1), (x2, y2) = np.array(box).astype(int)
    return img_array[y1:y2+1, x1:x2+1, ...]

#####################################

CameraInfo = namedtuple('CameraInfo', ['width', 'height', 'viewMatrix', 'projectionMatrix', 'cameraUp', 'cameraForward',
                                       'horizontal', 'vertical', 'yaw', 'pitch', 'dist', 'target'])

def get_camera():
    return CameraInfo(*p.getDebugVisualizerCamera(physicsClientId=CLIENT))

def set_camera(yaw, pitch, distance, target_position=np.zeros(3)):
    # TODO: in degrees
    p.resetDebugVisualizerCamera(distance, yaw, pitch, target_position, physicsClientId=CLIENT)

def get_pitch(point):
    dx, dy, dz = point
    return np.math.atan2(dz, np.sqrt(dx ** 2 + dy ** 2))

def get_yaw(point):
    dx, dy = point[:2]
    return np.math.atan2(dy, dx)

def set_camera_pose(camera_point, target_point=np.zeros(3)):
    delta_point = np.array(target_point) - np.array(camera_point)
    distance = np.linalg.norm(delta_point)
    yaw = get_yaw(delta_point) - np.pi/2 # TODO: hack
    pitch = get_pitch(delta_point)
    p.resetDebugVisualizerCamera(distance, math.degrees(yaw), math.degrees(pitch),
                                 target_point, physicsClientId=CLIENT)

def set_camera_pose2(world_from_camera, distance=2):
    target_camera = np.array([0, 0, distance])
    target_world = tform_point(world_from_camera, target_camera)
    camera_world = point_from_pose(world_from_camera)
    set_camera_pose(camera_world, target_world)
    #roll, pitch, yaw = euler_from_quat(quat_from_pose(world_from_camera))
    # TODO: assert that roll is about zero?
    #p.resetDebugVisualizerCamera(cameraDistance=distance, cameraYaw=math.degrees(yaw), cameraPitch=math.degrees(-pitch),
    #                             cameraTargetPosition=target_world, physicsClientId=CLIENT)

CameraImage = namedtuple('CameraImage', ['rgbPixels', 'depthPixels', 'segmentationMaskBuffer',
                                         'camera_pose', 'camera_matrix'])
#CameraImage = namedtuple('CameraImage', ['rgb', 'depth', 'segmentation', 'camera_pose'])

def demask_pixel(pixel):
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/segmask_linkindex.py
    # Not needed when p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX is not enabled
    #if 0 <= pixel:
    #    return None
    # Returns a large value when undefined
    body = pixel & ((1 << 24) - 1)
    link = (pixel >> 24) - 1
    return body, link

def save_image(filename, rgba):
    import imageio
    imageio.imwrite(filename, rgba)
    # import scipy.misc
    # if filename.endswith('.jpg'):
    #     scipy.misc.imsave(filename, rgba[:, :, :3])
    # elif filename.endswith('.png'):
    #     scipy.misc.imsave(filename, rgba)  # (480, 640, 4)
    #     # scipy.misc.toimage(image_array, cmin=0.0, cmax=...).save('outfile.jpg')
    # else:
    #     raise ValueError(filename)
    print('Saved image at {}'.format(filename))

def get_projection_matrix(width, height, vertical_fov, near, far):
    """
    OpenGL projection matrix
    :param width:
    :param height:
    :param vertical_fov: vertical field of view in radians
    :param near:
    :param far:
    :return:
    """
    # http://ksimek.github.io/2013/08/13/intrinsic/
    # http://www.songho.ca/opengl/gl_projectionmatrix.html
    # http://www.songho.ca/opengl/gl_transform.html#matrix
    # https://www.edmundoptics.fr/resources/application-notes/imaging/understanding-focal-length-and-field-of-view/
    # gluPerspective() requires only 4 parameters; vertical field of view (FOV),
    # the aspect ratio of width to height and the distances to near and far clipping planes.
    aspect = float(width) / height
    fov_degrees = math.degrees(vertical_fov)
    projection_matrix = p.computeProjectionMatrixFOV(fov=fov_degrees, aspect=aspect,
                                                     nearVal=near, farVal=far, physicsClientId=CLIENT)
    #projection_matrix = p.computeProjectionMatrix(left=0, right=width, top=height, bottom=0,
    #                                              near=near, far=far, physicsClientId=CLIENT)
    return projection_matrix
    #return np.reshape(projection_matrix, [4, 4])

def image_from_segmented(segmented, color_from_body=None):
    if color_from_body is None:
        bodies = get_bodies()
        color_from_body = dict(zip(bodies, spaced_colors(len(bodies))))
    image = np.zeros(segmented.shape[:2] + (3,))
    for r in range(segmented.shape[0]):
        for c in range(segmented.shape[1]):
            body, link = segmented[r, c, :]
            image[r, c, :] = color_from_body.get(body, BLACK)[:3] # TODO: alpha
    return image

def get_image_flags(segment=False, segment_links=False):
    if segment:
        if segment_links:
            return p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        return 0 # TODO: adjust output dimension when not segmenting links
    return p.ER_NO_SEGMENTATION_MASK

def extract_segmented(seg_image):
    segmented = np.zeros(seg_image.shape + (2,))
    for r in range(segmented.shape[0]):
        for c in range(segmented.shape[1]):
            pixel = seg_image[r, c]
            segmented[r, c, :] = demask_pixel(pixel)
    return segmented

def get_image(camera_pos, target_pos, width=640, height=480, vertical_fov=60.0, near=0.02, far=5.0,
              tiny=False, segment=False, **kwargs):
    # computeViewMatrixFromYawPitchRoll
    up_vector = [0, 0, 1] # up vector of the camera, in Cartesian world coordinates
    view_matrix = p.computeViewMatrix(cameraEyePosition=camera_pos, cameraTargetPosition=target_pos,
                                      cameraUpVector=up_vector, physicsClientId=CLIENT)
    projection_matrix = get_projection_matrix(width, height, vertical_fov, near, far)

    # p.isNumpyEnabled() # copying pixels from C/C++ to Python can be really slow for large images, unless you compile PyBullet using NumPy
    flags = get_image_flags(segment=segment, **kwargs)
    # DIRECT mode has no OpenGL, so it requires ER_TINY_RENDERER
    renderer = p.ER_TINY_RENDERER if tiny else p.ER_BULLET_HARDWARE_OPENGL
    rgb, d, seg = p.getCameraImage(width, height,
                                   viewMatrix=view_matrix,
                                   projectionMatrix=projection_matrix,
                                   shadow=False, # only applies to ER_TINY_RENDERER
                                   flags=flags,
                                   renderer=renderer,
                                   physicsClientId=CLIENT)[2:]

    depth = far * near / (far - (far - near) * d)
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/pointCloudFromCameraImage.py
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/getCameraImageTest.py
    segmented = None
    if segment:
        segmented = extract_segmented(seg)

    camera_tform = np.reshape(view_matrix, [4, 4])
    camera_tform[:3, 3] = camera_pos
    view_pose = multiply(pose_from_tform(camera_tform), Pose(euler=Euler(roll=PI)))

    focal_length = get_focal_lengths(height, vertical_fov) # TODO: horizontal_fov
    camera_matrix = get_camera_matrix(width, height, focal_length)

    return CameraImage(rgb, depth, segmented, view_pose, camera_matrix)

def get_image_at_pose(camera_pose, camera_matrix, far=5.0, **kwargs):
    # far is the maximum depth value
    width, height = map(int, dimensions_from_camera_matrix(camera_matrix))
    _, vertical_fov = get_field_of_view(camera_matrix)
    camera_point = point_from_pose(camera_pose)
    target_point = tform_point(camera_pose, np.array([0, 0, far]))
    return get_image(camera_point, target_point, width=width, height=height,
                     vertical_fov=vertical_fov, far=far, **kwargs)

def set_default_camera(yaw=160, pitch=-35, distance=2.5):
    # TODO: deprecate
    set_camera(yaw, pitch, distance, Point())

#####################################

def save_state():
    return p.saveState(physicsClientId=CLIENT)

def restore_state(state_id):
    p.restoreState(stateId=state_id, physicsClientId=CLIENT)

def save_bullet(filename):
    p.saveBullet(filename, physicsClientId=CLIENT)

def restore_bullet(filename):
    p.restoreState(fileName=filename, physicsClientId=CLIENT)

#####################################

# Geometry

#Pose = namedtuple('Pose', ['position', 'orientation'])

def Point(x=0., y=0., z=0.):
    return np.array([x, y, z])

def Euler(roll=0., pitch=0., yaw=0.):
    return np.array([roll, pitch, yaw])

def Pose(point=None, euler=None):
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return point, quat_from_euler(euler)

def Pose2d(x=0., y=0., yaw=0.):
    return np.array([x, y, yaw])

def invert(pose):
    point, quat = pose
    return p.invertTransform(point, quat)

def multiply(*poses):
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = p.multiplyTransforms(pose[0], pose[1], *next_pose)
    return pose

def invert_quat(quat):
    pose = (unit_point(), quat)
    return quat_from_pose(invert(pose))

def multiply_quats(*quats):
    return quat_from_pose(multiply(*[(unit_point(), quat) for quat in quats]))

def unit_from_theta(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def quat_from_euler(euler):
    return p.getQuaternionFromEuler(euler) # TODO: extrinsic (static) vs intrinsic (rotating)

def euler_from_quat(quat):
    return p.getEulerFromQuaternion(quat) # rotation around fixed axis

def intrinsic_euler_from_quat(quat):
    #axes = 'sxyz' if static else 'rxyz'
    return euler_from_quaternion(quat, axes='rxyz')

def unit_point():
    return (0., 0., 0.)

def unit_quat():
    return quat_from_euler([0, 0, 0]) # [X,Y,Z,W]

def quat_from_axis_angle(axis, angle): # axis-angle
    #return get_unit_vector(np.append(vec, [angle]))
    return np.append(math.sin(angle/2) * get_unit_vector(axis), [math.cos(angle / 2)])

def unit_pose():
    return (unit_point(), unit_quat())

def get_length(vec, norm=2):
    return np.linalg.norm(vec, ord=norm)

def get_difference(p1, p2):
    assert len(p1) == len(p2)
    return np.array(p2) - np.array(p1)

def get_distance(p1, p2, **kwargs):
    return get_length(get_difference(p1, p2), **kwargs)

def angle_between(vec1, vec2):
    inner_product = np.dot(vec1, vec2) / (get_length(vec1) * get_length(vec2))
    return math.acos(clip(inner_product, min_value=-1., max_value=+1.))

def get_angle(q1, q2):
    return get_yaw(np.array(q2) - np.array(q1))

def get_unit_vector(vec):
    norm = get_length(vec)
    if norm == 0:
        return vec
    return np.array(vec) / norm

def z_rotation(theta):
    return quat_from_euler([0, 0, theta])

def matrix_from_quat(quat):
    return np.array(p.getMatrixFromQuaternion(quat, physicsClientId=CLIENT)).reshape(3, 3)

def quat_from_matrix(rot):
    matrix = np.eye(4)
    matrix[:3, :3] = rot[:3, :3]
    return quaternion_from_matrix(matrix)

def point_from_tform(tform):
    return np.array(tform)[:3,3]

def matrix_from_tform(tform):
    return np.array(tform)[:3,:3]

def point_from_pose(pose):
    return pose[0]

def quat_from_pose(pose):
    return pose[1]

def tform_from_pose(pose):
    (point, quat) = pose
    tform = np.eye(4)
    tform[:3,3] = point
    tform[:3,:3] = matrix_from_quat(quat)
    return tform

def pose_from_tform(tform):
    return point_from_tform(tform), quat_from_matrix(matrix_from_tform(tform))

def normalize_interval(value, interval=UNIT_LIMITS):
    lower, upper = interval
    assert lower <= upper
    return (value - lower) / (upper - lower)

def rescale_interval(value, old_interval=UNIT_LIMITS, new_interval=UNIT_LIMITS):
    lower, upper = new_interval
    return convex_combination(lower, upper, w=normalize_interval(value, old_interval))

def wrap_interval(value, interval=UNIT_LIMITS):
    lower, upper = interval
    assert lower <= upper
    return (value - lower) % (upper - lower) + lower

def interval_distance(value1, value2, interval=UNIT_LIMITS):
    value1 = wrap_interval(value1, interval)
    value2 = wrap_interval(value2, interval)
    if value1 > value2:
        value1, value2 = value2, value1
    lower, upper = interval
    return min(value2 - value1, (value1 - lower) + (upper - value2))

def circular_interval(lower=-PI): # [-np.pi, np.pi)
    return Interval(lower, lower + 2*PI)

def wrap_angle(theta, **kwargs):
    return wrap_interval(theta, interval=circular_interval(**kwargs))

def circular_difference(theta2, theta1, **kwargs):
    return wrap_angle(theta2 - theta1, **kwargs)

def cache_decorator(function):
    """
    A decorator for class methods, replaces @property
    but will store and retrieve function return values
    in object cache.
    Parameters
    ------------
    function : method
      This is used as a decorator:
      ```
      @cache_decorator
      def foo(self, things):
        return 'happy days'
      ```
    """
    # https://github.com/mikedh/trimesh/blob/60dae2352875f48c4476e01052829e8b9166d9e5/trimesh/caching.py#L64
    from functools import wraps
    #from functools import cached_property # TODO: New in version 3.8

    # use wraps to preserve docstring
    @wraps(function)
    def get_cached(*args, **kwargs):
        """
        Only execute the function if its value isn't stored
        in cache already.
        """
        self = args[0]
        # use function name as key in cache
        name = function.__name__
        # do the dump logic ourselves to avoid
        # verifying cache twice per call
        self._cache.verify()
        # access cache dict to avoid automatic validation
        # since we already called cache.verify manually
        if name in self._cache.cache:
            # already stored so return value
            return self._cache.cache[name]
        # value not in cache so execute the function
        value = function(*args, **kwargs)
        # store the value
        if self._cache.force_immutable and hasattr(
                value, 'flags') and len(value.shape) > 0:
            value.flags.writeable = False

        self._cache.cache[name] = value

        return value

    # all cached values are also properties
    # so they can be accessed like value attributes
    # rather than functions
    return property(get_cached)

def cached_fn(fn, cache=True, **global_kargs):
    # https://docs.python.org/3/library/functools.html#functools.cache
    def normal(*args, **local_kwargs):
        kwargs = dict(global_kargs)
        kwargs.update(local_kwargs)
        return fn(*args, **kwargs)
    if not cache:
        return normal

    #from functools import cache # # New in version 3.9
    #from functools import lru_cache as cache
    #@cache(maxsize=None, typed=False)
    @cache_decorator
    def wrapped(*args, **local_kwargs):
        return normal(*args, **local_kwargs)
    return wrapped
def base_values_from_pose(pose, tolerance=1e-3):
    (point, quat) = pose
    x, y, _ = point
    roll, pitch, yaw = euler_from_quat(quat)
    assert (abs(roll) < tolerance) and (abs(pitch) < tolerance)
    return Pose2d(x, y, yaw)

pose2d_from_pose = base_values_from_pose

def pose_from_base_values(base_values, default_pose=unit_pose()):
    x, y, yaw = base_values
    _, _, z = point_from_pose(default_pose)
    roll, pitch, _ = euler_from_quat(quat_from_pose(default_pose))
    return (x, y, z), quat_from_euler([roll, pitch, yaw])

def quat_combination(quat1, quat2, fraction=0.5):
    #return p.getQuaternionSlerp(quat1, quat2, interpolationFraction=fraction)
    return quaternion_slerp(quat1, quat2, fraction)

def quat_angle_between(quat0, quat1):
    # #p.computeViewMatrixFromYawPitchRoll()
    # q0 = unit_vector(quat0[:4])
    # q1 = unit_vector(quat1[:4])
    # d = clip(np.dot(q0, q1), min_value=-1., max_value=+1.)
    # angle = math.acos(d)

    # TODO: angle_between
    delta = p.getDifferenceQuaternion(quat0, quat1)
    d = clip(delta[-1], min_value=-1., max_value=1.)
    angle = math.acos(d)
    return angle

def all_between(lower_limits, values, upper_limits):
    assert len(lower_limits) == len(values)
    assert len(values) == len(upper_limits)
    return np.less_equal(lower_limits, values).all() and \
           np.less_equal(values, upper_limits).all()

def convex_combination(x, y, w=0.5):
    return (1-w)*np.array(x) + w*np.array(y)

#####################################

# Bodies

def get_bodies():
    return [p.getBodyUniqueId(i, physicsClientId=CLIENT)
            for i in range(p.getNumBodies(physicsClientId=CLIENT))]

BodyInfo = namedtuple('BodyInfo', ['base_name', 'body_name'])

def get_body_info(body):
    return BodyInfo(*p.getBodyInfo(body, physicsClientId=CLIENT))

def get_base_name(body):
    return get_body_info(body).base_name.decode(encoding='UTF-8')

def get_body_name(body):
    return get_body_info(body).body_name.decode(encoding='UTF-8')

def get_name(body):
    name = get_body_name(body)
    if name == '':
        name = 'body'
    return '{}{}'.format(name, int(body))

def has_body(name):
    try:
        body_from_name(name)
    except ValueError:
        return False
    return True

def body_from_name(name):
    for body in get_bodies():
        if get_body_name(body) == name:
            return body
    raise ValueError(name)

def is_b1_on_b2(b1, b2):
    AABB = get_aabb(b2, -1)
    pose = get_pose(b1)
    isOn = pose[0][0] < AABB[1][0] and pose[0][0] > AABB[0][0]
    isOn = isOn and pose[0][1] < AABB[1][1] and pose[0][1] > AABB[0][1]
    isOn = isOn and pose[0][2] > AABB[0][2]

    return isOn

def is_pose_on_r(pose, r):
    AABB = get_aabb(r)
    isOn = pose[0][0] <= AABB[1][0] and pose[0][0] >= AABB[0][0]
    isOn = isOn and pose[0][1] <= AABB[1][1] and pose[0][1] >= AABB[0][1]
    isOn = isOn and pose[0][2] >= AABB[0][2]
    return isOn

def remove_body(body):
    if (CLIENT, body) in INFO_FROM_BODY:
        del INFO_FROM_BODY[CLIENT, body]
    return p.removeBody(body, physicsClientId=CLIENT)

def get_pose(body):
    return p.getBasePositionAndOrientation(body, physicsClientId=CLIENT)
    #return np.concatenate([point, quat])

def get_point(body):
    return get_pose(body)[0]

def get_quat(body):
    return get_pose(body)[1] # [x,y,z,w]

def get_euler(body):
    return euler_from_quat(get_quat(body))

def get_base_values(body):
    return base_values_from_pose(get_pose(body))

def set_pose(body, pose):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(body, point, quat, physicsClientId=CLIENT)

def set_point(body, point):
    print('###################')
    print(get_quat(body))
    set_pose(body, (point, get_quat(body)))

def set_quat(body, quat):
    set_pose(body, (get_point(body), quat))

def set_euler(body, euler):
    set_quat(body, quat_from_euler(euler))

def pose_from_pose2d(pose2d, z=0.):
    x, y, theta = pose2d
    return Pose(Point(x=x, y=y, z=z), Euler(yaw=theta))

def get_COM(bodies, b2, b1 = None, p1 = None, poses = None):
    totalMass = get_mass(b2)
    comR = get_pose(b2)[0]
    comMX =  totalMass * comR[0]
    comMY = totalMass * comR[1]
    comMZ = totalMass * comR[2]
    num_obs = 1
    if poses is not None:
        for body in poses:
            if(body != b2 and body > 3):
                pos = poses[body][0]
                mass = get_mass(body)
                ratio = totalMass / (totalMass + mass)
                totalMass = totalMass + mass
                comMX = (comMX * ratio) + (mass*pos[0]/totalMass)
                comMY = (comMY * ratio) + (mass*pos[1]/totalMass)
                comMZ = (comMZ * ratio) + (mass*pos[2]/totalMass)
    else:
        for body in bodies:
            if(body != b2 and body > 3 and is_b1_on_b2(body, b2)):
                pos = get_pose(body)[0]
                mass = get_mass(body)
                ratio = totalMass / (totalMass + mass)
                totalMass = totalMass + mass
                comMX = (comMX * ratio) + (mass*pos[0]/totalMass)
                comMY = (comMY * ratio) + (mass*pos[1]/totalMass)
                comMZ = (comMZ * ratio) + (mass*pos[2]/totalMass)
    if b1 is not None and p1 is not None:
        pos = p1.value[0]
        mass = get_mass(b1)
        ratio = totalMass / (totalMass + mass)
        totalMass = totalMass + mass
        comMX = (comMX * ratio) + (mass*pos[0]/totalMass)
        comMY = (comMY * ratio) + (mass*pos[1]/totalMass)
        comMZ = (comMZ * ratio) + (mass*pos[2]/totalMass)
    return (comMX, comMY, comMZ), totalMass

def set_base_values(body, values):
    _, _, z = get_point(body)
    x, y, theta = values
    set_point(body, (x, y, z))
    set_quat(body, z_rotation(theta))

def get_velocity(body):
    linear, angular = p.getBaseVelocity(body, physicsClientId=CLIENT)
    return linear, angular # [x,y,z], [wx,wy,wz]

def set_velocity(body, linear=None, angular=None):
    if linear is not None:
        p.resetBaseVelocity(body, linearVelocity=linear, physicsClientId=CLIENT)
    if angular is not None:
        p.resetBaseVelocity(body, angularVelocity=angular, physicsClientId=CLIENT)

def is_rigid_body(body):
    for joint in get_joints(body):
        if is_movable(body, joint):
            return False
    return True

def is_fixed_base(body):
    return get_mass(body) == STATIC_MASS

def dump_joint(body, joint):
    print('Joint id: {} | Name: {} | Type: {} | Circular: {} | Lower: {:.3f} | Upper: {:.3f}'.format(
        joint, get_joint_name(body, joint), JOINT_TYPES[get_joint_type(body, joint)],
        is_circular(body, joint), *get_joint_limits(body, joint)))

def dump_link(body, link):
    joint = parent_joint_from_link(link)
    joint_name = JOINT_TYPES[get_joint_type(body, joint)] if is_fixed(body, joint) else get_joint_name(body, joint)
    print('Link id: {} | Name: {} | Joint: {} | Parent: {} | Mass: {} | Collision: {} | Visual: {}'.format(
        link, get_link_name(body, link), joint_name,
        get_link_name(body, get_link_parent(body, link)), get_mass(body, link),
        len(get_collision_data(body, link)), NULL_ID))  # len(get_visual_data(body, link))))
    # print(get_joint_parent_frame(body, link))
    # print(map(get_data_geometry, get_visual_data(body, link)))
    # print(map(get_data_geometry, get_collision_data(body, link)))

def dump(*args, **kwargs):
    load = 'poop'
    print(load)
    return load

def dump_body(body, fixed=False, links=True):
    print('Body id: {} | Name: {} | Rigid: {} | Fixed: {}'.format(
        body, get_body_name(body), is_rigid_body(body), is_fixed_base(body)))
    for joint in get_joints(body):
        if fixed or is_movable(body, joint):
            dump_joint(body, joint)

    if not links:
        return
    base_link = NULL_ID
    print('Link id: {} | Name: {} | Mass: {} | Collision: {} | Visual: {}'.format(
        base_link, get_base_name(body), get_mass(body),
        len(get_collision_data(body, base_link)), NULL_ID)) # len(get_visual_data(body, link))))
    for link in get_links(body):
        dump_link(body, link)

def dump_world():
    for body in get_bodies():
        dump_body(body)
        print()

#####################################

# Joints

JOINT_TYPES = {
    p.JOINT_REVOLUTE: 'revolute', # 0
    p.JOINT_PRISMATIC: 'prismatic', # 1
    p.JOINT_SPHERICAL: 'spherical', # 2
    p.JOINT_PLANAR: 'planar', # 3
    p.JOINT_FIXED: 'fixed', # 4
    p.JOINT_POINT2POINT: 'point2point', # 5
    p.JOINT_GEAR: 'gear', # 6
}

def get_num_joints(body):
    return p.getNumJoints(body, physicsClientId=CLIENT)

def get_joints(body):
    return list(range(get_num_joints(body)))

def get_joint(body, joint_or_name):
    if type(joint_or_name) is str:
        return joint_from_name(body, joint_or_name)
    return joint_or_name

JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])

def get_joint_info(body, joint):
    return JointInfo(*p.getJointInfo(body, joint, physicsClientId=CLIENT))

def get_joint_name(body, joint):
    return get_joint_info(body, joint).jointName.decode('UTF-8')

def get_joint_names(body, joints):
    return [get_joint_name(body, joint) for joint in joints] # .encode('ascii')

def joint_from_name(body, name):
    for joint in get_joints(body):
        if get_joint_name(body, joint) == name:
            return joint
    raise ValueError(body, name)

def has_joint(body, name):
    try:
        joint_from_name(body, name)
    except ValueError:
        return False
    return True

def joints_from_names(body, names):
    return tuple(joint_from_name(body, name) for name in names)

##########

JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                       'jointReactionForces', 'appliedJointMotorTorque'])

def get_joint_state(body, joint):
    return JointState(*p.getJointState(body, joint, physicsClientId=CLIENT))

def get_joint_position(body, joint):
    return get_joint_state(body, joint).jointPosition

def get_joint_velocity(body, joint):
    return get_joint_state(body, joint).jointVelocity

def get_joint_reaction_force(body, joint):
    return get_joint_state(body, joint).jointReactionForces

def get_joint_torque(body, joint):
    return get_joint_state(body, joint).appliedJointMotorTorque

##########

def get_joint_positions(body, joints): # joints=None):
    return tuple(get_joint_position(body, joint) for joint in joints)

def get_joint_velocities(body, joints):
    return tuple(get_joint_velocity(body, joint) for joint in joints)

def get_joint_torques(body, joints):
    return tuple(get_joint_torque(body, joint) for joint in joints)

##########

def set_joint_state(body, joint, position, velocity):
    p.resetJointState(body, joint, targetValue=position, targetVelocity=velocity, physicsClientId=CLIENT)


def set_joint_position(body, joint, value):
    # TODO: remove targetVelocity=0
    p.resetJointState(body, joint, targetValue=value, targetVelocity=0, physicsClientId=CLIENT)
    # p.setJointMotorControl2(body, joint, p.POSITION_CONTROL, targetPosition=value, targetVelocity=0)
# def set_joint_velocity(body, joint, velocity):
#     p.resetJointState(body, joint, targetVelocity=velocity, physicsClientId=CLIENT) # TODO: targetValue required
def set_joint_position_torque(body, joint, value):
    p.setJointMotorControl2(body, joint, p.POSITION_CONTROL, targetPosition=value, force=get_max_force(body, joint))

def set_joint_positions_torque(body, joints, values, velocities=None):
    max_forces = [get_max_force(body, joint) for joint in joints]
    if velocities is not None:
        p.setJointMotorControlArray(body, joints, p.POSITION_CONTROL, targetPositions=values, forces = max_forces, targetVelocities=velocities)
    else:
        p.setJointMotorControlArray(body, joints, p.POSITION_CONTROL, targetPositions=values, forces = max_forces)

def set_joint_states(body, joints, positions, velocities):
    assert len(joints) == len(positions) == len(velocities)
    for joint, position, velocity in zip(joints, positions, velocities):
        set_joint_state(body, joint, position, velocity)

def set_joint_positions(body, joints, values):
    for joint, value in safe_zip(joints, values):
        set_joint_position(body, joint, value)

# def set_joint_velocities(body, joints, velocities):
#     assert len(joints) == len(velocities)
#     for joint, velocity in zip(joints, velocities):
#         set_joint_velocity(body, joint, velocity)

def get_configuration(body):
    return get_joint_positions(body, get_movable_joints(body))

def set_configuration(body, values):
    set_joint_positions(body, get_movable_joints(body), values)

def modify_configuration(body, joints, positions=None):
    if positions is None:
        positions = get_joint_positions(body, joints)
    configuration = list(get_configuration(body))
    for joint, value in safe_zip(movable_from_joints(body, joints), positions):
        configuration[joint] = value
    return configuration

def get_full_configuration(body):
    # Cannot alter fixed joints
    return get_joint_positions(body, get_joints(body))

def get_labeled_configuration(body):
    movable_joints = get_movable_joints(body)
    return dict(safe_zip(get_joint_names(body, movable_joints),
                         get_joint_positions(body, movable_joints)))

def get_joint_type(body, joint):
    return get_joint_info(body, joint).jointType

def is_fixed(body, joint):
    return get_joint_type(body, joint) == p.JOINT_FIXED

def is_movable(body, joint):
    return not is_fixed(body, joint)

def prune_fixed_joints(body, joints):
    return [joint for joint in joints if is_movable(body, joint)]

def get_movable_joints(body):
    return prune_fixed_joints(body, get_joints(body))

def joint_from_movable(body, index):
    return get_joints(body)[index]

def movable_from_joints(body, joints):
    movable_from_original = {o: m for m, o in enumerate(get_movable_joints(body))}
    return [movable_from_original[joint] for joint in joints]

def is_circular(body, joint):
    joint_info = get_joint_info(body, joint)
    if joint_info.jointType == p.JOINT_FIXED:
        return False
    return joint_info.jointUpperLimit < joint_info.jointLowerLimit

def get_joint_limits(body, joint):
    # TODO: make a version for several joints?
    if is_circular(body, joint):
        # TODO: return UNBOUNDED_LIMITS
        return CIRCULAR_LIMITS
    joint_info = get_joint_info(body, joint)
    return joint_info.jointLowerLimit, joint_info.jointUpperLimit

def get_min_limit(body, joint):
    # TODO: rename to min_position
    return get_joint_limits(body, joint)[0]

def get_min_limits(body, joints):
    return [get_min_limit(body, joint) for joint in joints]

def get_max_limit(body, joint):
    return get_joint_limits(body, joint)[1]

def get_max_limits(body, joints):
    return [get_max_limit(body, joint) for joint in joints]

def get_max_velocity(body, joint):
    return get_joint_info(body, joint).jointMaxVelocity

def get_max_velocities(body, joints):
    return tuple(get_max_velocity(body, joint) for joint in joints)

def get_max_force(body, joint):
    return get_joint_info(body, joint).jointMaxForce

def get_joint_q_index(body, joint):
    return get_joint_info(body, joint).qIndex

def get_joint_v_index(body, joint):
    return get_joint_info(body, joint).uIndex

def get_joint_axis(body, joint):
    return get_joint_info(body, joint).jointAxis

def get_joint_parent_frame(body, joint):
    joint_info = get_joint_info(body, joint)
    return joint_info.parentFramePos, joint_info.parentFrameOrn

def violates_limit(body, joint, value):
    if is_circular(body, joint):
        return False
    lower, upper = get_joint_limits(body, joint)
    return (value < lower) or (upper < value)

def violates_limits(body, joints, values):
    return any(violates_limit(body, joint, value) for joint, value in zip(joints, values))

def wrap_position(body, joint, position):
    if is_circular(body, joint):
        return wrap_angle(position)
    return position

def wrap_positions(body, joints, positions):
    assert len(joints) == len(positions)
    return [wrap_position(body, joint, position)
            for joint, position in zip(joints, positions)]

def get_custom_limits(body, joints, custom_limits={}, circular_limits=UNBOUNDED_LIMITS):
    joint_limits = []
    for joint in joints:
        if joint in custom_limits:
            joint_limits.append(custom_limits[joint])
        elif is_circular(body, joint):
            joint_limits.append(circular_limits)
        else:
            joint_limits.append(get_joint_limits(body, joint))
    return zip(*joint_limits)

#####################################

# Links

BASE_LINK = -1
STATIC_MASS = 0.5

get_num_links = get_num_joints
get_links = get_joints # Does not include BASE_LINK

def child_link_from_joint(joint):
    return joint # link

def parent_joint_from_link(link):
    return link # joint

def get_all_links(body):
    return [BASE_LINK] + list(get_links(body))

def get_link_name(body, link):
    if link == BASE_LINK:
        return get_base_name(body)
    return get_joint_info(body, link).linkName.decode('UTF-8')

def get_link_names(body, links):
    return [get_link_name(body, link) for link in links]

def get_link_parent(body, link):
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link).parentIndex

parent_link_from_joint = get_link_parent

def link_from_name(body, name):
    if name == get_base_name(body):
        return BASE_LINK
    for link in get_links(body):
        if get_link_name(body, link) == name:
            return link
    raise ValueError(body, name)


def has_link(body, name):
    try:
        link_from_name(body, name)
    except ValueError:
        return False
    return True

LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                     'localInertialFramePosition', 'localInertialFrameOrientation',
                                     'worldLinkFramePosition', 'worldLinkFrameOrientation'])

def get_link_state(body, link, kinematics=True, velocity=True):
    # TODO: the defaults are set to False?
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/pybullet.c

    return LinkState(*p.getLinkState(body, link,
                                     #computeForwardKinematics=kinematics,
                                     #computeLinkVelocity=velocity,
                                     physicsClientId=CLIENT))

def get_com_pose(body, link): # COM = center of mass
    if link == BASE_LINK:
        return get_pose(body)
    link_state = get_link_state(body, link)
    # urdfLinkFrame = comLinkFrame * localInertialFrame.inverse()
    return link_state.linkWorldPosition, link_state.linkWorldOrientation

def get_link_inertial_pose(body, link):
    link_state = get_link_state(body, link)
    return link_state.localInertialFramePosition, link_state.localInertialFrameOrientation

def get_link_pose(body, link):
    if link == BASE_LINK:
        return get_pose(body)
    # if set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
    link_state = get_link_state(body, link) #, kinematics=True, velocity=False)
    return link_state.worldLinkFramePosition, link_state.worldLinkFrameOrientation

def get_relative_pose(body, link1, link2=BASE_LINK):
    world_from_link1 = get_link_pose(body, link1)
    world_from_link2 = get_link_pose(body, link2)
    link2_from_link1 = multiply(invert(world_from_link2), world_from_link1)
    return link2_from_link1

#####################################
def get_connected_components(vertices, edges):
    undirected_edges = defaultdict(set)
    for v1, v2 in edges:
        undirected_edges[v1].add(v2)
        undirected_edges[v2].add(v1)
    clusters = []
    processed = set()
    for v0 in vertices:
        if v0 in processed:
            continue
        processed.add(v0)
        cluster = {v0}
        queue = deque([v0])
        while queue:
            v1 = queue.popleft()
            for v2 in (undirected_edges[v1] - processed):
                processed.add(v2)
                cluster.add(v2)
                queue.append(v2)
        if cluster: # preserves order
            clusters.append(frozenset(cluster))
    return clusters

def get_all_link_parents(body):
    return {link: get_link_parent(body, link) for link in get_links(body)}

def get_all_link_children(body):
    children = {}
    for child, parent in get_all_link_parents(body).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children

def get_link_children(body, link):
    children = get_all_link_children(body)
    return children.get(link, [])

def get_link_ancestors(body, link):
    # Returns in order of depth
    # Does not include link
    parent = get_link_parent(body, link)
    if parent is None:
        return []
    return get_link_ancestors(body, parent) + [parent]

def get_ordered_ancestors(robot, link):
    #return prune_fixed_joints(robot, get_link_ancestors(robot, link)[1:] + [link])
    return get_link_ancestors(robot, link)[1:] + [link]

def get_joint_ancestors(body, joint):
    link = child_link_from_joint(joint)
    return get_link_ancestors(body, link) + [link]

def get_movable_joint_ancestors(body, link):
    return prune_fixed_joints(body, get_joint_ancestors(body, link))

def get_joint_descendants(body, link):
    return list(map(parent_joint_from_link, get_link_descendants(body, link)))

def get_movable_joint_descendants(body, link):
    return prune_fixed_joints(body, get_joint_descendants(body, link))

def get_link_descendants(body, link, test=lambda l: True):
    descendants = []
    for child in get_link_children(body, link):
        if test(child):
            descendants.append(child)
            descendants.extend(get_link_descendants(body, child, test=test))
    return descendants

def get_link_subtree(body, link, **kwargs):
    return [link] + get_link_descendants(body, link, **kwargs)

def are_links_adjacent(body, link1, link2):
    return (get_link_parent(body, link1) == link2) or \
           (get_link_parent(body, link2) == link1)

def get_adjacent_links(body):
    adjacent = set()
    for link in get_links(body):
        parent = get_link_parent(body, link)
        adjacent.add((link, parent))
        #adjacent.add((parent, link))
    return adjacent

def get_adjacent_fixed_links(body):
    return list(filter(lambda item: not is_movable(body, item[0]),
                       get_adjacent_links(body)))

def get_rigid_clusters(body):
    return get_connected_components(vertices=get_all_links(body),
                                    edges=get_adjacent_fixed_links(body))

def assign_link_colors(body, max_colors=3, alpha=1., s=0.5, **kwargs):
    # TODO: graph coloring
    components = sorted(map(list, get_rigid_clusters(body)),
                        key=np.average,
                        #key=min,
                        ) # TODO: only if any have visual data
    num_colors = min(len(components), max_colors)
    colors = spaced_colors(num_colors, s=s, **kwargs)
    colors = islice(cycle(colors), len(components))
    for component, color in zip(components, colors):
        for link in component:
            #print(get_color(body, link=link))
            set_color(body, link=link, color=apply_alpha(color, alpha=alpha))
    return components

def get_fixed_links(body):
    fixed = set()
    for cluster in get_rigid_clusters(body):
        fixed.update(product(cluster, cluster))
    return fixed

#####################################

DynamicsInfo = namedtuple('DynamicsInfo', [
    'mass', 'lateral_friction', 'local_inertia_diagonal', 'local_inertial_pos',  'local_inertial_orn',
    'restitution', 'rolling_friction', 'spinning_friction', 'contact_damping', 'contact_stiffness']) #, 'body_type'])

def get_dynamics_info(body, link=BASE_LINK):
    return DynamicsInfo(*p.getDynamicsInfo(body, link, physicsClientId=CLIENT)[:len(DynamicsInfo._fields)])

get_link_info = get_dynamics_info

def get_mass(body, link=BASE_LINK): # mass in kg
    # TODO: get full mass
    return get_dynamics_info(body, link).mass

def set_dynamics(body, link=BASE_LINK, **kwargs):
    # TODO: iterate over all links
    p.changeDynamics(body, link, physicsClientId=CLIENT, **kwargs)

def set_mass(body, mass, link=BASE_LINK): # mass in kg
    set_dynamics(body, link=link, mass=mass)

def set_static(body):
    for link in get_all_links(body):
        set_mass(body, mass=STATIC_MASS, link=link)

def set_all_static():
    # TODO: mass saver
    disable_gravity()
    for body in get_bodies():
        set_static(body)

def get_joint_inertial_pose(body, joint):
    dynamics_info = get_dynamics_info(body, joint)
    return dynamics_info.local_inertial_pos, dynamics_info.local_inertial_orn

def get_local_link_pose(body, joint):
    parent_joint = parent_link_from_joint(body, joint)

    #world_child = get_link_pose(body, joint)
    #world_parent = get_link_pose(body, parent_joint)
    ##return multiply(invert(world_parent), world_child)
    #return multiply(world_child, invert(world_parent))

    # https://github.com/bulletphysics/bullet3/blob/9c9ac6cba8118544808889664326fd6f06d9eeba/examples/pybullet/gym/pybullet_utils/urdfEditor.py#L169
    parent_com = get_joint_parent_frame(body, joint)
    tmp_pose = invert(multiply(get_joint_inertial_pose(body, joint), parent_com))
    parent_inertia = get_joint_inertial_pose(body, parent_joint)
    #return multiply(parent_inertia, tmp_pose) # TODO: why is this wrong...
    _, orn = multiply(parent_inertia, tmp_pose)
    pos, _ = multiply(parent_inertia, Pose(parent_com[0]))
    return (pos, orn)

#####################################

# Shapes

SHAPE_TYPES = {
    p.GEOM_SPHERE: 'sphere', # 2
    p.GEOM_BOX: 'box', # 3
    p.GEOM_CYLINDER: 'cylinder', # 4
    p.GEOM_MESH: 'mesh', # 5
    p.GEOM_PLANE: 'plane',  # 6
    p.GEOM_CAPSULE: 'capsule',  # 7
    # p.GEOM_FORCE_CONCAVE_TRIMESH
}

# TODO: clean this up to avoid repeated work

def get_box_geometry(width, length, height):
    return {
        'shapeType': p.GEOM_BOX,
        'halfExtents': [width/2., length/2., height/2.]
    }

def get_cylinder_geometry(radius, height):
    return {
        'shapeType': p.GEOM_CYLINDER,
        'radius': radius,
        'length': height,
    }

def get_sphere_geometry(radius):
    return {
        'shapeType': p.GEOM_SPHERE,
        'radius': radius,
    }

def get_capsule_geometry(radius, height):
    return {
        'shapeType': p.GEOM_CAPSULE,
        'radius': radius,
        'length': height,
    }

def get_plane_geometry(normal):
    return {
        'shapeType': p.GEOM_PLANE,
        'planeNormal': normal,
    }

def get_mesh_geometry(path, scale=1.):
    return {
        'shapeType': p.GEOM_MESH,
        'fileName': path,
        'meshScale': scale*np.ones(3),
    }

def get_faces_geometry(mesh, vertex_textures=None, vertex_normals=None, scale=1.):
    # TODO: p.createCollisionShape(p.GEOM_MESH, vertices=[], indices=[])
    # https://github.com/bulletphysics/bullet3/blob/ddc47f932888a6ea3b4e11bd5ce73e8deba0c9a1/examples/pybullet/examples/createMesh.py
    vertices, faces = mesh
    indices = []
    for face in faces:
        assert len(face) == 3
        indices.extend(face)
    geometry = {
        'shapeType': p.GEOM_MESH,
        'meshScale': scale * np.ones(3),
        'vertices': vertices,
        'indices': indices,
        # 'visualFramePosition': None,
        # 'collisionFramePosition': None,
    }
    if vertex_textures is not None:
        geometry['uvs'] = vertex_textures
    if vertex_normals is not None:
        geometry['normals'] = vertex_normals
    return geometry

NULL_ID = -1

def create_collision_shape(geometry, pose=unit_pose()):
    point, quat = pose
    collision_args = {
        'collisionFramePosition': point,
        'collisionFrameOrientation': quat,
        'physicsClientId': CLIENT,
    }
    collision_args.update(geometry)
    if 'length' in collision_args:
        # TODO: pybullet bug visual => length, collision => height
        collision_args['height'] = collision_args['length']
        del collision_args['length']
    return p.createCollisionShape(**collision_args)

def create_visual_shape(geometry, pose=unit_pose(), color=RED, specular=None):
    if (color is None): # or not has_gui():
        return NULL_ID
    point, quat = pose
    visual_args = {
        'rgbaColor': color,
        'visualFramePosition': point,
        'visualFrameOrientation': quat,
        'physicsClientId': CLIENT,
    }
    visual_args.update(geometry)
    if specular is not None:
        visual_args['specularColor'] = specular
    return p.createVisualShape(**visual_args)

def create_shape(geometry, pose=unit_pose(), collision=True, **kwargs):
    collision_id = create_collision_shape(geometry, pose=pose) if collision else NULL_ID
    visual_id = create_visual_shape(geometry, pose=pose, **kwargs) # if collision else NULL_ID

    return collision_id, visual_id

def plural(word):
    exceptions = {'radius': 'radii'}
    if word in exceptions:
        return exceptions[word]
    if word.endswith('s'):
        return word
    return word + 's'

def create_shape_array(geoms, poses, colors=None):
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/pybullet.c
    # createCollisionShape: height
    # createVisualShape: length
    # createCollisionShapeArray: lengths
    # createVisualShapeArray: lengths
    mega_geom = defaultdict(list)
    for geom in geoms:
        extended_geom = get_default_geometry()
        extended_geom.update(geom)
        #extended_geom = geom.copy()
        for key, value in extended_geom.items():
            mega_geom[plural(key)].append(value)

    collision_args = mega_geom.copy()
    for (point, quat) in poses:
        collision_args['collisionFramePositions'].append(point)
        collision_args['collisionFrameOrientations'].append(quat)
    collision_id = p.createCollisionShapeArray(physicsClientId=CLIENT, **collision_args)
    if (colors is None): # or not has_gui():
        return collision_id, NULL_ID

    visual_args = mega_geom.copy()
    for (point, quat), color in zip(poses, colors):
        # TODO: color doesn't seem to work correctly here
        visual_args['rgbaColors'].append(color)
        visual_args['visualFramePositions'].append(point)
        visual_args['visualFrameOrientations'].append(quat)
    visual_id = p.createVisualShapeArray(physicsClientId=CLIENT, **visual_args)
    return collision_id, visual_id

#####################################

def create_body(collision_id=NULL_ID, visual_id=NULL_ID, mass=STATIC_MASS, inertial=NULL_ID):
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id,
                             baseVisualShapeIndex=visual_id, physicsClientId=CLIENT)

CARTESIAN_TYPES = {
    'x': (p.JOINT_PRISMATIC, [1, 0, 0]),
    'y': (p.JOINT_PRISMATIC, [0, 1, 0]),
    'z': (p.JOINT_PRISMATIC, [0, 0, 1]),
    'roll': (p.JOINT_REVOLUTE, [1, 0, 0]),
    'pitch': (p.JOINT_REVOLUTE, [0, 1, 0]),
    'yaw': (p.JOINT_REVOLUTE, [0, 0, 1]),
}

T2 = ['x', 'y']
T3 = ['x', 'y', 'z']

SE2 = T2 + ['yaw']
SE3 = T3 + ['roll', 'pitch', 'yaw']

def create_flying_body(group, collision_id=NULL_ID, visual_id=NULL_ID, mass=STATIC_MASS):
    # TODO: more generally clone the body
    indices = list(range(len(group) + 1))
    masses = len(group) * [STATIC_MASS] + [mass]
    visuals = len(group) * [NULL_ID] + [visual_id]
    collisions = len(group) * [NULL_ID] + [collision_id]
    types = [CARTESIAN_TYPES[joint][0] for joint in group] + [p.JOINT_FIXED]
    #parents = [BASE_LINK] + indices[:-1]
    parents = indices

    assert len(indices) == len(visuals) == len(collisions) == len(types) == len(parents)
    link_positions = len(indices) * [unit_point()]
    link_orientations = len(indices) * [unit_quat()]
    inertial_positions = len(indices) * [unit_point()]
    inertial_orientations = len(indices) * [unit_quat()]
    axes = len(indices) * [unit_point()]
    axes = [CARTESIAN_TYPES[joint][1] for joint in group] + [unit_point()]
    # TODO: no way of specifying joint limits

    return p.createMultiBody(
        baseMass=STATIC_MASS,
        baseCollisionShapeIndex=NULL_ID,
        baseVisualShapeIndex=NULL_ID,
        basePosition=unit_point(),
        baseOrientation=unit_quat(),
        baseInertialFramePosition=unit_point(),
        baseInertialFrameOrientation=unit_quat(),
        linkMasses=masses,
        linkCollisionShapeIndices=collisions,
        linkVisualShapeIndices=visuals,
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkInertialFramePositions=inertial_positions,
        linkInertialFrameOrientations=inertial_orientations,
        linkParentIndices=parents,
        linkJointTypes=types,
        linkJointAxis=axes,
        physicsClientId=CLIENT,
    )

def create_box(w, l, h, mass=STATIC_MASS, color=RED, **kwargs):
    collision_id, visual_id = create_shape(get_box_geometry(w, l, h), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)
    # basePosition | baseOrientation
    # linkCollisionShapeIndices | linkVisualShapeIndices

def create_cylinder(radius, height, mass=STATIC_MASS, color=BLUE, **kwargs):
    collision_id, visual_id = create_shape(get_cylinder_geometry(radius, height), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)

def create_capsule(radius, height, mass=STATIC_MASS, color=BLUE, **kwargs):
    collision_id, visual_id = create_shape(get_capsule_geometry(radius, height), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)

def create_sphere(radius, mass=STATIC_MASS, color=BLUE, **kwargs):
    collision_id, visual_id = create_shape(get_sphere_geometry(radius), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)

def create_plane(normal=[0, 0, 1], mass=STATIC_MASS, color=BLACK, **kwargs):
    # color seems to be ignored in favor of a texture
    collision_id, visual_id = create_shape(get_plane_geometry(normal), color=color, **kwargs)
    body = create_body(collision_id, visual_id, mass=mass)
    set_texture(body, texture=None) # otherwise 'plane.urdf'
    set_color(body, color=color) # must perform after set_texture
    return body

def create_obj(path, scale=1., mass=STATIC_MASS, color=GREY, **kwargs):
    collision_id, visual_id = create_shape(get_mesh_geometry(path, scale=scale), color=color, **kwargs)
    body = create_body(collision_id, visual_id, mass=mass)
    fixed_base = (mass == STATIC_MASS)
    INFO_FROM_BODY[CLIENT, body] = ModelInfo(None, path, fixed_base, scale) # TODO: store geometry info instead?
    return body

Mesh = namedtuple('Mesh', ['vertices', 'faces'])
mesh_count = count()
TEMP_DIR = 'temp/'

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

# def create_mesh(mesh, under=True, **kwargs):
#     # http://people.sc.fsu.edu/~jburkardt/data/obj/obj.html
#     # TODO: read OFF / WRL / OBJ files
#     # TODO: maintain dict to file
#     ensure_dir(TEMP_DIR)
#     path = os.path.join(TEMP_DIR, 'mesh{}.obj'.format(next(mesh_count)))
#     write(path, obj_file_from_mesh(mesh, under=under))
#     return create_obj(path, **kwargs)
#     #safe_remove(path) # TODO: removing might delete mesh?

def create_faces(mesh, scale=1., mass=STATIC_MASS, collision=True, color=GREY, **kwargs):
    # TODO: rename to create_mesh
    collision_id, visual_id = create_shape(get_faces_geometry(mesh, scale=scale, **kwargs), collision=collision, color=color)
    body = create_body(collision_id, visual_id, mass=mass)
    # fixed_base = (mass == STATIC_MASS)
    # INFO_FROM_BODY[CLIENT, body] = ModelInfo(None, None, fixed_base, scale)
    return body

#####################################

VisualShapeData = namedtuple('VisualShapeData', ['objectUniqueId', 'linkIndex',
                                                 'visualGeometryType', 'dimensions', 'meshAssetFileName',
                                                 'localVisualFrame_position', 'localVisualFrame_orientation',
                                                 'rgbaColor']) # 'textureUniqueId'

UNKNOWN_FILE = 'unknown_file'

def visual_shape_from_data(data, client=None):
    client = get_client(client)
    if (data.visualGeometryType == p.GEOM_MESH) and (data.meshAssetFileName == UNKNOWN_FILE):
        return NULL_ID
    # visualFramePosition: translational offset of the visual shape with respect to the link
    # visualFrameOrientation: rotational offset (quaternion x,y,z,w) of the visual shape with respect to the link frame
    #inertial_pose = get_joint_inertial_pose(data.objectUniqueId, data.linkIndex)
    #point, quat = multiply(invert(inertial_pose), pose)
    point, quat = get_data_pose(data)
    return p.createVisualShape(shapeType=data.visualGeometryType,
                               radius=get_data_radius(data),
                               halfExtents=np.array(get_data_extents(data))/2,
                               length=get_data_height(data), # TODO: pybullet bug
                               fileName=data.meshAssetFileName,
                               meshScale=get_data_scale(data),
                               planeNormal=get_data_normal(data),
                               rgbaColor=data.rgbaColor,
                               #specularColor=,
                               visualFramePosition=point,
                               visualFrameOrientation=quat,
                               physicsClientId=client)

def get_visual_data(body, link=BASE_LINK):
    # TODO: might require the viewer to be active
    visual_data = [VisualShapeData(*tup) for tup in p.getVisualShapeData(body, physicsClientId=CLIENT)]
    return list(filter(lambda d: d.linkIndex == link, visual_data))

# object_unique_id and linkIndex seem to be noise
CollisionShapeData = namedtuple('CollisionShapeData', ['object_unique_id', 'linkIndex',
                                                       'geometry_type', 'dimensions', 'filename',
                                                       'local_frame_pos', 'local_frame_orn'])

def collision_shape_from_data(data, body, link, client=None):
    client = get_client(client)
    if (data.geometry_type == p.GEOM_MESH) and (data.filename == UNKNOWN_FILE):
        return NULL_ID
    pose = multiply(get_joint_inertial_pose(body, link), get_data_pose(data))
    point, quat = pose
    # TODO: the visual data seems affected by the collision data
    return p.createCollisionShape(shapeType=data.geometry_type,
                                  radius=get_data_radius(data),
                                  # halfExtents=get_data_extents(data.geometry_type, data.dimensions),
                                  halfExtents=np.array(get_data_extents(data)) / 2,
                                  height=get_data_height(data),
                                  fileName=data.filename.decode(encoding='UTF-8'),
                                  meshScale=get_data_scale(data),
                                  planeNormal=get_data_normal(data),
                                  flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                                  collisionFramePosition=point,
                                  collisionFrameOrientation=quat,
                                  physicsClientId=client)
    #return p.createCollisionShapeArray()

def clone_visual_shape(body, link, client=None):
    client = get_client(client)
    #if not has_gui(client):
    #    return NULL_ID
    visual_data = get_visual_data(body, link)
    if not visual_data:
        return NULL_ID
    assert (len(visual_data) == 1)
    return visual_shape_from_data(visual_data[0], client)

def clone_collision_shape(body, link, client=None):
    client = get_client(client)
    collision_data = get_collision_data(body, link)
    if not collision_data:
        return NULL_ID
    assert (len(collision_data) == 1)
    # TODO: can do CollisionArray
    return collision_shape_from_data(collision_data[0], body, link, client)

def clone_body(body, links=None, collision=True, visual=True, client=None):
    # TODO: names are not retained
    # TODO: error with createMultiBody link poses on PR2
    # localVisualFrame_position: position of local visual frame, relative to link/joint frame
    # localVisualFrame orientation: orientation of local visual frame relative to link/joint frame
    # parentFramePos: joint position in parent frame
    # parentFrameOrn: joint orientation in parent frame
    client = get_client(client) # client is the new client for the body
    if links is None:
        links = get_links(body)
    #movable_joints = [joint for joint in links if is_movable(body, joint)]
    new_from_original = {}
    base_link = get_link_parent(body, links[0]) if links else BASE_LINK
    new_from_original[base_link] = NULL_ID

    masses = []
    collision_shapes = []
    visual_shapes = []
    positions = [] # list of local link positions, with respect to parent
    orientations = [] # list of local link orientations, w.r.t. parent
    inertial_positions = [] # list of local inertial frame pos. in link frame
    inertial_orientations = [] # list of local inertial frame orn. in link frame
    parent_indices = []
    joint_types = []
    joint_axes = []
    for i, link in enumerate(links):
        new_from_original[link] = i
        joint_info = get_joint_info(body, link)
        dynamics_info = get_dynamics_info(body, link)
        masses.append(dynamics_info.mass)
        collision_shapes.append(clone_collision_shape(body, link, client) if collision else NULL_ID)
        visual_shapes.append(clone_visual_shape(body, link, client) if visual else NULL_ID)
        point, quat = get_local_link_pose(body, link)
        positions.append(point)
        orientations.append(quat)
        inertial_positions.append(dynamics_info.local_inertial_pos)
        inertial_orientations.append(dynamics_info.local_inertial_orn)
        parent_indices.append(new_from_original[joint_info.parentIndex] + 1) # TODO: need the increment to work
        joint_types.append(joint_info.jointType)
        joint_axes.append(joint_info.jointAxis)
    # https://github.com/bulletphysics/bullet3/blob/9c9ac6cba8118544808889664326fd6f06d9eeba/examples/pybullet/gym/pybullet_utils/urdfEditor.py#L169

    base_dynamics_info = get_dynamics_info(body, base_link)
    base_point, base_quat = get_link_pose(body, base_link)
    new_body = p.createMultiBody(baseMass=base_dynamics_info.mass,
                                 baseCollisionShapeIndex=clone_collision_shape(body, base_link, client) if collision else NULL_ID,
                                 baseVisualShapeIndex=clone_visual_shape(body, base_link, client) if visual else NULL_ID,
                                 basePosition=base_point,
                                 baseOrientation=base_quat,
                                 baseInertialFramePosition=base_dynamics_info.local_inertial_pos,
                                 baseInertialFrameOrientation=base_dynamics_info.local_inertial_orn,
                                 linkMasses=masses,
                                 linkCollisionShapeIndices=collision_shapes,
                                 linkVisualShapeIndices=visual_shapes,
                                 linkPositions=positions,
                                 linkOrientations=orientations,
                                 linkInertialFramePositions=inertial_positions,
                                 linkInertialFrameOrientations=inertial_orientations,
                                 linkParentIndices=parent_indices,
                                 linkJointTypes=joint_types,
                                 linkJointAxis=joint_axes,
                                 physicsClientId=client)
    #set_configuration(new_body, get_joint_positions(body, movable_joints)) # Need to use correct client
    for joint, value in zip(range(len(links)), get_joint_positions(body, links)):
        # TODO: check if movable?
        p.resetJointState(new_body, joint, value, targetVelocity=0, physicsClientId=client)
    return new_body

def clone_world(client=None, exclude=[]):
    visual = has_gui(client)
    mapping = {}
    for body in get_bodies():
        if body not in exclude:
            new_body = clone_body(body, collision=True, visual=visual, client=client)
            mapping[body] = new_body
    return mapping

#####################################

def get_mesh_data(obj, link=BASE_LINK, shape_index=0, visual=True):
    flags = 0 if visual else p.MESH_DATA_SIMULATION_MESH
    #collisionShapeIndex = shape_index
    return Mesh(*p.getMeshData(obj, linkIndex=link, flags=flags, physicsClientId=CLIENT))

def get_collision_data(body, link=BASE_LINK):
    # TODO: try catch
    return [CollisionShapeData(*tup) for tup in p.getCollisionShapeData(body, link, physicsClientId=CLIENT)]

def get_data_type(data):
    return data.geometry_type if isinstance(data, CollisionShapeData) else data.visualGeometryType

def get_data_filename(data):
    return (data.filename if isinstance(data, CollisionShapeData)
            else data.meshAssetFileName).decode(encoding='UTF-8')

def get_data_pose(data):
    if isinstance(data, CollisionShapeData):
        return (data.local_frame_pos, data.local_frame_orn)
    return (data.localVisualFrame_position, data.localVisualFrame_orientation)

def get_default_geometry():
    return {
        'halfExtents': DEFAULT_EXTENTS,
        'radius': DEFAULT_RADIUS,
        'length': DEFAULT_HEIGHT, # 'height'
        'fileName': DEFAULT_MESH,
        'meshScale': DEFAULT_SCALE,
        'planeNormal': DEFAULT_NORMAL,
    }

DEFAULT_MESH = ''

DEFAULT_EXTENTS = [1, 1, 1]

def get_data_extents(data):
    """
    depends on geometry type:
    for GEOM_BOX: extents,
    for GEOM_SPHERE dimensions[0] = radius,
    for GEOM_CAPSULE and GEOM_CYLINDER, dimensions[0] = height (length), dimensions[1] = radius.
    For GEOM_MESH, dimensions is the scaling factor.
    :return:
    """
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_BOX:
        return dimensions
    return DEFAULT_EXTENTS

DEFAULT_RADIUS = 0.5

def get_data_radius(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_SPHERE:
        return dimensions[0]
    if geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        return dimensions[1]
    return DEFAULT_RADIUS

DEFAULT_HEIGHT = 1

def get_data_height(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        return dimensions[0]
    return DEFAULT_HEIGHT

DEFAULT_SCALE = [1, 1, 1]

def get_data_scale(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_MESH:
        return dimensions
    return DEFAULT_SCALE

DEFAULT_NORMAL = [0, 0, 1]

def get_data_normal(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_PLANE:
        return dimensions
    return DEFAULT_NORMAL

def get_data_geometry(data):
    geometry_type = get_data_type(data)
    if geometry_type == p.GEOM_SPHERE:
        parameters = [get_data_radius(data)]
    elif geometry_type == p.GEOM_BOX:
        parameters = [get_data_extents(data)]
    elif geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        parameters = [get_data_height(data), get_data_radius(data)]
    elif geometry_type == p.GEOM_MESH:
        parameters = [get_data_filename(data), get_data_scale(data)]
    elif geometry_type == p.GEOM_PLANE:
        parameters = [get_data_extents(data)]
    else:
        raise ValueError(geometry_type)
    return SHAPE_TYPES[geometry_type], parameters

def get_color(body, **kwargs):
    # TODO: average over texture
    visual_data = get_visual_data(body, **kwargs)
    if not visual_data:
        # TODO: no viewer implies no visual data
        return None
    return visual_data[0].rgbaColor

def set_color(body, color, link=BASE_LINK, shape_index=NULL_ID):
    """
    Experimental for internal use, recommended ignore shapeIndex or leave it -1.
    Intention was to let you pick a specific shape index to modify,
    since URDF (and SDF etc) can have more than 1 visual shape per link.
    This shapeIndex matches the list ordering returned by getVisualShapeData.
    :param body:
    :param color: RGBA
    :param link:
    :param shape_index:
    :return:
    """
    # specularColor
    if link is None:
        return set_all_color(body, color)
    return p.changeVisualShape(body, link, shapeIndex=shape_index, rgbaColor=color,
                               #textureUniqueId=None, specularColor=None,
                               physicsClientId=CLIENT)

def set_all_color(body, color):
    for link in get_all_links(body):
        set_color(body, color, link)

def set_texture(body, texture=None, link=BASE_LINK, shape_index=NULL_ID):
    if texture is None:
        texture = NULL_ID
    return p.changeVisualShape(body, link, shapeIndex=shape_index, textureUniqueId=texture,
                               physicsClientId=CLIENT)

#####################################

# Bounding box

DEFAULT_AABB_BUFFER = 0.02

AABB = namedtuple('AABB', ['lower', 'upper'])

def aabb_from_points(points):
    return AABB(np.min(points, axis=0), np.max(points, axis=0))

def aabb_union(aabbs):
    if not aabbs:
        return None
    return aabb_from_points(np.vstack([aabb for aabb in aabbs]))

def aabb_overlap(aabb1, aabb2):
    if (aabb1 is None) or (aabb2 is None):
        return False
    lower1, upper1 = aabb1
    lower2, upper2 = aabb2
    return np.less_equal(lower1, upper2).all() and \
           np.less_equal(lower2, upper1).all()

def aabb_empty(aabb):
    lower, upper = aabb
    return np.less(upper, lower).any()

def is_aabb_degenerate(aabb):
    return get_aabb_volume(aabb) <= 0.

def aabb_intersection(*aabbs):
    # https://github.mit.edu/caelan/lis-openrave/blob/master/manipulation/bodies/bounding_volumes.py
    lower = np.max([lower for lower, _ in aabbs], axis=0)
    upper = np.min([upper for _, upper in aabbs], axis=0)
    aabb = AABB(lower, upper)
    if aabb_empty(aabb):
        return None
    return aabb

def get_subtree_aabb(body, root_link=BASE_LINK):
    return aabb_union(get_aabb(body, link) for link in get_link_subtree(body, root_link))

def get_aabbs(body, links=None, only_collision=True):
    if links is None:
        links = get_all_links(body)
    if only_collision:
        # TODO: return the null bounding box
        links = [link for link in links if get_collision_data(body, link)]
    return [get_aabb(body, link=link) for link in links]

def get_aabb(body, link=None, **kwargs):
    # Note that the query is conservative and may return additional objects that don't have actual AABB overlap.
    # This happens because the acceleration structures have some heuristic that enlarges the AABBs a bit
    # (extra margin and extruded along the velocity vector).
    # Contact points with distance exceeding this threshold are not processed by the LCP solver.
    # AABBs are extended by this number. Defaults to 0.02 in Bullet 2.x
    #p.setPhysicsEngineParameter(contactBreakingThreshold=0.0, physicsClientId=CLIENT)
    # Computes the AABB of the collision geometry
    if link is None:
        return aabb_union(get_aabbs(body, **kwargs))
    return AABB(*p.getAABB(body, linkIndex=link, physicsClientId=CLIENT))

get_lower_upper = get_aabb

def get_aabb_center(aabb):
    lower, upper = aabb
    return (np.array(lower) + np.array(upper)) / 2.

def get_aabb_extent(aabb):
    lower, upper = aabb
    return np.array(upper) - np.array(lower)

def get_center_extent(body, **kwargs):
    aabb = get_aabb(body, **kwargs)
    return get_aabb_center(aabb), get_aabb_extent(aabb)

def aabb2d_from_aabb(aabb):
    (lower, upper) = aabb
    return AABB(lower[:2], upper[:2])

def aabb_contains_aabb(contained, container):
    lower1, upper1 = contained
    lower2, upper2 = container
    return np.less_equal(lower2, lower1).all() and \
           np.less_equal(upper1, upper2).all()
    #return np.all(lower2 <= lower1) and np.all(upper1 <= upper2)

def aabb_contains_point(point, container):
    lower, upper = container
    return np.less_equal(lower, point).all() and \
           np.less_equal(point, upper).all()
    #return np.all(lower <= point) and np.all(point <= upper)

def sample_aabb(aabb):
    lower, upper = aabb
    return np.random.uniform(lower, upper)

def get_bodies_in_region(aabb):
    (lower, upper) = aabb
    #step_simulation() # Like visibility, need to step first
    bodies = p.getOverlappingObjects(lower, upper, physicsClientId=CLIENT)
    return [] if bodies is None else sorted(bodies)

def get_aabb_volume(aabb):
    if aabb_empty(aabb):
        return 0.
    return np.prod(get_aabb_extent(aabb))

def get_aabb_area(aabb):
    return get_aabb_volume(aabb2d_from_aabb(aabb))

def get_aabb_vertices(aabb):
    d = len(aabb[0])
    return [tuple(aabb[i[k]][k] for k in range(d))
            for i in product(range(len(aabb)), repeat=d)]

def get_aabb_edges(aabb):
    d = len(aabb[0])
    vertices = list(product(range(len(aabb)), repeat=d))
    lines = []
    for i1, i2 in combinations(vertices, 2):
        if sum(i1[k] != i2[k] for k in range(d)) == 1:
            p1 = [aabb[i1[k]][k] for k in range(d)]
            p2 = [aabb[i2[k]][k] for k in range(d)]
            lines.append((p1, p2))
    return lines

def aabb_from_extent_center(extent, center=None):
    if center is None:
        center = np.zeros(len(extent))
    else:
        center = np.array(center)
    half_extent = np.array(extent) / 2.
    lower = center - half_extent
    upper = center + half_extent
    return AABB(lower, upper)

def scale_aabb(aabb, scale):
    center = get_aabb_center(aabb)
    extent = get_aabb_extent(aabb)
    if np.isscalar(scale):
        scale = scale * np.ones(len(extent))
    new_extent = np.multiply(scale, extent)
    return aabb_from_extent_center(new_extent, center)

def buffer_aabb(aabb, buffer):
    if aabb is None:
        return aabb
    extent = get_aabb_extent(aabb)
    if np.isscalar(buffer):
        if buffer == 0.:
            return aabb
        buffer = buffer * np.ones(len(extent))
    new_extent = np.add(2*buffer, extent)
    center = get_aabb_center(aabb)
    return aabb_from_extent_center(new_extent, center)

def tform_point(affine, point):
    return point_from_pose(multiply(affine, Pose(point=point)))

def tform_points(affine, points):
    return [tform_point(affine, p) for p in points]

apply_affine = tform_points
#####################################

OOBB = namedtuple('OOBB', ['aabb', 'pose'])

def oobb_from_points(points): # Not necessarily minimal volume
    points = np.array(points).T
    d = points.shape[0]
    mu = np.resize(np.mean(points, axis=1), (d, 1))
    centered = points - mu
    u, _, _ = np.linalg.svd(centered)
    if np.linalg.det(u) < 0:
        u[:, 1] *= -1
    # TODO: rotate such that z is up

    aabb = aabb_from_points(np.dot(u.T, centered).T)
    tform = np.identity(4)
    tform[:d, :d] = u
    tform[:d, 3] = mu.T
    return OOBB(aabb, pose_from_tform(tform))

def oobb_contains_point(point, container):
    aabb, pose = container
    return aabb_contains_point(tform_point(invert(pose), point), aabb)

def tform_oobb(affine, oobb):
    aabb, pose = oobb
    return OOBB(aabb, multiply(affine, pose))

def aabb_from_oobb(oobb):
    aabb, pose = oobb
    return aabb_from_points(tform_points(pose, get_aabb_vertices(aabb)))

#####################################
def read_obj(path, decompose=True):
    mesh = Mesh([], [])
    meshes = {}
    vertices = []
    faces = []
    for line in read(path).split('\n'):
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == 'o':
            name = tokens[1]
            mesh = Mesh([], [])
            meshes[name] = mesh
        elif tokens[0] == 'v':
            vertex = tuple(map(float, tokens[1:4]))
            vertices.append(vertex)
        elif tokens[0] in ('vn', 's'):
            pass
        elif tokens[0] == 'f':
            face = tuple(int(token.split('/')[0]) - 1 for token in tokens[1:])
            faces.append(face)
            mesh.faces.append(face)
    if not decompose:
        return Mesh(vertices, faces)

    # TODO: separate into a standalone method
    #if not meshes:
    #    # TODO: ensure this still works if no objects
    #    meshes[None] = mesh
    #new_meshes = {}
    # TODO: make each triangle a separate object
    for name, mesh in meshes.items():
        indices = sorted({i for face in mesh.faces for i in face})
        mesh.vertices[:] = [vertices[i] for i in indices]
        new_index_from_old = {i2: i1 for i1, i2 in enumerate(indices)}
        mesh.faces[:] = [tuple(new_index_from_old[i1] for i1 in face) for face in mesh.faces]
        #edges = {edge for face in mesh.faces for edge in get_face_edges(face)}
        #for k, cluster in enumerate(get_connected_components(indices, edges)):
        #    new_name = '{}#{}'.format(name, k)
        #    new_indices = sorted(cluster)
        #    new_vertices = [vertices[i] for i in new_indices]
        #    new_index_from_old = {i2: i1 for i1, i2 in enumerate(new_indices)}
        #    new_faces = [tuple(new_index_from_old[i1] for i1 in face)
        #                 for face in mesh.faces if set(face) <= cluster]
        #    new_meshes[new_name] = Mesh(new_vertices, new_faces)
    return meshes
# AABB approximation

def vertices_from_data(data):
    geometry_type = get_data_type(data)
    #if geometry_type == p.GEOM_SPHERE:
    #    parameters = [get_data_radius(data)]
    if geometry_type == p.GEOM_BOX:
        extents = np.array(get_data_extents(data))
        aabb = aabb_from_extent_center(extents)
        vertices = get_aabb_vertices(aabb)
    elif geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        # TODO: p.URDF_USE_IMPLICIT_CYLINDER
        radius, height = get_data_radius(data), get_data_height(data)
        extents = np.array([2*radius, 2*radius, height])
        aabb = aabb_from_extent_center(extents)
        vertices = get_aabb_vertices(aabb)
    elif geometry_type == p.GEOM_SPHERE:
        radius = get_data_radius(data)
        extents = 2*radius*np.ones(3)
        aabb = aabb_from_extent_center(extents)
        vertices = get_aabb_vertices(aabb)
    elif geometry_type == p.GEOM_MESH:
        filename, scale = get_data_filename(data), get_data_scale(data)
        if filename == UNKNOWN_FILE:
            raise RuntimeError(filename)
        # _, ext = os.path.splitext(filename)
        # if ext != '.obj':
        #     raise RuntimeError(filename)
        mesh = read_obj(filename, decompose=False)
        vertices = [scale*np.array(vertex) for vertex in mesh.vertices]
        # TODO: could compute AABB here for improved speed at the cost of being conservative
    #elif geometry_type == p.GEOM_PLANE:
    #   parameters = [get_data_extents(data)]
    else:
        raise NotImplementedError(geometry_type)
    return vertices

def oobb_from_data(data):
    link_from_data = get_data_pose(data)
    vertices_data = vertices_from_data(data)
    return OOBB(aabb_from_points(vertices_data), link_from_data)

def vertices_from_link(body, link=BASE_LINK, collision=True):
    # TODO: get_mesh_data(body, link=link)
    # In local frame
    vertices = []
    # PyBullet creates multiple collision elements (with unknown_file) when nonconvex
    get_data = get_collision_data if collision else get_visual_data
    for data in get_data(body, link):
        # TODO: get_visual_data usually has a valid mesh file unlike get_collision_data
        # TODO: apply the inertial frame
        vertices.extend(apply_affine(get_data_pose(data), vertices_from_data(data)))
    return vertices

OBJ_MESH_CACHE = {}
URDF_MESH_CACHE = {}
def implies(p1, p2):
    return not p1 or p2
def vertices_from_rigid(body, link=BASE_LINK):
    assert implies(link == BASE_LINK, get_num_links(body) == 0)
    try:
        vertices = vertices_from_link(body, link)
    except RuntimeError:
        info = get_model_info(body)
        assert info is not None
        _, ext = os.path.splitext(info.path)
        if ext == '.obj':
            if info.path not in OBJ_MESH_CACHE:
                OBJ_MESH_CACHE[info.path] = read_obj(info.path, decompose=False)
            mesh = OBJ_MESH_CACHE[info.path]
            vertices = mesh.vertices
        elif ext==".urdf":
            print(URDF_MESH_CACHE)
            if info.path not in URDF_MESH_CACHE:
                URDF_MESH_CACHE[info.path] = p.getMeshData(body)
            mesh = URDF_MESH_CACHE[info.path]
            print(len(mesh))
            vertices = mesh[1]
        else:
            raise NotImplementedError(ext)
    return vertices

def approximate_as_prism(body, body_pose=unit_pose(), **kwargs):
    # TODO: make it just orientation
    vertices = apply_affine(body_pose, vertices_from_rigid(body, **kwargs))
    aabb = aabb_from_points(vertices)
    return get_aabb_center(aabb), get_aabb_extent(aabb)
    #with PoseSaver(body):
    #    set_pose(body, body_pose)
    #    set_velocity(body, linear=np.zeros(3), angular=np.zeros(3))
    #    return get_center_extent(body, **kwargs)

def approximate_as_cylinder(body, **kwargs):
    center, (width, length, height) = approximate_as_prism(body, **kwargs)
    diameter = (width + length) / 2  # TODO: check that these are close
    return center, (diameter, height)

#####################################

# Collision

MAX_DISTANCE = 0.04 # 0. | 1e-3

CollisionPair = namedtuple('Collision', ['body', 'links'])

def get_buffered_aabb(body, max_distance=MAX_DISTANCE, **kwargs):
    body, links = parse_body(body, **kwargs)
    return buffer_aabb(aabb_union(get_aabbs(body, links=links)), buffer=max_distance)

def get_unbuffered_aabb(body, **kwargs):
    return get_buffered_aabb(body, max_distance=-DEFAULT_AABB_BUFFER/2., **kwargs)

def contact_collision():
    step_simulation()
    return len(p.getContactPoints(physicsClientId=CLIENT)) != 0

ContactResult = namedtuple('ContactResult', ['contactFlag', 'bodyUniqueIdA', 'bodyUniqueIdB',
                                             'linkIndexA', 'linkIndexB', 'positionOnA', 'positionOnB',
                                             'contactNormalOnB', 'contactDistance', 'normalForce'])

def flatten_links(body, links=None):
    if links is None:
        links = get_all_links(body)
    return {CollisionPair(body, frozenset([link])) for link in links}

def parse_body(body, link=None):
    return body if isinstance(body, tuple) else CollisionPair(body, link)

def expand_links(body, **kwargs):
    body, links = parse_body(body, **kwargs)
    if links is None:
        links = get_all_links(body)
    return CollisionPair(body, links)

CollisionInfo = namedtuple('CollisionInfo',
                           '''
                           contactFlag
                           bodyUniqueIdA
                           bodyUniqueIdB
                           linkIndexA
                           linkIndexB
                           positionOnA
                           positionOnB
                           contactNormalOnB
                           contactDistance
                           normalForce
                           lateralFriction1
                           lateralFrictionDir1
                           lateralFriction2
                           lateralFrictionDir2
                           '''.split())


def get_closest_points(body1, body2, link1=None, link2=None, max_distance=-MAX_DISTANCE, use_aabb=False):
    if use_aabb and not aabb_overlap(get_buffered_aabb(body1, link1, max_distance=max_distance/2.),
                                     get_buffered_aabb(body2, link2, max_distance=max_distance/2.)):
        return []

    if (link1 is None) and (link2 is None):
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance, physicsClientId=CLIENT)
    elif link2 is None:
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, linkIndexA=link1,
                                     distance=max_distance, physicsClientId=CLIENT)
    elif link1 is None:
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, linkIndexB=link2,
                                     distance=max_distance, physicsClientId=CLIENT)
    else:
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, linkIndexA=link1, linkIndexB=link2,
                                     distance=max_distance, physicsClientId=CLIENT)
    return [CollisionInfo(*info) for info in results]

def pairwise_link_collision(body1, link1, body2, link2=BASE_LINK, **kwargs):
    return len(get_closest_points(body1, body2, link1=link1, link2=link2, **kwargs)) != 0

def any_link_pair_collision(body1, links1, body2, links2=None, **kwargs):
    # TODO: this likely isn't needed anymore
    if links1 is None:
        links1 = get_all_links(body1)
    if links2 is None:
        links2 = get_all_links(body2)
    for link1, link2 in product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue
        if pairwise_link_collision(body1, link1, body2, link2, **kwargs):
            return True
    return False

link_pairs_collision = any_link_pair_collision

def body_collision(body1, body2, **kwargs):
    return len(get_closest_points(body1, body2, **kwargs)) != 0

def pairwise_collision(body1, body2, **kwargs):
    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = expand_links(body1)
        body2, links2 = expand_links(body2)
        return any_link_pair_collision(body1, links1, body2, links2, **kwargs)
    val = body_collision(body1, body2, **kwargs)
    if val:
        print('body collision', body1, body2)
    return val

def pairwise_collisions(body, obstacles, link=None, **kwargs):
    return any(pairwise_collision(body1=body, body2=other, link1=link, **kwargs)
               for other in obstacles if body != other)

#def single_collision(body, max_distance=1e-3):
#    return len(p.getClosestPoints(body, max_distance=max_distance)) != 0

def single_collision(body, **kwargs):
    return pairwise_collisions(body, get_bodies(), **kwargs)

#####################################

Ray = namedtuple('Ray', ['start', 'end'])

def get_ray(ray):
    start, end = ray
    return np.array(end) - np.array(start)

RayResult = namedtuple('RayResult', ['objectUniqueId', 'linkIndex',
                                     'hit_fraction', 'hit_position', 'hit_normal']) # TODO: store Ray here

def ray_collision(ray):
    # TODO: be careful to disable gravity and set static masses for everything
    step_simulation() # Needed for some reason
    start, end = ray
    result, = p.rayTest(start, end, physicsClientId=CLIENT)
    # TODO: assign hit_position to be the end?
    return RayResult(*result)

def batch_ray_collision(rays, threads=1):
    assert 1 <= threads <= p.MAX_RAY_INTERSECTION_BATCH_SIZE
    if not rays:
        return []
    step_simulation() # Needed for some reason
    ray_starts = [start for start, _ in rays]
    ray_ends = [end for _, end in rays]
    return [RayResult(*tup) for tup in p.rayTestBatch(
        ray_starts, ray_ends,
        numThreads=threads,
        #parentObjectUniqueId=
        #parentLinkIndex=
        physicsClientId=CLIENT)]

def get_ray_from_to(mouseX, mouseY, farPlane=10000):
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/examples/pointCloudFromCameraImage.py
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/examples/addPlanarReflection.py
    width, height, _, _, _, camForward, horizon, vertical, _, _, dist, camTarget = p.getDebugVisualizerCamera()
    rayFrom = camPos = np.array(camTarget) - dist * np.array(camForward)
    rayForward = farPlane*get_unit_vector(np.array(camTarget) - rayFrom)
    dHor = np.array(horizon) / float(width)
    dVer = np.array(vertical) / float(height)
    #rayToCenter = rayFrom + rayForward
    rayTo = rayFrom + rayForward - 0.5*(np.array(horizon) - np.array(vertical)) + (mouseX*dHor - mouseY*dVer)
    return Ray(rayFrom, rayTo)

#####################################

# Joint motion planning

def uniform_generator(d):
    while True:
        yield np.random.uniform(size=d)

def sample_norm(mu, sigma, lower=0., upper=INF):
    # scipy.stats.truncnorm
    assert lower <= upper
    if lower == upper:
        return lower
    if sigma == 0.:
        assert lower <= mu <= upper
        return mu
    while True:
        x = random.gauss(mu=mu, sigma=sigma)
        if lower <= x <= upper:
            return x

def halton_generator(d, seed=None):
    import ghalton
    if seed is None:
        seed = random.randint(0, 1000)
    #sequencer = ghalton.Halton(d)
    sequencer = ghalton.GeneralizedHalton(d, seed)
    #sequencer.reset()
    while True:
        [weights] = sequencer.get(1)
        yield np.array(weights)

def unit_generator(d, use_halton=False):
    if use_halton:
        try:
            import ghalton
        except ImportError:
            print('ghalton is not installed (https://pypi.org/project/ghalton/)')
            use_halton = False
    return halton_generator(d) if use_halton else uniform_generator(d)

def interval_generator(lower, upper, **kwargs):
    assert len(lower) == len(upper)
    assert np.less_equal(lower, upper).all()
    if np.equal(lower, upper).all():
        return iter([lower])
    return (convex_combination(lower, upper, w=weights) for weights in unit_generator(d=len(lower), **kwargs))

def get_sample_fn(body, joints, custom_limits={}, **kwargs):
    lower_limits, upper_limits = get_custom_limits(body, joints, custom_limits, circular_limits=CIRCULAR_LIMITS)
    generator = interval_generator(lower_limits, upper_limits, **kwargs)
    def fn():
        return tuple(next(generator))
    return fn

def get_halton_sample_fn(body, joints, **kwargs):
    return get_sample_fn(body, joints, use_halton=True, **kwargs)

def get_difference_fn(body, joints):
    circular_joints = [is_circular(body, joint) for joint in joints]

    def fn(q2, q1):
        return tuple(circular_difference(value2, value1) if circular else (value2 - value1)
                     for circular, value2, value1 in zip(circular_joints, q2, q1))
    return fn

def get_default_weights(body, joints, weights=None):
    if weights is not None:
        return weights
    # TODO: derive from resolutions
    # TODO: use the energy resulting from the mass matrix here?
    return 1*np.ones(len(joints)) # TODO: use velocities here

def get_distance_fn(body, joints, weights=None): #, norm=2):
    weights = get_default_weights(body, joints, weights)
    difference_fn = get_difference_fn(body, joints)
    def fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, diff * diff))
        #return np.linalg.norm(np.multiply(weights * diff), ord=norm)
    return fn

def get_duration_fn(body, joints, velocities=None, norm=INF):
    # TODO: integrate with get_distance_fn weights
    # TODO: integrate with get_nonholonomic_distance_fn
    if velocities is None:
        velocities = np.array(get_max_velocities(body, joints))
    difference_fn = get_difference_fn(body, joints)
    def fn(q1, q2):
        distances = np.array(difference_fn(q2, q1))
        durations = np.divide(distances, np.abs(velocities))
        return np.linalg.norm(durations, ord=norm)
    return fn

def get_refine_fn(body, joints, num_steps=0):
    difference_fn = get_difference_fn(body, joints)
    num_steps = num_steps + 1
    def fn(q1, q2):
        q = q1
        for i in range(num_steps):
            positions = (1. / (num_steps - i)) * np.array(difference_fn(q2, q)) + q
            q = tuple(positions)
            #q = tuple(wrap_positions(body, joints, positions))
            yield q
    return fn

def get_pairs(sequence):
    # TODO: lazy version
    sequence = list(sequence)
    return safe_zip(sequence[:-1], sequence[1:])

def get_wrapped_pairs(sequence):
    # TODO: lazy version
    sequence = list(sequence)
    # zip(sequence, sequence[-1:] + sequence[:-1])
    return safe_zip(sequence, sequence[1:] + sequence[:1])

def refine_path(body, joints, waypoints, num_steps):
    refine_fn = get_refine_fn(body, joints, num_steps)
    refined_path = []
    for v1, v2 in get_pairs(waypoints):
        refined_path.extend(refine_fn(v1, v2))
    return refined_path

DEFAULT_RESOLUTION = math.radians(3) # 0.05

def get_default_resolutions(body, joints, resolutions=None):
    if resolutions is not None:
        return resolutions
    return DEFAULT_RESOLUTION*np.ones(len(joints))

def get_extend_fn(body, joints, resolutions=None, norm=2):
    # norm = 1, 2, INF
    resolutions = get_default_resolutions(body, joints, resolutions)
    difference_fn = get_difference_fn(body, joints)
    def fn(q1, q2):
        #steps = int(np.max(np.abs(np.divide(difference_fn(q2, q1), resolutions))))
        steps = int(np.linalg.norm(np.divide(difference_fn(q2, q1), resolutions), ord=norm))
        refine_fn = get_refine_fn(body, joints, num_steps=steps)
        return refine_fn(q1, q2)
    return fn

def remove_redundant(path, tolerance=1e-3):
    assert path
    new_path = [path[0]]
    for conf in path[1:]:
        difference = np.array(new_path[-1]) - np.array(conf)
        if not np.allclose(np.zeros(len(difference)), difference, atol=tolerance, rtol=0):
            new_path.append(conf)
    return new_path

def waypoints_from_path(path, tolerance=1e-3):
    path = remove_redundant(path, tolerance=tolerance)
    if len(path) < 2:
        return path
    difference_fn = lambda q2, q1: np.array(q2) - np.array(q1)
    #difference_fn = get_difference_fn(body, joints) # TODO: account for wrap around or use adjust_path

    waypoints = [path[0]]
    last_conf = path[1]
    last_difference = get_unit_vector(difference_fn(last_conf, waypoints[-1]))
    for conf in path[2:]:
        difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        if not np.allclose(last_difference, difference, atol=tolerance, rtol=0):
            waypoints.append(last_conf)
            difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        last_conf = conf
        last_difference = difference
    waypoints.append(last_conf)
    return waypoints

def adjust_path(robot, joints, path):
    difference_fn = get_difference_fn(robot, joints)
    differences = [difference_fn(q2, q1) for q1, q2 in get_pairs(path)]
    adjusted_path = [np.array(get_joint_positions(robot, joints))] # Assumed the same as path[0] mod rotation
    for difference in differences:
        if not np.array_equal(difference, np.zeros(len(joints))):
            adjusted_path.append(adjusted_path[-1] + difference)
    return adjusted_path

def get_moving_links(body, joints):
    moving_links = set()
    for joint in joints:
        link = child_link_from_joint(joint)
        if link not in moving_links:
            moving_links.update(get_link_subtree(body, link))
    return list(moving_links)

def get_moving_pairs(body, moving_joints):
    """
    Check all fixed and moving pairs
    Do not check all fixed and fixed pairs
    Check all moving pairs with a common
    """
    moving_links = get_moving_links(body, moving_joints)
    for link1, link2 in combinations(moving_links, 2):
        ancestors1 = set(get_joint_ancestors(body, link1)) & set(moving_joints)
        ancestors2 = set(get_joint_ancestors(body, link2)) & set(moving_joints)
        if ancestors1 != ancestors2:
            yield link1, link2

def get_self_link_pairs(body, joints, disabled_collisions=set(), only_moving=True):
    moving_links = get_moving_links(body, joints)
    fixed_links = list(set(get_links(body)) - set(moving_links))
    check_link_pairs = list(product(moving_links, fixed_links))
    if only_moving:
        check_link_pairs.extend(get_moving_pairs(body, joints))
    else:
        check_link_pairs.extend(combinations(moving_links, 2))
    check_link_pairs = list(filter(lambda pair: not are_links_adjacent(body, *pair), check_link_pairs))
    check_link_pairs = list(filter(lambda pair: (pair not in disabled_collisions) and
                                                (pair[::-1] not in disabled_collisions), check_link_pairs))
    return check_link_pairs

def can_collide(body, link=BASE_LINK, **kwargs):
    return len(get_collision_data(body, link=link, **kwargs)) != 0

def get_limits_fn(body, joints, custom_limits={}, verbose=False):
    lower_limits, upper_limits = get_custom_limits(body, joints, custom_limits)

    def limits_fn(q):
        if not all_between(lower_limits, q, upper_limits):
            #print('Joint limits violated')
            #if verbose: print(lower_limits, q, upper_limits)
            return True
        return False
    return limits_fn

def get_collision_fn(body, joints, obstacles=[], attachments=[], self_collisions=True, disabled_collisions=set(),
                     custom_limits={}, use_aabb=False, cache=False, max_distance=MAX_DISTANCE, **kwargs):
    # TODO: convert most of these to keyword arguments
    check_link_pairs = get_self_link_pairs(body, joints, disabled_collisions) if self_collisions else []
    moving_links = frozenset(link for link in get_moving_links(body, joints)
                             if can_collide(body, link)) # TODO: propagate elsewhere
    attached_bodies = [attachment.child for attachment in attachments]
    moving_bodies = [CollisionPair(body, moving_links)] + list(map(parse_body, attached_bodies))
    #moving_bodies = list(flatten(flatten_links(*pair) for pair in moving_bodies)) # Introduces overhead
    #moving_bodies = [body] + [attachment.child for attachment in attachments]
    get_obstacle_aabb = cached_fn(get_buffered_aabb, cache=cache, max_distance=max_distance/2., **kwargs)
    limits_fn = get_limits_fn(body, joints, custom_limits=custom_limits)
    # TODO: sort bodies by bounding box size
    # TODO: cluster together links that remain rigidly attached to reduce the number of checks

    def collision_fn(q, verbose=False):
        if limits_fn(q):
            return True
        set_joint_positions(body, joints, q)
        for attachment in attachments:
            attachment.assign()
        #wait_for_duration(1e-2)
        get_moving_aabb = cached_fn(get_buffered_aabb, cache=True, max_distance=max_distance/2., **kwargs)

        for link1, link2 in check_link_pairs:
            # Self-collisions should not have the max_distance parameter
            # TODO: self-collisions between body and attached_bodies (except for the link adjacent to the robot)
            if (not use_aabb or aabb_overlap(get_moving_aabb(body), get_moving_aabb(body))) and \
                    pairwise_link_collision(body, link1, body, link2): #, **kwargs):
                #print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                if verbose: print(body, link1, body, link2)
                return True

        # #step_simulation()
        # #update_scene()
        # for body1 in moving_bodies:
        #     overlapping_pairs = [(body2, link2) for body2, link2 in get_bodies_in_region(get_moving_aabb(body1))
        #                          if body2 in obstacles]
        #     overlapping_bodies = {body2 for body2, _ in overlapping_pairs}
        #     for body2 in overlapping_bodies:
        #         if pairwise_collision(body1, body2, **kwargs):
        #             #print(get_body_name(body1), get_body_name(body2))
        #             if verbose: print(body1, body2)
        #             return True
        # return False

        for body1, body2 in product(moving_bodies, obstacles):
            if (not use_aabb or aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))) \
                    and pairwise_collision(body1, body2, **kwargs):
                #print(get_body_name(body1), get_body_name(body2))
                if verbose: print(body1, body2)
                return True
        return False
    return collision_fn

def interpolate_joint_waypoints(body, joints, waypoints, resolutions=None,
                                collision_fn=lambda *args, **kwargs: False, **kwargs):
    # TODO: unify with refine_path
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions, **kwargs)
    path = waypoints[:1]
    for waypoint in waypoints[1:]:
        assert len(joints) == len(waypoint)
        for q in list(extend_fn(path[-1], waypoint)):
            if collision_fn(q):
                return None
            path.append(q) # TODO: could instead yield
    return path
def plan_waypoints_joint_motion(body, joints, waypoints, start_conf=None, obstacles=[], attachments=[],
                                self_collisions=True, disabled_collisions=set(),
                                resolutions=None, custom_limits={}, max_distance=MAX_DISTANCE,
                                use_aabb=False, cache=True):
    if start_conf is None:
        start_conf = get_joint_positions(body, joints)
    assert len(start_conf) == len(joints)
    collision_fn = get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance,
                                    use_aabb=use_aabb, cache=cache)
    waypoints = [start_conf] + list(waypoints)
    for i, waypoint in enumerate(waypoints):
        if collision_fn(waypoint):
            #print('Warning: waypoint configuration {}/{} is in collision'.format(i, len(waypoints)))
            return None
    return interpolate_joint_waypoints(body, joints, waypoints, resolutions=resolutions, collision_fn=collision_fn)

def plan_direct_joint_motion(body, joints, end_conf, waypoints=None, **kwargs):
    if waypoints == None:
        return plan_waypoints_joint_motion(body, joints, [end_conf], **kwargs)
    return plan_waypoints_joint_motion(body, joints, waypoints, **kwargs)

def interpolate_joint_waypoints_force_aware(body, joints, waypoints, torque_fn, dynam_fn, resolutions=None,
                                collision_fn=lambda *args, **kwargs: False, **kwargs):
    # TODO: unify with refine_path
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions, **kwargs)
    path = waypoints[:1]
    for waypoint in waypoints[1:]:
        assert len(joints) == len(waypoint)
        for q in list(extend_fn(path[-1], waypoint)):
            if collision_fn(q):
                return None
            if not torque_fn(q):
                return None
    path.append(q) # TODO: could in stead yield
    path, _, vels, accels = dynam_fn(path, len(path))
    for i in range(len(path)):
        if not torque_fn(path[i], velocities=vels[i], accelerations=accels[i]):
            return None
    return path

def plan_waypoints_joint_motion_force_aware(body, joints, waypoints, torque_fn, dynam_fn, start_conf=None, obstacles=[], attachments=[],
                                self_collisions=True, disabled_collisions=set(),
                                resolutions=None, custom_limits={}, max_distance=MAX_DISTANCE,
                                use_aabb=False, cache=True):
    if start_conf is None:
        start_conf = get_joint_positions(body, joints)
    assert len(start_conf) == len(joints)
    collision_fn = get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance,
                                    use_aabb=use_aabb, cache=cache)
    waypoints = [start_conf] + list(waypoints)
    vels1 = [[0.0]*len(joints)]
    accels1 = [[0.0]*len(joints)]
    vels2 = [[0.0]*len(joints)]
    accels2 = [[0.0]*len(joints)]
    for i in range(1,len(waypoints)//2):
        vel1, acc1 = dynam_fn(waypoints[i], waypoints[i-1], vels1[-1], accels1[-1])
        vels1.append(vel1)
        accels1.append(acc1)
    for i in range(len(waypoints)-2, len(waypoints)//2 -1, -1):
        vel2, acc2 = dynam_fn(waypoints[i], waypoints[i+1], vels2[0], accels2[0])
        vels2 = [vel2] + vels2
        accels2 = [acc2] + accels2
    vels = vels1 + vels2
    accels = accels1 + accels2
    for i, waypoint in enumerate(waypoints):
        if collision_fn(waypoint):
            #print('Warning: waypoint configuration {}/{} is in collision'.format(i, len(waypoints)))
            return None
        if i > 0:
            if not torque_fn(waypoint, velocities=vels[1], accelerations=accels[i]):
                return None
    return interpolate_joint_waypoints_force_aware(body, joints, waypoints, torque_fn, dynam_fn, resolutions=resolutions, collision_fn=collision_fn)


def plan_direct_joint_motion_force_aware(body, joints, end_conf, torque_fn, dynam_fn, waypoints=None, **kwargs):
    if waypoints == None:
        return plan_waypoints_joint_motion_force_aware(body, joints, [end_conf], torque_fn, dynam_fn, **kwargs)
    return plan_waypoints_joint_motion_force_aware(body, joints, waypoints, torque_fn, dynam_fn, **kwargs)

def check_initial_end(start_conf, end_conf, collision_fn, verbose=True):
    # TODO: collision_fn might not accept kwargs
    if collision_fn(start_conf, verbose=verbose):
        print('Warning: initial configuration is in collision')
        return False
    if collision_fn(end_conf, verbose=verbose):
        print('Warning: end configuration is in collision')
        return False
    return True

def check_initial_end_force_aware(start_conf, end_conf, collision_fn, torque_fn, verbose=True):
    # TODO: collision_fn might not accept kwargs
    if collision_fn(start_conf, verbose=verbose):
        print('Warning: initial configuration is in collision')
        return False
    if collision_fn(end_conf, verbose=verbose):
        print('Warning: end configuration is in collision')
        return False
    if not torque_fn(start_conf):
        print('Warning: initial configuration excedes torque limits')
        print(start_conf)
        return False
    if not torque_fn(end_conf):
        print('Warning: end configuration excedes torque limits')
        return False
    return True

def create_trajectory(robot, joints, path, bodies, velocities=None, accelerations = None, movables=None, dts = None, ts=None):
    confs = []
    index = 0
    if velocities is not None:
        for i in range(len(velocities)):
            confs.append(Conf(robot, joints, path[i], velocities=velocities[i], movables=bodies, accelerations=accelerations[i], dt=dts[i]))
            index+=1
    for i in range(index, len(path)):
            confs.append(Conf(robot, joints, path[i], velocities=None))
    return Trajectory(confs, bodies=bodies, ts=ts)

class Conf(object):
    def __init__(self, body, joints, values=None, init=False, velocities=None, accelerations=None, movables=None, dt=None):
        self.body = body
        self.joints = joints
        if values is None:
            values = get_joint_positions(self.body, self.joints)
        self.values = tuple(values)
        self.init = init
        self.torque_joints = {}
        self.velocities = velocities[:len(joints)] if velocities is not None else velocities
        self.accelerations = accelerations[:len(joints)] if accelerations is not None else accelerations
        self.dt = dt
        self.movables=movables

    @property
    def bodies(self): # TODO: misnomer
        return flatten_links(self.body, get_moving_links(self.body, self.joints))
    def assign(self, bodies=[]):
        set_joint_positions_torque(self.body, self.joints, self.values, self.velocities)
    def iterate(self):
        yield self
    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)
    
class Trajectory():
    _draw = False
    def __init__(self, path, bodies, ts = None, reverse_traj = False):
        if reverse_traj:
            path = list(path)
            hold = deepcopy(path)
            for i in range(len(path)):
                path[i].velocities = np.multiply(path[i].velocities, -1)
                path[i].velocities = path[i].velocities.tolist()
                path[i].accelerations = np.multiply(path[i].accelerations, -1)
                path[i].accelerations = path[i].accelerations.tolist()
                path[i].dt = hold[(len(path) - 1) - i].dt
        self.path = tuple(path)
        self.bodies = bodies
        # if ts is None:
        #     self.file = f'/home/liam/exp_data/torque_data_{METHOD}_{MASS}kg.csv'
        #     self.file2 = f'/home/liam/exp_data/velocity_data_{METHOD}_{MASS}kg.csv'
        #     self.file3 = f'/home/liam/exp_data/eoat_velocity_data_{METHOD}_{MASS}kg.csv'
        # else:
        #     self.file = f'/home/liam/exp_data/{ts}_torque_data_{METHOD}_{MASS}kg.csv'
        #     self.file2 = f'/home/liam/exp_data/{ts}_velocity_data_{METHOD}_{MASS}kg.csv'
        #     self.file3 = f'/home/liam/exp_data/{ts}_eoat_velocity_data_{METHOD}_{MASS}kg.csv'