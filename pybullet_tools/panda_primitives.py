import time

from itertools import count

from .utils import get_pose, set_pose, get_movable_joints, get_num_joints, \
    set_joint_positions, add_fixed_constraint, enable_real_time, disable_real_time, joint_controller, \
    enable_gravity, get_refine_fn, wait_for_duration, link_from_name, get_body_name, sample_placement, \
    end_effector_from_body, approach_from_grasp, plan_joint_motion, GraspInfo, Pose, INF, Point, \
    inverse_kinematics, pairwise_collision, remove_fixed_constraint, Attachment, get_sample_fn, \
    step_simulation, refine_path, plan_direct_joint_motion, get_joint_positions, dump_world, wait_if_gui, flatten
from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver

DEBUG_FAILURE = False

##################################################

class BodyPose(object):
    num = count()
    def __init__(self, body, pose=None):
        if pose is None:
            pose = get_pose(body)
        self.body = body
        self.pose = pose
        self.index = next(self.num)
    
    @property
    def value(self):
        return self.pose
    
    def assign(self):
        set_pose(self.body, self.pose)
        return self.pose
    
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'p{}'.format(index)


class BodyGrasp(object):
    num = count()
    def __init__(self, body, grasp_pose, approach_pose, robot, link):
        self.body = body
        self.grasp_pose = grasp_pose
        self.approach_pose = approach_pose
        self.robot = robot
        self.link = link
        self.index = next(self.num)
    
    @property
    def value(self):
        return self.grasp_pose
    
    @property
    def approach(self):
        return self.approach_pose
    
    #def constraint(self):
    #    grasp_constraint()
    
    def attachment(self):
        return Attachment(self.robot, self.link, self.grasp_pose, self.body)
    
    def assign(self):
        return self.attachment().assign()
    
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'g{}'.format(index)


class BodyConf(object):
    num = count()
    def __init__(self, body, configuration=None, joints=None):
        if joints is None:
            joints = get_movable_joints(body)
        if configuration is None:
            configuration = get_joint_positions(body, joints)
        self.body = body
        self.joints = joints
        self.configuration = configuration
        self.index = next(self.num)
    
    @property
    def values(self):
        return self.configuration
    
    def assign(self):
        set_joint_positions(self.body, self.joints, self.configuration)
        return self.configuration
    
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'q{}'.format(index)


class BodyPath(object):
    def __init__(self, body, path, joints=None, attachments=[]):
        if joints is None:
            joints = get_movable_joints(body)
        self.body = body
        self.path = path
        self.joints = joints
        self.attachments = attachments
    
    def bodies(self):
        return set([self.body] + [attachment.body for attachment in self.attachments])
    
    def iterator(self):
        # TODO: compute and cache these
        # TODO: compute bounding boxes as well
        for i, configuration in enumerate(self.path):
            set_joint_positions(self.body, self.joints, configuration)
            for grasp in self.attachments:
                grasp.assign()
            yield i
    
    def control(self, real_time=False, dt=0):
        # TODO: just waypoints
        if real_time:
            enable_real_time()
        else:
            disable_real_time()
        for values in self.path:
            for _ in joint_controller(self.body, self.joints, values):
                enable_gravity()
                if not real_time:
                    step_simulation()
                time.sleep(dt)
    
    def refine(self, num_steps=0):
        return self.__class__(self.body, refine_path(self.body, self.joints, self.path, num_steps), self.joints, self.attachments)
    
    def reverse(self):
        return self.__class__(self.body, self.path[::-1], self.joints, self.attachments)
    
    def __repr__(self):
        return '{}({},{},{},{})'.format(self.__class__.__name__, self.body, len(self.joints), len(self.path), len(self.attachments))

##################################################

class ApplyForce(object):
    def __init__(self, body, robot, link):
        self.body = body
        self.robot = robot
        self.link = link
    def bodies(self):
        return {self.body, self.robot}
    def iterator(self, **kwargs):
        return []
    def refine(self, **kwargs):
        return self
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.robot, self.body)

class Attach(ApplyForce):
    def control(self, **kwargs):
        # TODO: store the constraint_id?
        add_fixed_constraint(self.body, self.robot, self.link)
    def reverse(self):
        return Detach(self.body, self.robot, self.link)

class Detach(ApplyForce):
    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)
    def reverse(self):
        return Attach(self.body, self.robot, self.link)

class Command(object):
    num = count()
    def __init__(self, body_paths):
        self.body_paths = body_paths
        self.index = next(self.num)
    
    def bodies(self):
        return set(flatten(path.bodies() for path in self.body_paths))
    
    def step(self):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                msg = '{},{}) step?'.format(i, j)
                wait_if_gui(msg)
                #print(msg)
    
    def execute(self, time_step=0.05):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                #time.sleep(time_step)
                wait_for_duration(time_step)
    
    def control(self, real_time=False, dt=0): # TODO: real_time
        for body_path in self.body_paths:
            body_path.control(real_time=real_time, dt=dt)
    
    def refine(self, **kwargs):
        return self.__class__([body_path.refine(**kwargs) for body_path in self.body_paths])
    
    def reverse(self):
        return self.__class__([body_path.reverse() for body_path in reversed(self.body_paths)])
    
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'c{}'.format(index)

#######################################################


def get_stable_gen(fixed=[]):
    def gen(body, surface):
        while True:
            pose = sample_placement(body, surface)
            if (pose is None) or any(pairwise_collision(body, b) for b in fixed):
                continue
            body_pose = BodyPose(body, pose)
            yield (body_pose,)
    return gen


    movable_joints = get_movable_joints(robot)
    sample_fn = get_sample_fn(robot, movable_joints)
    def fn(body, pose, grasp):
        obstacles = [body] + fixed
        gripper_pose = end_effector_from_body(pose.pose, grasp.grasp_pose) # gripper in world
        approach_pose = approach_from_grasp(grasp.approach_pose, gripper_pose)
        for _ in range(num_attempts):
            set_joint_positions(robot, movable_joints, sample_fn()) # Random seed
            # TODO: multiple attempts?
            q_approach = inverse_kinematics(robot, grasp.link, approach_pose)
            if (q_approach is None) or any(pairwise_collision(robot, b) for b in obstacles):
                continue
            conf = BodyConf(robot, q_approach)
            q_grasp = inverse_kinematics(robot, grasp.link, gripper_pose)
            if (q_grasp is None) or any(pairwise_collision(robot, b) for b in obstacles):
                continue
            if teleport:
                path = [q_approach, q_grasp]
            else:
                conf.assign()
                #direction, _ = grasp.approach_pose
                #path = workspace_trajectory(robot, grasp.link, point_from_pose(approach_pose), -direction,
                #                                   quat_from_pose(approach_pose))
                path = plan_direct_joint_motion(robot, conf.joints, q_grasp, obstacles=obstacles)
                if path is None:
                    if DEBUG_FAILURE: wait_if_gui('Approach motion failed')
                    continue
            command = Command([BodyPath(robot, path),
                               Attach(body, robot, grasp.link),
                               BodyPath(robot, path[::-1], attachments=[grasp])]) # go, grasp, back
            return (conf, command)
            # TODO: holding collisions
        return None
    return fn


def get_ik_conf(robot, ik_info, ik_joints, tool_link, world_from_pose, obstacles=[]):
    
    q_conf = None
    for conf in either_inverse_kinematics(robot, ik_info, tool_link, world_from_pose, use_pybullet=False, 
                                        max_distance=INF, max_time=10, max_candidates=INF):
        set_joint_positions(robot, ik_joints, conf)
        if any(pairwise_collision(robot, b) for b in obstacles):
            continue
        else:
            q_conf = conf
            break
    return q_conf


##################################################

def free_motion(robot, joints, conf1, conf2, obstacles=[], self_collisions=True):
    set_joint_positions(robot, joints, conf1)
    path = plan_joint_motion(robot, joints, conf2, obstacles=obstacles, self_collisions=self_collisions)
    if path is None:
        return None
    command = Command([BodyPath(robot, path, joints=conf2.joints)])
    
    return command

##################################################

def get_movable_collision_test():
    def test(command, body, pose):
        if body in command.bodies():
            return False
        pose.assign()
        for path in command.body_paths:
            moving = path.bodies()
            if body in moving:
                # TODO: cannot collide with itself
                continue
            for _ in path.iterator():
                # TODO: could shuffle this
                if any(pairwise_collision(mov, body) for mov in moving):
                    if DEBUG_FAILURE: wait_if_gui('Movable collision')
                    return True
        return False
    return test
