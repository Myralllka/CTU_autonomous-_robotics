#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function
from aro_msgs.srv import GenerateFrontier, PlanPath, PlanPathRequest, PlanPathResponse
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, PoseArray, Quaternion, Transform, TransformStamped
from aro_msgs.msg import FollowPathAction, FollowPathFeedback, FollowPathResult, FollowPathGoal
from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from aro_msgs.msg import Path
from std_srvs.srv import SetBool, SetBoolRequest
import actionlib
import numpy as np
from ros_numpy import msgify, numpify
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf2_py import TransformException
import tf2_ros
import time
from threading import RLock

np.set_printoptions(precision=3)


def slots(msg):
    """Return message attributes (slots) as list."""
    return [getattr(msg, var) for var in msg.__slots__]


def array(msg):
    """Return message attributes (slots) as array."""
    return np.array(slots(msg))


def pose2to3(pose2):
    """Convert Pose2D to Pose."""
    pose3 = Pose()
    pose3.position.x = pose2.x
    pose3.position.y = pose2.y
    rpy = 0.0, 0.0, pose2.theta
    q = quaternion_from_euler(*rpy)
    pose3.orientation = Quaternion(*q)
    return pose3


def tf3to2(tf):
    """Convert Transform to Pose2D."""
    pose2 = Pose2D()
    pose2.x = tf.translation.x
    pose2.y = tf.translation.y
    rpy = euler_from_quaternion(slots(tf.rotation))
    pose2.theta = rpy[2]
    return pose2


class Explorer(object):
    def __init__(self):
        self.current_path = None
        self.current_goal = None
        rospy.loginfo('Initializing explorer.')
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.robot_frame = rospy.get_param('~robot_frame', 'base_footprint')
        self.plan_interval = rospy.Duration(rospy.get_param('~plan_interval', 10.0))
        self.goal_reached_dist = rospy.get_param('~goal_reached_dist', 0.2)
        self.retries = rospy.get_param('~retries', 3)
        self.run_mode = rospy.get_param('~run_mode', 'eval')
        assert self.retries >= 1

        # Mind thread safety which is not guaranteed in rospy when multiple callbacks access the same data.
        self.lock = RLock()

        # Using exploration / frontier.py
        rospy.wait_for_service('get_closest_frontier')
        self.get_closest_frontier = rospy.ServiceProxy('get_closest_frontier', GenerateFrontier)
        rospy.wait_for_service('get_random_frontier')
        self.get_random_frontier = rospy.ServiceProxy('get_random_frontier', GenerateFrontier)
        rospy.wait_for_service('get_largest_frontier')
        self.get_largest_frontier = rospy.ServiceProxy('get_largest_frontier', GenerateFrontier)
        # rospy.wait_for_service('get_best_value_frontier')

        # Utilize stop_simulation service when submitting for faster evaluation. Refer to evaluator package for details.

        # rospy.wait_for_service('stop_simulation')
        # self.stop_sim = rospy.ServiceProxy('stop_simulation', SetBool)

        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)
        rospy.loginfo('Initializing services and publishers.')

        rospy.wait_for_service('plan_path')
        self.plan_path = rospy.ServiceProxy('plan_path', PlanPath)

        self.last_markers = None
        # For exploration / path_follower
        self.path_pub = rospy.Publisher('path', Path, queue_size=2)
        self.poses_pub = rospy.Publisher('poses', PoseArray, queue_size=2)

        self.action_client = actionlib.SimpleActionClient('follow_path', FollowPathAction)

        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)

        # Wait for initial pose.
        self.action_client.wait_for_server()

        self.init_pose = None
        while self.init_pose == None:
            try:
                self.init_pose = self.get_robot_pose()
                rospy.loginfo('Explorer initial pose:')
                rospy.loginfo(self.init_pose)
            except:
                rospy.loginfo('Robot odometry is not ready yet.')
                self.init_pose = None
                rospy.sleep(1.0)

        # Create an exploration control loop state machine using timer and packages created for your homeworks. Good luck.
        self.timer_exploration = rospy.Timer(rospy.Duration(1), self.tim_callback_exploration)
        rospy.loginfo('Initializing timer.')

    def tim_callback_exploration(self, timer):
        while not rospy.is_shutdown():
            print("[ATTANTION!!!!!!] timer callback start")
            rospy.loginfo('Timer callback running.')
            frontier_goal = self.get_largest_frontier()
            curent_pose = self.get_robot_pose()
            req = PlanPathRequest(curent_pose, frontier_goal.goal_pose)
            req = self.plan_path(req)
            if req.path is None or len(req.path) == 0:
                frontier_goal = self.get_random_frontier()
                curent_pose = self.get_robot_pose()
                if frontier_goal.goal_pose is None:
                    req = PlanPathRequest(curent_pose, self.init_pose)
                else:
                    req = PlanPathRequest(curent_pose, frontier_goal.goal_pose)
                req = self.plan_path(req)
                if req.path is None or len(req.path) == 0:
                    print("[ATTANTION!!!!!!] horrible mistake")
                    continue
            path = req.path
            self.perform_single_plan(path)
            self.current_goal = path[-1]

            while self.current_goal is not None:
                rospy.loginfo_throttle(5.0, 'Waiting for end of path following.')
                time.sleep(1.0)
            print("[ATTANTION!!!!!!] timer callback stop")
            break

    def lookup_transform(self, target_frame, source_frame, t,
                         no_wait_frame=None, timeout=rospy.Duration.from_sec(0.0)):
        if no_wait_frame is None or no_wait_frame == target_frame:
            return self.tf.lookup_transform(target_frame, source_frame, t, timeout=timeout)

        tf_n2t = self.tf.lookup_transform(self.map_frame, self.odom_frame, rospy.Time())
        tf_s2n = self.tf.lookup_transform(self.odom_frame, self.robot_frame, t, timeout=timeout)
        tf_s2t = TransformStamped()
        tf_s2t.header.frame_id = target_frame
        tf_s2t.header.stamp = t
        tf_s2t.child_frame_id = source_frame
        tf_s2t.transform = msgify(Transform,
                                  np.matmul(numpify(tf_n2t.transform),
                                            numpify(tf_s2n.transform)))
        return tf_s2t

    def marker_cb(self, msg):
        self.marker_pose = msg.pose
        rospy.loginfo('Marker localized at pose: [%.1f,%.1f].', self.marker_pose.position.x,
                      self.marker_pose.position.y)

    def get_robot_pose(self):
        tf = self.lookup_transform(self.map_frame, self.robot_frame, rospy.Time.now(),
                                   timeout=rospy.Duration.from_sec(0.5), no_wait_frame=self.odom_frame)
        pose = tf3to2(tf.transform)
        return pose

    def perform_single_plan(self, plan):
        """Sends single plan request to action server."""
        try:
            with self.lock:
                self.action_client.cancel_all_goals()
                self.current_path = plan
                path_stamped = Path()
                path_stamped.header.stamp = rospy.Time.now()
                path_stamped.header.frame_id = self.map_frame
                path_stamped.poses = [PoseStamped(pose=pose2to3(p)) for p in self.current_path]
                time.sleep(1)
                self.action_client.send_goal(FollowPathGoal(path_stamped.header, self.current_path),
                                             feedback_cb=self.action_feedback_cb,
                                             done_cb=self.action_done_cb)
                rospy.loginfo('New path sent to path follower.')

        except TransformException as ex:
            rospy.logerr('Robot pose lookup failed: %s.', ex)

    @staticmethod
    def pose2array(pose):
        return np.array([pose.x, pose.y, 0.0])

    @staticmethod
    def segment2point(seg_start, seg_end, point):
        """Returns distance of point from a segment."""
        seg = seg_end - seg_start
        len_seg = np.linalg.norm(seg)

        if len_seg == 0:
            return np.linalg.norm((seg_start - point)[:2])

        t = max(0, min(1, np.dot((point - seg_start)[:2], seg[:2]) / len_seg ** 2))
        proj = seg_start + t * seg

        return np.linalg.norm((point - proj)[:2])

    def path_point_dist(self, path, point):
        """Return distance from the path defined by sequence of points."""
        if path is None or len(path) < 2:
            rospy.logwarn('Cannot compute dist to path for empty path or path containing a single point.')
            return -1.0

        min_dist = 1e3
        for k in range(0, len(path) - 1):
            dist = self.segment2point(self.pose2array(path[k]), self.pose2array(path[k + 1]), point)
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def action_feedback_cb(self, feedback):
        """Action feedback callback for action client"""
        try:
            with self.lock:
                pose = array(self.get_robot_pose())

            dist = self.path_point_dist(self.current_path, pose)
            if dist < 0:
                rospy.logerr('Distance to path cannot be computed since path is too short.')

            # if len(self.current_path) > 0:
            #     self.max_start_dist = max(self.max_start_dist, np.sqrt(
            #         (pose[0] - self.current_path[0].x) ** 2 + (pose[1] - self.current_path[0].y) ** 2))

        except TransformException as ex:
            rospy.logerr('Robot pose lookup failed: %s.', ex)

        rospy.loginfo_throttle(2.0, 'Received control feedback. Position = [%.2f, %.2f], deviation = %.2f m.',
                               feedback.position.x, feedback.position.y, feedback.error)

    def action_done_cb(self, state, result):
        """Action done callback for action client"""
        self.current_goal = None
        rospy.loginfo('Control done. %s Final position = [%.2f, %.2f]', self.action_client.get_goal_status_text(),
                      result.finalPosition.x, result.finalPosition.y)


if __name__ == '__main__':
    rospy.init_node('explorer', log_level=rospy.INFO)
    node = Explorer()
    rospy.spin()
