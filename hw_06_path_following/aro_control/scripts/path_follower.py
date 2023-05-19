#!/usr/bin/env python3
"""
Simple path follower.

Always acts on the last received plan.
An empty plan means no action (stopping the robot).
"""

from __future__ import absolute_import, division, print_function

import time

import rospy
import numpy as np
from ros_numpy import msgify, numpify
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
from tf2_py import TransformException
import tf2_ros
from threading import RLock
from timeit import default_timer as timer
import actionlib
from aro_msgs.msg import FollowPathAction, FollowPathFeedback, FollowPathResult, FollowPathGoal
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Pose, Pose2D, Quaternion, Transform, TransformStamped, Twist, PoseStamped, Point
import random

np.set_printoptions(precision=3)


def dist_segment2point(seg_start, seg_end, point):
    """
    Returns distance of point from a segment.

    Parameters:
        seg_start (numpy array [2]): x, y coordinates of segment beginning.
        seg_end (numpy array [2]): x, y coordinates of segment end
        point (numpy array [2]): x, y coordinates of point

    Returns:
        dist (float): Euclidean distance between segment and point.
    """
    seg = seg_end - seg_start
    len_seg = np.linalg.norm(seg)

    if len_seg == 0:
        return np.linalg.norm(seg_start - point)

    t = max(0.0, min(1.0, np.dot(point - seg_start, seg) / len_seg ** 2))
    proj = seg_start + t * seg

    return np.linalg.norm(point - proj)


def point_slope_form(p1, p2):
    """
    Returns coefficients of a point-slope form of a line equation given by two points.

    Parameters:
        p1 (numpy array [2] (float)): x, y coordinates of first point
        p2 (numpy array [2] (float)): x, y coordinates of second point

    Returns:
        a, b, c (float): coefficients of a point-slope form of a line equation ax + by + c = 0.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dx != 0:
        return dy / dx, -1, -(dy / dx) * p1[0] + p1[1]
    else:
        return 1, 0, -p1[0]


def get_circ_line_intersect(p1, p2, circ_c, circ_r):
    """
    Returns intersection points of a line given by two points
    and circle given by its center and radius.

    Parameters:
        p1 (numpy array [2] (float)): x, y coordinates of point on the line
        p2 (numpy array [2] (float)): x, y coordinates of second point on the line
        circ_c (numpy array [2] (float)): x, y coordinates of circle center
        circ_r (float): circle radius

    Returns:
        points list(numpy array [2] (float)): x, y coordinates of intersection points
    """
    a, b, c = point_slope_form(p1, p2)  # get point-slope form of line ax + by + c = 0

    # find intersection based on a line and circle equation
    if b != 0:  # line is not parallel to y axis
        const_t = circ_c[0] ** 2 + 2 * circ_c[1] * c / b + circ_c[1] ** 2 + c ** 2 / b ** 2 - circ_r ** 2
        lin_t = 2 * a * c / b ** 2 + 2 * circ_c[1] * a / b - 2 * circ_c[0]
        quad_t = 1 + a ** 2 / b ** 2
        x_vals = np.roots([quad_t, lin_t, const_t])  # find roots of quadratic equation
        y_vals = [-a / b * x - c / b for x in x_vals]  # compute y from substitution in line eq.
    else:
        const_t = c ** 2 + 2 * circ_c[0] * c + circ_c[0] ** 2 + circ_c[1] ** 2 - circ_r ** 2
        lin_t = -2 * circ_c[1]
        quad_t = 1
        y_vals = np.real(np.roots([quad_t, lin_t, const_t]))
        x_vals = [p1[0] for i in y_vals]  # compute x from substitution in line eq.

    points = [[x_vals[i], y_vals[i]] for i in range(len(x_vals))]  # intersection points
    return points


def dist_line2point(line_begin, line_end, point):
    """
    Returns distance of point from a line.

    Parameters:
        line_begin (numpy array [2] (float)): x, y coordinates of line beginning.
        line_end (numpy array [2] (float)): x, y coordinates of line end
        point (numpy array [2] (float)): x, y coordinates of point

    Returns:
        dist (float): Euclidean distance between line and point.
    """
    p = point - line_begin
    v = line_end - line_begin
    return abs((v[0]) * (p[1]) - (p[0]) * (v[1])) / np.linalg.norm(v[:2])


def slots(msg):
    """
    Returns message attributes (slots) as list.

    Parameters:
        msg (ros msg): ROS message

    Returns:
        attributes (list): list of attributes of message
    """
    return [getattr(msg, var) for var in msg.__slots__]


def tf3to2(tf):
    """
    Converts tf to Pose2D.

    Parameters:
        tf (tf): tf to be converted

    Returns:
        pose2 (geometry_msgs/Pose2D): tf converted to Pose2D
    """
    pose2 = Pose2D()
    pose2.x = tf.translation.x
    pose2.y = tf.translation.y
    rpy = euler_from_quaternion(slots(tf.rotation))
    pose2.theta = rpy[2]
    return pose2


class PathFollower(object):
    def __init__(self):
        self.path_received_time = None
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')  # No-wait frame
        self.robot_frame = rospy.get_param('~robot_frame', 'base_footprint')  # base_footprint for simulation
        self.control_freq = rospy.get_param('~control_freq', 10.0)  # control loop frequency (Hz)
        assert 1.0 <= self.control_freq <= 10.0
        self.goal_reached_dist = rospy.get_param('~goal_reached_dist',
                                                 0.1)  # allowed distance from goal to be supposed reached
        self.max_path_dist = rospy.get_param('~max_path_dist',
                                             4.0)  # maximum distance from a path start to enable start of path following
        self.max_velocity = rospy.get_param('~max_velocity', 0.50)  # maximumm allowed velocity (m/s)
        self.max_angular_rate = rospy.get_param('~max_angular_rate', 2.0)  # maximum allowed angular rate (rad/s)
        self.look_ahead_dist = rospy.get_param('~look_ahead_dist', 0.35)  # look ahead distance for pure pursuit (m)
        self.radius_carrot_following = self.look_ahead_dist
        self.action_server = actionlib.SimpleActionServer('follow_path', FollowPathAction, execute_cb=self.control,
                                                          auto_start=False)
        self.action_server.register_preempt_callback(self.preempt_control)
        self.action_server.start()

        self.lock = RLock()
        self.path_msg = None  # nav_msgs Path message
        self.path = None  # n-by-3 path array
        self.path_frame = None  # string 
        self.path_index = 0  # path position index

        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)  # Publisher of velocity commands
        self.lookahead_point_pub = rospy.Publisher('/vis/lookahead_point', Marker,
                                                   queue_size=1)  # Publisher of the lookahead point (visualization only)
        self.path_pub = rospy.Publisher('/vis/path', Path, queue_size=2)
        self.waypoints_pub = rospy.Publisher('/vis/waypoints', PointCloud, queue_size=2)

        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)
        self.velocity = 0
        self.angular_rate = 0
        rospy.loginfo('Path follower initialized.')

    def lookup_transform(self, target_frame, source_frame, time,
                         no_wait_frame=None, timeout=rospy.Duration.from_sec(0.0)):
        """
        Returns transformation between to frames on specified time.

        Parameters:
            target_frame (String): target frame (transform to)
            source_frame (String): source frame (transform from)
            time (rospy time): reference time
            no_wait_frame (String): connection frame
            timeout:

        Returns:
            transformation (tf): tf from source frame to target frame
        """
        if no_wait_frame is None or no_wait_frame == target_frame:
            return self.tf.lookup_transform(target_frame, source_frame, time, timeout=timeout)

        tf_n2t = self.tf.lookup_transform(self.map_frame, self.odom_frame, rospy.Time())
        tf_s2n = self.tf.lookup_transform(self.odom_frame, self.robot_frame, time, timeout=timeout)
        tf_s2t = TransformStamped()
        tf_s2t.header.frame_id = target_frame
        tf_s2t.header.stamp = time
        tf_s2t.child_frame_id = source_frame
        tf_s2t.transform = msgify(Transform,
                                  np.matmul(numpify(tf_n2t.transform),
                                            numpify(tf_s2n.transform)))
        return tf_s2t

    def get_robot_pose(self, target_frame):
        """
        Returns robot's pose in the specified frame.

        Parameters:
            target_frame (String): the reference frame

        Returns:
            pose (geometry_msgs/Pose2D): Current position of the robot in the reference frame
        """
        t = timer()
        tf = self.lookup_transform(target_frame, self.robot_frame, rospy.Time(),
                                   timeout=rospy.Duration.from_sec(0.5), no_wait_frame=self.odom_frame)
        pose = tf3to2(tf.transform)
        return pose

    def clear_path(self):
        """
        Reinitializes path variables.
        """
        self.path_msg = None
        self.path = None
        self.path_index = 0

    def set_path_msg(self, msg):
        """
        Initializes path variables based on received path request.

        Parameters:
            msg (aro_msgs/FollowPathAction request): msg containing header and required path (array of Pose2D)
        """
        with self.lock:
            if len(msg.poses) > 0:
                self.path_msg = msg
                self.path = np.array([slots(p) for p in msg.poses])
                self.path_index = 0  # path position index
                self.path_frame = msg.header.frame_id
            else:
                self.clear_path()

        self.path_received_time = rospy.Time.now()
        rospy.loginfo('Path received (%i poses).', len(msg.poses))

    def stop_robot(self):
        with self.lock:
            self.send_velocity_command(self.velocity / 2, 0)
            time.sleep(4 / self.control_freq)
            self.send_velocity_command(self.velocity / 4, 0)
            time.sleep(4 / self.control_freq)
            self.send_velocity_command(self.velocity / 8, 0)
            time.sleep(4 / self.control_freq)
            self.velocity = 0
            self.angular_rate = 0
            self.send_velocity_command(0, 0)

    def get_lookahead_point(self, pose):
        """
        Returns lookahead point used as a reference point for path following.

        Parameters:
            pose (numpy array [3] (float)): x, y coordinates and heading of the robot

        Returns:
            goal (numpy array [2] (float)): x, y coordinates of the lookahead point
        """

        # Find local goal (lookahead point) on current path

        if self.path_index >= self.path.shape[0]:
            return self.path[-1][:2]

        path_next = self.path[self.path_index].flatten()[:2]

        lp1, lp2 = self.path[self.path_index - 1][:2], self.path[self.path_index][:2]

        pts = np.stack(get_circ_line_intersect(lp1, lp2, pose, self.radius_carrot_following))
        goal = pts[np.argmin(np.sqrt(np.sum((pts - path_next) ** 2, axis=1)))]

        if np.linalg.norm(pose[:2] - self.path[self.path_index][:2]) <= self.radius_carrot_following:
            print("going to the next point", self.path_index + 1)
            self.path_index += 1
        return goal

    def publishPathVisualization(self, path):
        """
        Publishes a given path as sequence of lines and point cloud of particular waypoints.

        Parameters:
            path (aro_msgs/FollowPathGoal): path to be visualized
        """

        if self.path_msg is None:
            return

        msg = PointCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = rospy.get_rostime()

        path_msg = Path()
        path_msg.header.frame_id = self.map_frame
        path_msg.header.stamp = rospy.get_rostime()

        for p in path.poses:
            msg.points.append(Point(x=p.x, y=p.y, z=0.0))
            pose_stamped = PoseStamped()
            pose_stamped.pose.position = msg.points[-1]
            path_msg.poses.append(pose_stamped)

        self.waypoints_pub.publish(msg)
        self.path_pub.publish(path_msg)

    def publishTemporaryGoal(self, goal_pose):
        """
        Publishes a given pose as red cicular marker.

        Parameters:
            goal_pose (numpy array [2] (float)): desired x, y coordinates of the marker in map frame
        """
        msg = Marker()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = rospy.get_rostime()
        msg.id = 1
        msg.type = 2
        msg.action = 0
        msg.pose = Pose()
        msg.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        msg.pose.position = Point(goal_pose[0], goal_pose[1], 0.0)
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.scale.x = 0.05
        msg.scale.y = 0.05
        msg.scale.z = 0.01
        self.lookahead_point_pub.publish(msg)

    def send_velocity_command(self, linear_velocity, angular_rate):
        """
        Calls command to set robot velocity and angular rate.

        Parameters:
            linear_velocity (float): desired forward linear velocity
            angular_rate (float): desired angular rate
        """
        msg = Twist()
        msg.angular.z = angular_rate
        msg.linear.x = linear_velocity
        self.cmd_pub.publish(msg)

    def control(self, msg):
        """
        Callback function of action server. Starts the path following process.

        Parameters:
            msg (aro_msgs/FollowPathGoal): msg containing header and required path (array of Pose2D)
        """
        rospy.loginfo('New control request obtained.')
        self.set_path_msg(msg)
        rate = rospy.Rate(self.control_freq)
        pose = np.array([0.0, 0.0, 0.0])

        self.publishPathVisualization(self.path_msg)

        # implement correct reaction if empty path is received
        if self.path_msg is None:
            self.stop_robot()
            rospy.logwarn('Empty path msg received.')
            self.action_server.set_succeeded(FollowPathResult(Pose2D(pose[0], pose[1], 0)), text='Empty msg received.')

        # Main control loop.
        while True:
            t = timer()  # save current time

            rospy.loginfo_throttle(1.0, 'Following path.')
            pose = np.array([0.0, 0.0, 0.0])
            with self.lock:

                # get robot pose
                try:
                    if self.path_frame is None:
                        rospy.logwarn('No valid path received so far, returning zero position.')
                    else:
                        if self.path_msg is None:
                            pose_msg = self.get_robot_pose(self.path_frame)
                        else:
                            pose_msg = self.get_robot_pose(self.path_msg.header.frame_id)

                        pose = np.array(slots(pose_msg))

                except TransformException as ex:
                    rospy.logerr('Robot pose lookup failed: %s.', ex)
                    continue

                if self.path_msg is None:
                    rospy.logwarn('Path following was preempted. Leaving the control loop.')
                    self.stop_robot()
                    return

                goal = self.get_lookahead_point(pose)

                # publish visualization of lookahaed point
            self.publishTemporaryGoal(goal)

            # # TODO: react on situation when the robot is too far from the path
            # if False:  # FIXME: replace by your code
            #     # TODO student's code
            #     self.action_server.set_aborted(FollowPathResult(Pose2D(pose[0], pose[1], 0)), text='Distance too high.')

            # Position displacement (direction and Euclidean distance)
            goal_dir = (goal - pose[:2])
            goal_dist = np.linalg.norm(goal_dir)

            # react on situation when the robot has reached the goal
            if self.path_index == self.path.shape[0] and np.linalg.norm(self.path[-1][:2] - pose[:2]) < self.goal_reached_dist:
                self.stop_robot()
                self.action_server.set_succeeded(FollowPathResult(Pose2D(pose[0], pose[1], 0)), text='Goal reached.')
                return

            # apply control law to produce control inputs
            tf = self.lookup_transform(self.robot_frame, self.path_frame, rospy.Time(0)).transform

            # compute a tmp goal in the robot coordinate frame

            translation = np.array([tf.translation.x, tf.translation.y])
            rotation = quaternion_matrix(slots(tf.rotation))[:2, :2]
            tp = rotation @ goal + translation
            # compute angular velocity
            dy = tp[1]
            R = (goal_dist ** 2) / (2 * dy)

            if tp[0] <= self.radius_carrot_following * 0.95:
                self.angular_rate += np.sign(dy) * self.max_angular_rate * 0.08
                self.velocity -= self.max_velocity * 0.08
                self.velocity = max(self.velocity, self.max_velocity/15)
            else:
                self.velocity += self.max_velocity * 0.08
                self.angular_rate = self.velocity / R

            # apply limits on angular rate and linear velocity
            self.angular_rate = np.clip(self.angular_rate, -self.max_angular_rate, self.max_angular_rate)
            self.velocity = np.clip(self.velocity, 0.0, self.max_velocity)

            # Apply desired velocity and angular rate
            self.send_velocity_command(self.velocity, self.angular_rate)

            self.action_server.publish_feedback(
                FollowPathFeedback(Pose2D(pose[0], pose[1], 0), 0.0))  # compute path deviation if needed
            rospy.logdebug(
                'Speed: %.2f m/s, angular rate: %.1f rad/s. (%.3f s), pose = [%.2f, %.2f], goal = [%.2f, %.2f], time = %.2f',
                self.velocity, self.angular_rate, timer() - t, pose[0], pose[1], goal[0], goal[1],
                rospy.get_rostime().to_sec())
            rate.sleep()

    def preempt_control(self):
        """
        Preemption callback function of action server. Safely preempts the path following process.
        """

        # TODO: implement correct behaviour when control is preempted

        with self.lock:
            pose = np.array([0.0, 0.0, 0.0])
            self.action_server.set_preempted(FollowPathResult(Pose2D(pose[0], pose[1], 0)), text='Control preempted.')

        rospy.logwarn('Control preempted.')


if __name__ == '__main__':
    rospy.init_node('path_follower', log_level=rospy.INFO)
    node = PathFollower()
    rospy.spin()
