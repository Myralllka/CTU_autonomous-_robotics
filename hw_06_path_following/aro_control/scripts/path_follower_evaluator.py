#!/usr/bin/env python3
"""
Simple node for path following testing.
"""
from __future__ import absolute_import, division, print_function
from geometry_msgs.msg import Point, Pose, Pose2D, PoseStamped, PoseArray, Quaternion, Transform, TransformStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud
import numpy as np
from ros_numpy import msgify, numpify
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf2_py import TransformException
import tf2_ros
from threading import RLock
import actionlib
from aro_msgs.msg import FollowPathAction, FollowPathFeedback, FollowPathResult, FollowPathGoal
import csv
import time

np.set_printoptions(precision=3)

def slots(msg):
    """Return message attributes (slots) as list."""
    return [getattr(msg, var) for var in msg.__slots__]

def array(msg):
    """Return message attributes (slots) as array."""
    return np.array(slots(msg))

def pose2to3(pose2):
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

def pose2ToArray(pose): 
    return np.array([pose.x, pose.y, 0.0]) 

def segmentPointDist(seg_start, seg_end, point):
    """Returns distance of point from a segment."""
    seg = seg_end - seg_start
    len_seg = np.linalg.norm(seg)

    if len_seg == 0: 
        return np.linalg.norm((seg_start - point)[:2])

    t = max(0, min(1, np.dot((point - seg_start)[:2], seg[:2]) / len_seg**2))
    proj = seg_start + t * seg 

    return np.linalg.norm((point - proj)[:2])

def pathPointDist(path, point):
    """Return distance from the path defined by sequence of points."""
    if path is None or len(path) < 2: 
        rospy.logwarn('Cannot compute dist to path for empty path or path containing a single point.')
        return -1.0

    min_dist = 1e3 # TODO: replace by flt max
    for k in range(0, len(path) - 1):
        dist = segmentPointDist(pose2ToArray(path[k]), pose2ToArray(path[k+1]), point)
        if dist < min_dist:
            min_dist = dist

    return min_dist

class EvaluatorAction(object):
    def __init__(self):
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.robot_frame = rospy.get_param('~robot_frame', 'base_footprint')
        self.goal_reached_dist = rospy.get_param('~goal_reached_dist', 0.2)
        self.paths_filename = rospy.get_param('~paths_filename', 'test_paths.csv')
        self.results_filename = rospy.get_param('~results_filename', 'results.csv')
        self.max_path_deviation = rospy.get_param('~max_path_deviation', 0.25)
        self.lock = RLock()
        self.current_pose = None
        self.current_goal = None  # Pose2D
        self.current_path = None  # Pose2D[]
        self.current_following_time_start = rospy.Time.now()
        self.current_following_time_end = rospy.Time.now()
        self.current_following_dev = [] # floats in meters

        # For exploration / path_follower
        self._ac = actionlib.SimpleActionClient('follow_path', FollowPathAction)

        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)
        
        rospy.loginfo('Waiting for path follower action server')
        self._ac.wait_for_server()
        rospy.loginfo('Action server found')

        time.sleep(5) # TODO wait for slam and tfs 

        rospy.loginfo('Start evaluation process')
        self.runTest(self.paths_filename)
        rospy.loginfo('Evaluation process finished.')

    def lookupTransform(self, target_frame, source_frame, time,
                         no_wait_frame=None, timeout=rospy.Duration.from_sec(0.0)):
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

    def getRobotPose(self):
        """Returns robot pose."""
        tf = self.lookupTransform(self.map_frame, self.robot_frame, rospy.Time.now(),
                                   timeout=rospy.Duration.from_sec(0.5), no_wait_frame=self.odom_frame)
        pose = tf3to2(tf.transform)
        return pose

    def loadPaths(self, filename):
        """Loads paths from file."""
        path_list = []
        with open(filename, 'r') as f:
            path = []
            r = csv.reader(f)
            for row in r:

                if 'PATH_END' in row:
                    rospy.loginfo('Path end detected.')
                    path_list.append(path)
                    path = []
                    continue
                 
                point = [float(v) for v in row]
                path.append(Pose2D(point[0], point[1], 0.0))
        
        rospy.loginfo('Loaded %d paths.', len(path_list))
        return path_list

    def runTest(self, filename):
        """Runs test for all paths specified in a file."""
        paths = self.loadPaths(filename)
        path_idx = 0

        for path in paths:
            path_idx += 1
            self.current_following_time_start = rospy.Time.now() 
            self.current_following_dev = [] # floats in meters
            self.performSinglePlan(path)
            self.current_goal = path[-1] 
            self.max_start_dist = 0.0
            
            # wait for following start
            time.sleep(3.0)

            while self.current_goal is not None:
                rospy.loginfo_throttle(5.0, 'Waiting for end of path following.')
                time.sleep(1.0)
            
            # process data
            dist = 0.0
            goal = self.current_path[-1]

            if self.max_start_dist < self.goal_reached_dist: 
                rospy.logwarn('Path following has not started at all.')

            dist = -1
            pose = np.array([0.0, 0.0, 0.0])
            
            try:
                with self.lock:
                    pose = array(self.getRobotPose())
                    dist = np.sqrt((pose[0] - goal.x)**2 + (pose[1] - goal.y)**2)
            except TransformException as ex:
                rospy.logerr('Robot pose lookup failed: %s.', ex)

            if len(self.current_following_dev):
                avg_dev = np.mean(self.current_following_dev)
                min_dev = min(self.current_following_dev)
                max_dev = max(self.current_following_dev)
            else:
                avg_dev = -1
                min_dev = -1
                max_dev = -1

            goal_reached = dist <= self.goal_reached_dist and self.max_start_dist > self.goal_reached_dist # second term meeded for circular paths
            following_time = (self.current_following_time_end - self.current_following_time_start).to_sec()
            self.printResults(path_idx, goal, pose, goal_reached, avg_dev, min_dev, max_dev, following_time)

        rospy.loginfo('--------- Path test end ---------')

                

    def printResults(self, path_idx, goal, final_pose, goal_reached, avg_dev, min_dev, max_dev, following_time): 
        """Print results to terminal."""
        rospy.loginfo('--------- Path following results ---------')
        rospy.loginfo('Path idx = %d', path_idx)
        rospy.loginfo('Goal [x, y] = [%.2f, %.2f]', goal.x, goal.y)
        rospy.loginfo('Final position [x, y] = [%.2f, %.2f]', final_pose[0], final_pose[1])
        gr_str = "True" if goal_reached else "False"
        rospy.loginfo('Goal reached = %s', gr_str)
        rospy.loginfo('Average deviation from path = %.2f m', avg_dev)
        rospy.loginfo('Minimum deviation from path = %.2f m', min_dev)
        rospy.loginfo('Maximum deviation from path = %.2f m', max_dev)
        rospy.loginfo('Following time = %.2f s', following_time)
        rospy.loginfo('------- Path following results end --------')
        
    def performSinglePlan(self, plan):
        """Sends single plan request to action server."""
        try:
            with self.lock:
                self._ac.cancel_all_goals()
                self.current_path = plan
                path_stamped = Path()
                path_stamped.header.stamp = rospy.Time.now()
                path_stamped.header.frame_id = self.map_frame
                path_stamped.poses = [PoseStamped(pose=pose2to3(p)) for p in self.current_path]
                time.sleep(1)
                self._ac.send_goal(FollowPathGoal(path_stamped.header, self.current_path),feedback_cb=self.action_feedback_cb, done_cb=self.action_done_cb)
                rospy.loginfo('New path sent to path follower.')

        except TransformException as ex:
            rospy.logerr('Robot pose lookup failed: %s.', ex)


    def action_feedback_cb(self, feedback):
        """Action feedback callback for action client"""
        try:
            with self.lock:
                pose = array(self.getRobotPose())

            dist = pathPointDist(self.current_path, pose) # TODO add conversion 
            if dist < 0:
                rospy.logerr('Distance to path cannot be computed since path is too short.')

            if len(self.current_path) > 0: 
                self.max_start_dist = max(self.max_start_dist, np.sqrt((pose[0] - self.current_path[0].x)**2 + (pose[1] - self.current_path[0].y)**2))

            self.current_following_dev.append(dist)

        except TransformException as ex:
            rospy.logerr('Robot pose lookup failed: %s.', ex)


        rospy.loginfo_throttle(2.0, 'Received control feedback. Position = [%.2f, %.2f], deviation = %.2f m.', feedback.position.x, feedback.position.y, feedback.error)

    def action_done_cb(self, state, result):
        """Action done callback for action client"""
        self.current_following_time_end = rospy.Time.now()
        self.current_goal = None
        rospy.loginfo('Control done. %s Final position = [%.2f, %.2f]', self._ac.get_goal_status_text(), result.finalPosition.x, result.finalPosition.y)

if __name__ == '__main__':
    rospy.init_node('explorer', log_level=rospy.INFO)
    node = EvaluatorAction()
    rospy.spin()
