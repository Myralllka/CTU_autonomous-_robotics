#!/usr/bin/env python3
import rospy
import numpy as np  # you probably gonna need this
import sys

from geometry_msgs.msg import Twist, Vector3, Point
from std_srvs.srv import Empty, EmptyResponse
from gazebo_msgs.srv import DeleteLight
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import ros_numpy

from aro_robocop_goto_as.msg import (
    GoToPositionFeedback,
    GoToPositionResult,
    GoToPositionAction,
)

import actionlib


class RobocopGoToPositionAction:
    """Class that implements GoTo action server"""

    def __init__(self):
        rospy.init_node("robocop_goto_as")
        self.speed_linear = 0
        self.speed_angular = 0
        self.min_dist_ahead = 42
        rospy.on_shutdown(self.full_stop)

        # create pub_cmd publisher to "/cmd_vel" topic with Twist message type
        self.pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1, latch=True)

        # subscribe for the lidar scan in "/scan" topic with LaserScan message type and use the self.scan_callback for it
        self.sub_scan = rospy.Subscriber("/scan", LaserScan, self.scan_callback)

        # subscribe odometry in "/odom" topic with Odometry message type and use the self.odom_callback for it
        self.current_point = np.array(
            [0, 0, 0]
        )  # for storing current positions of the robot
        self.current_yaw_rad = 0  # for storing current yaw of the robot
        self.odom_subscriber = rospy.Subscriber("/odom", Odometry, self.odom_callback)

        self.feedback = GoToPositionFeedback()  # action server feedback message
        self.result = GoToPositionResult()  # ction server result message

        # create action server object on topic "robocop_goto_as" with action type GoToPositionAction and callback self.execute_cb
        self.action_server = actionlib.SimpleActionServer("robovop_goto_as", GoToPositionAction, execute_cb=self.execute_cb)
        self.action_server.start()

    def scan_callback(self, msg):
        """callback method for geting lidar scan and to calculate minimal distance in range -30 to 30 deg"""
        # angle = msg.angle_min + i * msg.angle_increment
        id_from = int(((-30 / 180) * np.pi - msg.angle_min) / msg.angle_increment)
        id_to = int(((30 / 180) * np.pi - msg.angle_min) / msg.angle_increment)
        min_dist = msg.range_max
        for i in range(id_from, id_to):
            if (
                msg.ranges[i] > msg.range_min
                and msg.ranges[i] < msg.range_max
                and msg.ranges[i] < min_dist
            ):
                min_dist = msg.ranges[i]
        self.min_dist_ahead = min_dist

    def odom_callback(self, msg):
        """callback method for getting odometry == position of the robot"""
        self.current_point = ros_numpy.geometry.point_to_numpy(
            msg.pose.pose.position
        )  # x,y,z position of robot
        q = msg.pose.pose.orientation
        self.current_yaw_rad = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y), q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z
        )  # robot yaw in radians assuming always only rotating around z axis

    def execute_cb(self, goal):
        goal_point_vec = ros_numpy.geometry.point_to_numpy(goal.position)

        # Run the while driving loop on 10Hz
        rate = rospy.Rate(10)
        success = True

        rospy.loginfo("GoToServer: starting the GoTo action")

        # start executing the action
        driving = True
        while driving:
            # check that preempt has not been requested by the client
            if self.action_server.is_preempt_requested():
                rospy.loginfo("GoToServer: Preempted action")
                self.result.success = False
                self.result.message = "canceled"
                self.action_server.set_preempted(self.result)
                self.full_stop()  # set speeds to zero and send them
                success = False
                break

            # calculate the position and heading (angle) errors
            to_goal_vector = goal_point_vec - self.current_point
            to_goal_angle = np.arctan2(to_goal_vector[1], to_goal_vector[0])
            angle_error = to_goal_angle - self.current_yaw_rad
            if angle_error > np.pi:
                angle_error -= 2 * np.pi
            elif angle_error < -np.pi:
                angle_error += 2 * np.pi
            position_error = np.linalg.norm(to_goal_vector)

            # fill the position and angle errors to the feedback message  self.feedback
            self.feedback.angle_error = angle_error
            self.feedback.position_error = position_error

            # TODO set appropriate self.speed_angular and self.speed_linear based on the error values
            if angle_error < -0.1 or angle_error > 0.1:
                self.feedback.message = "rotating to align angle"
                self.angle_error = np.sign(angle_error) * 0.1
            else:
                self.feedback.message = "going forward only"
                self.speed_angular = 0
                self.speed_linear = 0.1

            if position_error < 0.1:
                self.feedback.message = "reached goal"
                driving = False
                success = True
            if self.min_dist_ahead < 0.3 and self.speed_linear > 0:
                self.feedback.message = "stopped before collision"
                driving = False
                success = False

            self.pub_cmd.publish(
                Twist(
                    linear=Vector3(x=self.speed_linear),
                    angular=Vector3(z=self.speed_angular),
                )
            )

            # send the feedback message (self.feedback) using publish_feedback of the action server
            self.action_server.publish_feedback(self.feedback)

            rate.sleep()  # sleep for the rest of the rate

        # fill the results and either call set_succeeded or set_aborted in the self.action_server with the result message
        self.result.success = success
        self.full_stop()  # set speeds to zero and send them
        if success:
            rospy.loginfo("GoToServer: Succeeded action")
            # fill the result messages and send it
            self.result.message = "GoToServer: Succeeded action"
            self.action_server.set_succeeded(self.result)

        else:
            rospy.loginfo("GoToServer: Failed action")
            # fill the result messages and send it
            self.result.message = "GoToServer: Failed action"
            self.action_server.set_aborted(self.result)

    def full_stop(self):
        """method to stop the movement"""
        self.speed_linear = 0
        self.speed_angular = 0
        self.pub_cmd.publish(
            Twist(
                linear=Vector3(x=self.speed_linear),
                angular=Vector3(z=self.speed_angular),
            )
        )


if __name__ == "__main__":
    rc = RobocopGoToPositionAction()
    rospy.spin()
