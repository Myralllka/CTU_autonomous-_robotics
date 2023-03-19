#!/usr/bin/env python3
import rospy
import numpy as np  # you probably gonna need this
import sys
from geometry_msgs.msg import PoseStamped
import actionlib
from actionlib_msgs.msg import GoalStatus

from aro_robocop_goto_as.msg import (
    GoToPositionFeedback,
    GoToPositionResult,
    GoToPositionAction,
    GoToPositionGoal,
)


class RobocopRvizGoto:
    """Class that implements GoTo action client and listens to RViz 2D Nav Goal to drive the robot to desire position"""

    def __init__(self):
        rospy.init_node("robotop_rviz_goto_ac")

        # create action client self.client for action "robocop_goto_as" of type GoToPositionAction
        self.client = actionlib.SimpleActionClient("robovop_goto_as", GoToPositionAction)
        self.client.wait_for_server()

        self.sub_scan = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.point_clicked
        )  # subscribe the position clicked in RViz

    def point_clicked(self, msg):
        """Callback for RViz 2D Nav Goal click"""
        state = self.client.get_state()
        # cancel the all goals if the server is active
        if state == GoalStatus.ACTIVE:
            self.client.cancel_all_goals()
            self.action_status_timer.shutdown()
            rospy.sleep(0.5)  # sleep to settle the state of action server

        # create the goal message of type GoToPositionGoal from the PoseStamped msg
        goal = GoToPositionGoal()
        goal.position = msg.pose.position
        self.client.send_goal(goal)
        # create a Timer self.action_status_timer with self.check_action_status callback  to check the status of the action server
        self.action_status_timer = rospy.Timer(rospy.Duration(0.1), self.check_action_status)

    def check_action_status(self, timer):
        state = self.client.get_state()
        if state != GoalStatus.ACTIVE:
            result = self.client.get_result()
            rospy.loginfo("RvizClick - Action server stopped")
            if result is not None:
                rospy.loginfo("RvizClick - Message %s", result.message)
            self.action_status_timer.shutdown()  # stop the timer
        elif state == GoalStatus.ACTIVE:
            rospy.loginfo("RvizClick - Action still active")


if __name__ == "__main__":
    rrg = RobocopRvizGoto()
    rospy.spin()
