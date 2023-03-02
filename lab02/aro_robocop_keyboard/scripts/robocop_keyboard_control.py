#!/usr/bin/env python3
import rospy
import numpy as np  # you probably gonna need this
import sys

from geometry_msgs.msg import Twist, Vector3
from std_srvs.srv import Empty, EmptyResponse
from gazebo_msgs.srv import DeleteLight
import termios
import tty
from select import select


class Robocop:
    def __init__(self):
        rospy.init_node("robocop_keyboard_control")
        self.key_time_wait = 0.1
        self.speed_linear = 0
        self.speed_angular = 0
        self.speed_max = rospy.get_param("~max_speed")
        self.speed_change = rospy.get_param("~change_speed")


        self.terminal_settings = termios.tcgetattr(
            sys.stdin
        )  # save terminal setting for later use with key press listening

        self.pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # service to stop the robot
        self.stop_service = rospy.Service("stop", Empty, self.stop_service_callback)

        rospy.on_shutdown(self.full_stop)  # function to run when node is killed
        self.key_press_callback()  # blocking function to run the keyboard control

    def key_press_callback(self):
        """method that continuously listen for the key pressed"""

        while not rospy.is_shutdown():  # loop until node is killed

            # loading the pressed keys with some terminal magic
            tty.setraw(sys.stdin.fileno())
            rlist, _, _ = select([sys.stdin], [], [], self.key_time_wait)
            last_key = ""
            if rlist:
                last_key = sys.stdin.read(1)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.terminal_settings)

            if last_key == "w":
                if self.speed_linear <= self.speed_max:
                    self.speed_linear += self.speed_change
                else:
                    rospy.loginfo("max linear speed already reached!")

                rospy.loginfo("w pressed")

            elif last_key == "x":
                if self.speed_linear >= -self.speed_max:
                    self.speed_linear -= self.speed_change
                else:
                    rospy.loginfo("min linear speed already reached!")

                rospy.loginfo("x pressed")

            elif last_key == "a":
                if self.speed_angular <= self.speed_max:
                    self.speed_angular += self.speed_change
                else:
                    rospy.loginfo("min angular speed already reached!")

                rospy.loginfo("a pressed")

            elif last_key == "d":
                if self.speed_angular >= -self.speed_max:
                    self.speed_angular -= self.speed_change
                else:
                    rospy.loginfo("max angular speed already reached!")

                rospy.loginfo("d pressed")

            elif last_key == "s":

                self.speed_linear = 0
                self.speed_angular = 0
                rospy.loginfo("s pressed")
            
            rospy.loginfo(f"{self.speed_angular=}, {self.speed_linear=}")
            self.pub_cmd.publish(Twist(linear=Vector3(x=self.speed_linear), angular=Vector3(z=self.speed_angular)))
        

            if last_key == "\x03":
                rospy.signal_shutdown("killed by user")
                break

    def stop_service_callback(self, req):
        """callback to stop service"""
        rospy.loginfo("stopping service received")
        self.full_stop()
        return EmptyResponse()

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
    rc = Robocop()
    rospy.spin()
