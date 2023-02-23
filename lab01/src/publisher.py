#!/usr/bin/env python3
"""
ROS custom message publisher
"""
import rospy  # import rospy, as always


from lab01_package.msg import AROMessage  # import our custom message
from numpy.random import rand  # just to be able to generate random numbers


if __name__ == "__main__":
    rospy.init_node("ARO1publisher")  # initialize node with the name "ARO1publisher"
    rate = rospy.Rate(1)  # instantiate Rate class with the frequency of 1 Hz

    publisher = rospy.Publisher("messages", AROMessage, queue_size=10)

    while not rospy.is_shutdown():  # loop until the node is killed or ros shutdowns
        mymsg = AROMessage()
        mymsg.message = "Hello"
        mymsg.number = rand()
        publisher.publish(mymsg)
        rate.sleep()  # pause the code execution for given by the rate period
