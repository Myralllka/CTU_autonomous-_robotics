#!/usr/bin/env python3
"""
ARO lab01 message ROS Subscriber
"""
import rospy  # the essential rospy import
from lab01_package.msg import AROMessage  # import our custom message
from std_msgs.msg import Float32

num_messages = 0
history = []

def callback(msg):
    """This is a callback that shall handle the incoming message"""
    # rospy.loginfo("%s", msg)
    global history
    history.append(msg.number)
    if (len(history) > num_messages):
        history = history[-num_messages:]
    s = sum(history)
    rospy.loginfo(s)


if __name__ == "__main__":
    rospy.init_node("ARO1subscriber")  # initialize the node with name "ARO1subscriber"
    num_messages = rospy.get_param("~num_messages")  # load num_messages parameter

    subscrbr = rospy.Subscriber("messages", AROMessage, callback, queue_size=10)

    publisher = rospy.Publisher("messages_sum", Float32, queue_size=10)
    rospy.spin()  # hold the execution here and periodically check for messages
