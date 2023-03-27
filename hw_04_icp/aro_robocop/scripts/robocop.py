#!/usr/bin/env python3
import rospy
import numpy as np  # you probably gonna need this
# TODO: import laser scan message
# I'll help you with the other imports:
from geometry_msgs.msg import Twist, Vector3
from gazebo_msgs.srv import DeleteLight

class Robocop():
    SERVICE_NAME = "/gazebo/delete_light"  # call this service to turn of the lights
    RUNTIME_LIMIT = None  # check out rospy.Duration and set this limit to 30 seconds

    def __init__(self):
        # TODO: initialize the node
        # TODO: register listener for "/scan" topic & publisher for "/cmd_vel" topic (use arg "latch=True")
        # TODO: wait till the delete light service is up
        # TODO: create proxy for the delete light service
        # TODO: remember start time (something about rospy.Time...)
        # TODO: you are probably going to need to add some variables
        self.speed_limit_angular = 0.4
        self.speed_limit_slow = 0.2
        self.speed_limit_fast = 0.5

        rospy.on_shutdown(self.full_stop)

    def scan_cb(self, msg):
        if None:  # <-- if (message timestamp - start timestamp) > self.RUNTIME_LIMIT
            # TODO unregister this listener
            # TODO: finis this line: response = <something>.call(light_name='sun')
            rospy.loginfo(str(response))  # printing response is good for debugging
            self.full_stop()

        rng_front = None  # TODO: compute minimum distance for scan angles between -20 to 20 degrees

        if rng_front < 1.5:
            if rng_front < 0.5:
                # TODO: stop the robot
                rospy.loginfo("Movement stopped.")
            else:
                if self.linear == 0:
                    # TODO: move slowly forward
                    rospy.loginfo("Slow movement.")

            rng_right = None  # compute average distance of obstacles on the right (30-70 degrees to the right)
            rng_left = None  # compute average distance of obstacles on the left (30-70 degrees to the left)
            if rng_left > rng_right:
                if None:  # TODO: if not rotating left
                    # TODO: rotate left
                    rospy.loginfo("Started rotating left.")
            else:
                if None:  # TODO: if not rotating right
                    # TODO: rotate right
                    rospy.loginfo("Started rotating right.")
        else:
            if None:  # TODO: if currently rotating
                # TODO: stop rotation
                rospy.loginfo("Rotation stopped.")
            if None:  # TODO: if not moving fast
                # TODO: start moving fast
                rospy.loginfo("Go!")

    def full_stop(self):
        rospy.loginfo("Stopped.")
        # TODO: use your publisher to send commands to the robot
        self.pub_cmd.publish(
            Twist(
                linear=Vector3(x=0.0),
                angular=Vector3(z=0.0)
            )
        )


if __name__ == '__main__':
    rc = Robocop()
    rospy.spin()
