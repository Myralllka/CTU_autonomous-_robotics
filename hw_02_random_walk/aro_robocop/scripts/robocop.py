#!/usr/bin/env python3
import rospy
import numpy as np  # you probably gonna need this
# import laser scan message
from sensor_msgs.msg import LaserScan
# I'll help you with the other imports:
from geometry_msgs.msg import Twist, Vector3
from gazebo_msgs.srv import DeleteLight
from std_srvs.srv import Empty, EmptyResponse

class Robocop():
    SERVICE_NAME = "/gazebo/delete_light"  # call this service to turn of the lights
    RUNTIME_LIMIT = 30  # check out rospy.Duration and set this limit to 30 seconds
    
    def __init__(self):
        # initialize the node
        rospy.init_node("robocop")
        # register listener for "/scan" topic & publisher for "/cmd_vel" topic (use arg "latch=True")
        self.sub_scan = rospy.Subscriber("scan", LaserScan, self.scan_cb, queue_size = 8)
        self.pub_cmd = rospy.Publisher("cmd_vel", Twist, queue_size=10, latch=True)
        # wait till the delete light service is up
        rospy.wait_for_service(self.SERVICE_NAME, timeout=None)
        # create proxy for the delete light service
        self.srv_shutdown_sun = rospy.ServiceProxy(self.SERVICE_NAME, DeleteLight)
        # remember start time (something about rospy.Time...)
        self.start_time = rospy.Time.now().to_sec()
        # you are probably going to need to add some variables

        self.speed_limit_angular = 0.8
        self.speed_limit_slow = 0.2
        self.speed_limit_fast = 0.7
        self.speed_linear = 0
        self.speed_angular = 0

        rospy.on_shutdown(self.full_stop)

    def scan_cb(self, msg):
        if (msg.header.stamp.to_sec() - self.start_time) > self.RUNTIME_LIMIT:  # <-- if (message timestamp - start timestamp) > self.RUNTIME_LIMIT
            # unregister this listener
            self.sub_scan.unregister()
            # finish this line: response = <something>.call(light_name='sun')
            response = self.srv_shutdown_sun.call(light_name='sun')
            rospy.loginfo(str(response))  # printing response is good for debugging
            self.full_stop()
            return

        angles = np.rad2deg(np.arange(msg.angle_min, msg.angle_max + msg.angle_increment, msg.angle_increment)) 
        filtered_msg = np.array(msg.ranges)

        mask = np.logical_and(filtered_msg > msg.range_min, filtered_msg < msg.range_max)

        rng_front = np.min(np.array(msg.ranges)[np.logical_and(np.logical_or(angles <= 30, angles >= 330), mask)])

        rng_frontright = np.min(np.array(msg.ranges)[np.logical_and(angles <= 30, mask)])
        rng_frontleft = np.min(np.array(msg.ranges)[np.logical_and(angles >= 330, mask)])
        
        #print(f'{rng_front=}')

        if rng_front < 1.5:
            if rng_front < 0.7:
                # stop the robot
                self.speed_linear = -self.speed_limit_slow
                rospy.loginfo("Rotation only.")
            else:
                if self.speed_linear < self.speed_limit_slow:
                    self.speed_linear = self.speed_limit_slow
                    # move slowly forward
                    rospy.loginfo("Slow movement.")

            left_mask = np.logical_and(np.logical_and(angles >= 30, angles <= 70), mask)
            right_mask = np.logical_and(np.logical_and(angles >= 290, angles <= 330), mask)

            right = np.array(msg.ranges)[right_mask]
            left = np.array(msg.ranges)[left_mask]

            rng_right = np.sum(right) / len(right)  # compute average distance of obstacles on the right (30-70 degrees to the right)
            rng_left = np.sum(left) / len(left)  # compute average distance of obstacles on the left (30-70 degrees to the left)

            #rng_left = np.min(left)
            #rng_right = np.min(rng_right)
            #print(f'{rng_right=}')
            #print(f'{rng_left=}')
            if rng_left > rng_right:
                if self.speed_angular <= 0:  # if not rotating left
                    # rotate left
                    self.speed_angular = self.speed_limit_angular
                    rospy.loginfo("Started rotating left.")
            else:
                if self.speed_angular >= 0:  # if not rotating right
                    self.speed_angular = -self.speed_limit_angular
                    # rotate right
                    rospy.loginfo("Started rotating right.")
        else:
            if self.speed_angular != 0:  # if currently rotating
                self.speed_angular = 0
                # stop rotation
                rospy.loginfo("Rotation stopped.")
            if self.speed_linear <= self.speed_limit_fast:  # if not moving fast
                self.speed_linear = self.speed_limit_fast
                # start moving fast
                rospy.loginfo("Go!")
        
        self.pub_cmd.publish(
            Twist(
                linear=Vector3(x=self.speed_linear),
                angular=Vector3(z=self.speed_angular)
            )
        )

    def full_stop(self):
        rospy.loginfo("Stopped.")
        # use your publisher to send commands to the robot
        self.speed_linear = 0
        self.speed_angular = 0
        self.pub_cmd.publish(
            Twist(
                linear=Vector3(x=self.speed_linear),
                angular=Vector3(z=self.speed_angular)
            )
        )
        self.pub_cmd.unregister()


if __name__ == '__main__':
    rc = Robocop()
    rospy.spin()
