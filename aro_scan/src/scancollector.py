#!/usr/bin/env python3
import rospy
import copy
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
 
class ScanCollector():
 
    def __init__(self):
        global batch_length
        global start_time
        # Initialize the node here
        rospy.init_node("aro_scan")
        # retrieve the necessary parameters from the parameter server
        # and store them into variables
        self.batch_length = rospy.get_param("/batch_length")
        self.start_time = None
        # create the listener object and assign a class method as the callback
        self.subscriber = rospy.Subscriber("scan", LaserScan, self.scan_callback, queue_size = 8)
        # possibly do some additional stuff
        self.publisher = rospy.Publisher("scan_filtered", Float32MultiArray, queue_size = 8)
        self.buff = []
        self.timestamps = []
 
    def scan_callback(self, msg: LaserScan):
        ranges_filtered = []
        step = msg.angle_increment
        counter = -1
        if self.start_time is None:
            self.start_time = msg.header.stamp.to_sec()
            rospy.set_param("/start_time", self.start_time)

        for each in msg.ranges:
            counter += 1
            deg = np.rad2deg(msg.angle_min + step * counter)
            #print(deg)
            if np.abs(deg) > 30:
                # Discard measurements taken at an angle greater than 30° or lower than -30°
                continue
            if (each < msg.range_min) or (each > msg.range_max):
                # data filtering, see documentation for LaserScan
                continue
            ranges_filtered.append(each)
        # Compute the mean of the remaining values and store it into a buffer
        mean = sum(ranges_filtered) / len(ranges_filtered)
        self.buff.append(mean)
        # Store the message timestamp as well
        self.timestamps.append(msg.header.stamp.to_sec())
        # If the number of stored values had reached the 
        # number specified in the batch_length global parameter: 
        # print(len(self.buff))
        if len(self.buff) >= self.batch_length:

            # Publish the data to the scan_filtered topic 
            # as std_msgs/Float32MultiArray containing computed 
            # means in field data. The layout field does not have 
            # to be filled with any values.
            filtered_msg = Float32MultiArray()
            filtered_msg.data = copy.copy(self.buff)
            self.publisher.publish(filtered_msg)

            # debug
            #plt.plot(self.timestamps, self.buff)
            #plt.show()
            # Stop accumulating the data
            self.buff = []
            self.timestamps = []

 
if __name__ == '__main__':
    sc = ScanCollector()
    rospy.spin()

