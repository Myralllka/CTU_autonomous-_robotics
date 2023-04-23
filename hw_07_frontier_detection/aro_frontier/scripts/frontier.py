#!/usr/bin/env python3
from __future__ import division, print_function
import rospy
import numpy as np
from scipy.ndimage import morphology
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose2D
from aro_msgs.srv import GenerateFrontier, GenerateFrontierRequest, GenerateFrontierResponse
import tf2_ros
import geometry_msgs.msg

"""
Here are imports that you are most likely will need. However, you may wish to remove or add your own import.
"""


class FrontierExplorer():

    def __init__(self):
        # Initialize the node
        rospy.init_node("frontier_explorer")

        # Get some useful parameters
        self.mapFrame = rospy.get_param("~map_frame", "map")
        self.robotFrame = rospy.get_param("~robot_frame", "base_footprint")
        self.robotDiameter = float(rospy.get_param("~robot_diameter", 0.8))
        self.occupancyThreshold = int(rospy.get_param("~occupancy_threshold", 90))

        # Helper variable to determine if grid was received at least once
        self.gridReady = False

        # You may wish to listen to the transformations of the robot
        self.tfBuffer = tf2_ros.Buffer()
        # Use the tfBuffer to obtain transformation as needed
        tfListener = tf2_ros.TransformListener(self.tfBuffer)

        # Subscribe to grid
        self.gridSubscriber = rospy.Subscriber('occupancy', OccupancyGrid, self.grid_cb)

        # Publish detected frontiers
        self.vis_pub = rospy.Publisher('frontier_vis', MarkerArray, queue_size=2)
        
        # Publish visibility grid
        self.vis_map_pub = rospy.Publisher('frontier_map', OccupancyGrid, queue_size=2)

        # TODO: you may wish to do initialization of other variables

    def publishFrontiersVis(self, frontiers):
        
        # TODO: Modify according to your frontier representation

        marker_array = MarkerArray()
        marker = Marker()
        marker.action = marker.DELETEALL
        marker.header.frame_id = self.mapFrame
        marker_array.markers.append(marker)
        self.vis_pub.publish(marker_array)

        marker_array = MarkerArray()
        marker_id = 0
        for f in frontiers:
            for i in range(len(f)):
                marker = Marker()
                marker.ns = 'frontier'
                marker.id = marker_id
                marker_id += 1
                marker.header.frame_id = self.mapFrame
                # marker.frame_locked = True
                marker.type = visualization_msgs.msg.Marker.CUBE
                marker.action = 0
                marker.scale.x = self.resolution
                marker.scale.y = self.resolution
                marker.scale.z = self.resolution
                x, y = gridToMapCoordinates(np.array([f[i][1], f[i][0]]), self.gridInfo)
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = 0
                marker.pose.orientation.x = 0
                marker.pose.orientation.y = 0
                marker.pose.orientation.z = 0
                marker.pose.orientation.w = 1
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker_array.markers.append(marker)
        self.vis_pub.publish(marker_array)

    def computeWFD(self):
        """ Run the Wavefront detector """

        frontiers = []

        # TODO: First, you should try to obtain the robots coordinates from frame transformation

        # TODO: Then, copy the occupancy grid into some temporary variable and inflate the obstacles

        # TODO: Run the WFD algorithm - see the presentation slides for details on how to implement it

        # TODO: Store each frontier as a list of coordinates in common list

        self.publishFrontiersVis(frontiers)

        return frontiers

    def getRandomFrontier(self, request):
        """ Return random frontier """
        # TODO
        frontiers = self.computeWFD()
        frontier = np.random.choice(frontiers)

        frontierCenter = 0  # TODO: compute center of the randomly drawn frontier here
        x, y = 0, 0  # TODO: transform the coordinates from grid to real-world coordinates (in meters)
        response = GenerateFrontierResponse(Pose2D(x, y, 0.0))
        return response

    def getClosestFrontier(self, request):
        """ Return frontier closest to the robot """
        # TODO
        frontiers = self.computeWFD()
        bestFrontierIdx = 0  # TODO: compute the index of the best frontier
        frontier = frontiers[bestFrontierIdx]

        frontierCenter = 0  # TODO: compute the center of the chosen frontier
        x, y = 0, 0  # TODO: compute the index of the best frontier
        response = GenerateFrontierResponse(Pose2D(x, y, 0.0))
        return response

    def getRobotCoordinates(self):
        """ Get the current robot position in the grid """
        try:
            trans = self.tfBuffer.lookup_transform(self.mapFrame, self.robotFrame, rospy.Time(), rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Cannot get the robot position!")
            self.robotPosition = None
        else:
            self.robotPosition = 0  # TODO: transform the robot coordinates from real-world (in meters) into grid

    def extractGrid(self, msg):
        # TODO: extract grid from msg.data and other usefull information
        pass

    def grid_cb(self, msg):
        self.extractGrid(msg)
        if not self.gridReady:
            # TODO: Do some initialization of necessary variables

            # Create services
            self.grf_service = rospy.Service('get_random_frontier', GenerateFrontier, self.getRandomFrontier)
            self.gcf_service = rospy.Service('get_closest_frontier', GenerateFrontier, self.getClosestFrontier)
            self.gridReady = True


if __name__ == "__main__":
    fe = FrontierExplorer()

    rospy.spin()
