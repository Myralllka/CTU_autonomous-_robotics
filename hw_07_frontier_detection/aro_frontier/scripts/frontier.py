#!/usr/bin/env python3
from __future__ import division, print_function
import rospy
import numpy as np
import tf.transformations
import queue as Q
from scipy.ndimage import morphology
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import PoseStamped
from aro_msgs.srv import GenerateFrontier, GenerateFrontierRequest, GenerateFrontierResponse
import tf2_ros
import scipy.ndimage
import geometry_msgs.msg
import copy

"""
Here are imports that you are most likely will need. However, you may wish to remove or add your own import.
"""


class FrontierExplorer():

    def __init__(self):
        # Initialize the node
        rospy.init_node("frontier_explorer")

        # Get some useful parameters
        self.frame_map = rospy.get_param("~map_frame", "map")
        self.frame_robot = rospy.get_param("~robot_frame", "base_footprint")
        self.robot_diameter = float(rospy.get_param("~robot_diameter", 0.8))
        self.threshold_occupancy = int(rospy.get_param("~occupancy_threshold", 90))

        # Helper variable to determine if grid was received at least once
        self.grid_ready = False

        # You may wish to listen to the transformations of the robot
        self.tfBuffer = tf2_ros.Buffer()
        # Use the tfBuffer to obtain transformation as needed
        tf_listener = tf2_ros.TransformListener(self.tfBuffer)

        # Subscribe to grid
        self.sub_grid = rospy.Subscriber('occupancy', OccupancyGrid, self.grid_cb)

        # Publish detected frontiers
        self.pub_vis = rospy.Publisher('frontier_vis', MarkerArray, queue_size=2)

        # Publish visibility grid
        self.pub_vis_map = rospy.Publisher('frontier_map', OccupancyGrid, queue_size=2)

        self.pub_goal = rospy.Publisher("goal", PoseStamped, queue_size=2)

        # you may wish to do initialization of other variables
        self.grid_info = None
        self.grid = None
        self.grid_lut = None
        self.grid_msg_header = None
        self.grf_service = None
        self.gcf_service = None

    @staticmethod
    def get_world_pose(pt, gr_info):
        """
        :param pt:      np array
        :param gr_info: grid info, contains map_load_time, resolution, width, height, origin (real-world pose of the cell (0, 0) in the map)
        """
        # pt: x y theta;
        # gr origin: position, rotation;
        t = np.array([gr_info.origin.position.x, gr_info.origin.position.y])
        res = np.asarray(pt) * gr_info.resolution
        res = res + t
        return res

    @staticmethod
    def get_grid_position(pt, gr_info):
        """
        :param pt:      geometry_msgs/Pose2D
        :param gr_info: grid info, contains map_load_time, resolution, width, height,
        origin (real-world pose of the cell (0, 0) in the map)
        :return tuple of coordinates in grid frame
        """
        try:
            t = np.array([gr_info.origin.position.x, gr_info.origin.position.y])
            res = (int((pt.x - t[0]) // gr_info.resolution), int((pt.y - t[1]) // gr_info.resolution))
        except:
            t = np.array([gr_info.origin.position.x, gr_info.origin.position.y])
            res = (int((pt[0] - t[0]) // gr_info.resolution), int((pt[1] - t[1]) // gr_info.resolution))

        return res

    def get_robot_coordinates(self):
        """ Get the current robot position in the grid """
        try:
            trans = self.tfBuffer.lookup_transform(self.frame_map, self.frame_robot, rospy.Time(),
                                                   rospy.Duration(0.5)).transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Cannot get the robot position!")
            return None
        else:
            # transform the robot coordinates from real-world (in meters) into grid
            resolution = self.grid_info.resolution
            t = np.array([self.grid_info.origin.position.x, self.grid_info.origin.position.y])
            pt = trans.translation
            return [int((pt.y - t[1]) // resolution), int((pt.x - t[0]) // resolution)]

    def extract_grid(self, msg):
        # extract grid from msg.data and other usefull information
        grid = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.grid_info = msg.info
        self.grid = grid
        self.grid_msg_header = msg.header

    def publish_frontiers_map(self, grid):
        msg = OccupancyGrid()
        msg.info = self.grid_info
        msg.header.frame_id = self.grid_msg_header.frame_id
        msg.header.stamp = rospy.Time.now()
        occ_grid = tuple(grid.reshape(-1))
        msg.data = occ_grid
        self.pub_vis_map.publish(msg)

    def publish_goal(self, x, y):
        msg = PoseStamped()
        msg.header.frame_id = self.frame_map
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.01
        msg.pose.orientation.x = 0
        msg.pose.orientation.y = 0
        msg.pose.orientation.z = 0
        msg.pose.orientation.w = 1
        self.pub_goal.publish(msg)

    def publish_frontiers_vis(self, frontiers):
        # Modify according to your frontier representation
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = marker.DELETEALL
        marker.header.frame_id = self.frame_map
        marker_array.markers.append(marker)
        self.pub_vis.publish(marker_array)

        marker_array = MarkerArray()
        marker_id = 0
        for f in frontiers:
            for i in range(len(f)):
                marker = Marker()
                marker.ns = 'frontier'
                marker.id = marker_id
                marker_id += 1
                marker.header.frame_id = self.frame_map
                # marker.frame_locked = True
                marker.type = Marker.CUBE
                marker.action = 0
                marker.scale.x = self.grid_info.resolution
                marker.scale.y = self.grid_info.resolution
                marker.scale.z = self.grid_info.resolution
                x, y = self.get_world_pose(np.array([f[i][1], f[i][0]]), self.grid_info)
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
        self.pub_vis.publish(marker_array)

    def wfd(self, l_rpose, l_grid, WALL=1, EMPTY=0, UNKNOWN=-1):
        def compute_neighbours_isfrontier(ll_grid, x, y, h, w, WALL, EMPTY, UNKNOWN):
            l_neighbours_tmp, l_neighbours = list(), list()
            l_is_front = False
            if x > 0:
                l_neighbours_tmp.append((x - 1, y))
                if ll_grid[x - 1, y] == UNKNOWN:
                    l_is_front = True
            if x < h:
                l_neighbours_tmp.append((x + 1, y))
                if ll_grid[x + 1, y] == UNKNOWN:
                    l_is_front = True
            if y > 0:
                l_neighbours_tmp.append((x, y - 1))
                if ll_grid[x, y - 1] == UNKNOWN:
                    l_is_front = True
            if y < w:
                l_neighbours_tmp.append((x, y + 1))
                if ll_grid[x, y + 1] == UNKNOWN:
                    l_is_front = True

            if x > 0 and y > 0:
                l_neighbours_tmp.append((x - 1, y - 1))
            if x < h and y < w:
                l_neighbours_tmp.append((x + 1, y + 1))
            if x > 0 and y < w:
                l_neighbours_tmp.append((x - 1, y + 1))
            if x < h and y > 0:
                l_neighbours_tmp.append((x + 1, y - 1))

            for n in l_neighbours_tmp:
                if ll_grid[n[0], n[1]] == EMPTY:
                    l_neighbours.append((n[0], n[1]))

            if ll_grid[x, y] != EMPTY:
                l_is_front = False
            return l_neighbours, l_is_front

        MARKED = 1
        UNMARKED = 1
        qm = Q.Queue()
        front_res = list()
        h, w = l_grid.shape[0] - 1, l_grid.shape[1] - 1
        qm.put(l_rpose)

        lut_frontier_open_list = np.zeros_like(l_grid)
        lut_map_open_list = np.zeros_like(l_grid)
        lut_map_close_list = np.zeros_like(l_grid)
        lut_frontier_close_list = np.zeros_like(l_grid)

        rx, ry = l_rpose[0], l_rpose[1]
        lut_map_open_list[rx, ry] = MARKED

        while not qm.empty():
            p = qm.get()
            if lut_map_close_list[p[0], p[1]] == MARKED:
                continue
            px, py = p[0], p[1]
            p_neighbours, pis_frontier = compute_neighbours_isfrontier(l_grid, px, py, h, w, WALL, EMPTY, UNKNOWN)

            if pis_frontier:  # frontier pt
                qf = Q.Queue()
                new_frontier = list()
                qf.put((px, py))
                lut_map_open_list[px, py] = MARKED
                while not qf.empty():
                    q = qf.get()
                    qx, qy = q[0], q[1]
                    if lut_map_close_list[qx, qy] == MARKED or lut_frontier_close_list[qx, qy] == MARKED:
                        continue

                    q_neighbours, qis_frontier = compute_neighbours_isfrontier(l_grid, qx, qy, h, w, WALL, EMPTY,
                                                                               UNKNOWN)

                    if qis_frontier:
                        new_frontier.append(q)
                        for n in q_neighbours:
                            if not (lut_frontier_open_list[n[0], n[1]] == MARKED or
                                    lut_frontier_close_list[n[0], n[1]] == MARKED or
                                    lut_map_close_list[n[0], n[1]] == 1):
                                qf.put(n)
                                lut_frontier_open_list[n[0], n[1]] = MARKED
                    lut_frontier_close_list[qx, qy] = MARKED
                front_res.append(new_frontier)
                for pt in new_frontier:
                    lut_map_close_list[pt[0], pt[1]] = MARKED
            for n in p_neighbours:
                if not (lut_map_open_list[n[0], n[1]] == MARKED or
                        lut_map_close_list[n[0], n[1]] == MARKED):
                    qm.put(n)
                    lut_map_open_list[n[0], n[1]] = MARKED
            lut_map_close_list[px, py] = MARKED
        return front_res

    def compute_WFD(self):
        """ Run the Wavefront detector """
        frontiers = []

        # First, you should try to obtain the robots coordinates from frame transformation
        robot_pose = self.get_robot_coordinates()
        if robot_pose is None:
            return frontiers
        # Then, copy the occupancy grid into some temporary variable and inflate the obstacles
        grid = self.grid.copy()
        grinfo = copy.deepcopy(self.grid_info)

        # import matplotlib.pyplot as plt
        # plt.imshow(grid)
        # plt.show()
        grid[grid >= self.threshold_occupancy] = 100
        grid[np.logical_and(grid > 0, grid < self.threshold_occupancy)] = 0

        area_diam = int(np.ceil(self.robot_diameter / self.grid_info.resolution))
        Y, X = np.ogrid[:area_diam, :area_diam]
        dist_from_center = np.sqrt((X - (area_diam - 1) / 2) ** 2 + (Y - (area_diam - 1) / 2) ** 2)

        grid[grid == -1] = 50
        grid = scipy.ndimage.grey_dilation(grid, footprint=(dist_from_center < area_diam / 2))
        self.publish_frontiers_map(grid)

        # Run the WFD algorithm - see the presentation slides for details on how to implement it
        frontiers_px = self.wfd(robot_pose, grid, UNKNOWN=50)

        # Store each frontier as a list of coordinates in common list
        self.publish_frontiers_vis(frontiers_px)

        return frontiers_px

    def get_random_frontier(self, request):
        """ Return random frontier """
        #
        frontiers = self.compute_WFD()
        if len(frontiers) == 0:
            return None
        frontier = np.random.choice(frontiers)
        # compute center of the randomly drawn frontier here
        frontier_center = np.mean(frontier, axis=0)

        # transform the coordinates from grid to real-world coordinates (in meters)
        x, y = self.get_world_pose([frontier_center[1], frontier_center[0]], self.grid_info)
        response = GenerateFrontierResponse(Pose2D(x, y, 0.0))
        self.publish_goal(x, y)
        return response

    def get_closest_frontier(self, request):
        """ Return frontier closest to the robot """
        #
        frontiers = self.compute_WFD()
        # compute the index of the best frontier

        dists = list()
        coors = list()
        x, y = self.get_robot_coordinates()
        x, y = self.get_world_pose([y, x], self.grid_info)
        for i in range(len(frontiers)):
            tx, ty = np.mean(frontiers[i], axis=0)
            tx, ty = self.get_world_pose([ty, tx], self.grid_info)
            dists.append(np.sqrt((x - tx) ** 2 + (y - ty) ** 2))
            coors.append((tx, ty))
            # print(f"distance from ({x}, {y}) to ({coors[-1]}) is exactly {dists[-1]=}, sum is {np.sqrt((x - tx) ** 2 + (y - ty) ** 2)}", )
        best_frontier_idx = np.argmin(np.array(dists))
        # compute the center of the chosen frontier
        frontier_center = coors[best_frontier_idx]
        # compute the index of the best frontier
        x, y = frontier_center[0], frontier_center[1]
        response = GenerateFrontierResponse(Pose2D(x, y, 0.0))
        self.publish_goal(x, y)
        return response

    def grid_cb(self, msg):
        self.extract_grid(msg)
        if not self.grid_ready:
            # Do some initialization of necessary variables
            # Create services
            self.grf_service = rospy.Service('get_random_frontier', GenerateFrontier, self.get_random_frontier)
            self.gcf_service = rospy.Service('get_closest_frontier', GenerateFrontier, self.get_closest_frontier)
            self.grid_ready = True


if __name__ == "__main__":
    fe = FrontierExplorer()
    rospy.spin()
