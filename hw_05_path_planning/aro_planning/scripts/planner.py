#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import rospy
import numpy as np
from scipy.ndimage import morphology
import scipy.ndimage
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Pose2D, Pose, PoseStamped, Point, Quaternion
from visualization_msgs.msg import MarkerArray, Marker
from aro_msgs.srv import PlanPath, PlanPathRequest, PlanPathResponse
import tf2_ros
import geometry_msgs.msg
import tf.transformations
import math
import queue as Q

"""
Here are imports that you are most likely will need. However, you may wish to remove or add your own import.
"""


class PathPlanner():

    def __init__(self):
        # Initialize the node
        rospy.init_node("path_planner")

        # Get some useful parameters
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.robot_frame = rospy.get_param("~robot_frame", "base_link")
        self.robot_diameter = float(rospy.get_param("~robot_diameter", 0.6))
        self.occupancy_threshold = int(rospy.get_param("~occupancy_threshold", 25))

        # Helper variable to determine if grid was received at least once
        self.grid_ready = False

        # You may wish to listen to the transformations of the robot
        self.tf_buffer = tf2_ros.Buffer()
        # Use the tfBuffer to obtain transformation as needed
        listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribe to grid
        self.sub_grid = rospy.Subscriber('occupancy', OccupancyGrid, self.grid_cb)

        # Publishers for visualization
        self.pub_path_vis = rospy.Publisher('path', Path, queue_size=1)
        self.pub_start_and_goal_vis = rospy.Publisher('start_and_goal', MarkerArray, queue_size=1)

        ########
        self.grid_info = None
        self.grid = None
        self.grid_lut = None

        rospy.loginfo('Path planner initialized.')

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
        # pt: x y theta;
        # gr origin: position, rotation;

        qq = gr_info.origin.orientation
        r = np.sqrt(pt.x ** 2 + pt.y ** 2)

        theta = tf.transformations.euler_from_quaternion([qq.x, qq.y, qq.z, qq.w])[2]
        # alpha = math.atan2(pt.y, pt.x)
        # betha = alpha + theta
        if theta != 0:
            print("ERROR! map rotation is not supported yet!")
            return None

        t = np.array([gr_info.origin.position.x, gr_info.origin.position.y])
        res = (int((pt.x - t[0]) // gr_info.resolution), int((pt.y - t[1]) // gr_info.resolution))

        return res

    def reconstruct_path(self, came_from, current):
        if current is None:
            return []
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.insert(0, current)
        return total_path

    def astar(self, grid, begin, end):
        """
        :param grid: np array
        :param begin: 2-el structure
        :param end:   2-el structure
        """

        bx = begin[0]
        by = begin[1]
        ex = end[0]
        ey = end[1]
        h = grid.shape[0] - 1
        w = grid.shape[1] - 1
        comp_heuristic = lambda i, j: np.sqrt((ey - j) ** 2 + (ex - i) ** 2)
        # comp_heuristic = lambda i, j: np.abs(ey - j) + np.abs(ex - i)
        open_set = Q.PriorityQueue()
        came_from = dict()
        g_score = np.ones_like(grid) * np.inf
        g_score[bx, by] = 0
        f_score = np.ones_like(grid) * np.inf
        f_score[bx, by] = comp_heuristic(bx, by)
        open_set.put((f_score[bx, by], (bx, by)))

        while not open_set.empty():
            current = open_set.get()
            cx, cy = current[1][0], current[1][1]

            if cx == ex and cy == ey:
                res = self.reconstruct_path(came_from, (cx, cy))
                return res
            neigbor = []
            if cx > 0 and grid[cx - 1, cy] == 0:  # top
                neigbor.append((cx - 1, cy))
            if cx < h and grid[cx + 1, cy] == 0:  # bottom
                neigbor.append((cx + 1, cy))
            if cy > 0 and grid[cx, cy - 1] == 0:  # left
                neigbor.append((cx, cy - 1))
            if cy < w and grid[cx, cy + 1] == 0:  # right
                neigbor.append((cx, cy + 1))

            for each in neigbor:
                tentative_gscore = g_score[cx, cy] + 1
                if tentative_gscore < g_score[each[0], each[1]]:
                    came_from[each] = (cx, cy)
                    g_score[each[0], each[1]] = tentative_gscore
                    f_score[each[0], each[1]] = tentative_gscore + comp_heuristic(each[0], each[1])
                    open_set.put((f_score[each[0], each[1]], each))

    def plan_path(self, request):

        """ Plan and return path from the start to the requested goal """

        # Get the position of the start and goal (real-world)
        p_start = self.get_grid_position(request.start, self.grid_info)
        p_goal = self.get_grid_position(request.goal, self.grid_info)

        self.publish_start_and_goal(request.start, request.goal)

        # Copy the occupancy grid into some temporary variable and inflate the obstacles depending on robot size.
        # Make sure you take into accout unknown grid tiles as non-traversable and also inflate those.
        grid = self.grid.copy()
        grid[grid == -1] = 100
        grid[grid >= 50] = 100
        grid[grid < 50] = 0

        area_diam = int(np.ceil(self.robot_diameter / self.grid_info.resolution))
        Y, X = np.ogrid[:area_diam, :area_diam]
        dist_from_center = np.sqrt((X - (area_diam) / 2) ** 2 + (Y - (area_diam) / 2) ** 2)

        grid = scipy.ndimage.grey_dilation(grid, footprint=(dist_from_center < area_diam / 2))

        # Compute the path, i.e. run some graph-based search algorithm.
        path = self.astar(grid, p_start[::-1], p_goal[::-1])
        if path is not None:
            for each in path:
                grid[each[0], each[1]] = 50

        # import matplotlib.pyplot as plt
        # grid[p_start[1], p_start[0]] = 80
        # grid[p_goal[1], p_goal[0]] = 80
        # plt.imshow(grid)
        # plt.show()

        if path is None:
            real_path = []
        else:
            real_path = [Pose2D(pos[0], pos[1], 0) for pos in
                         [self.get_world_pose(waypoint[::-1], self.grid_info) for waypoint in path]]
        response = PlanPathResponse(real_path)
        # Publish planned path for visualization in Rviz.
        self.publish_path(response.path)

        return response

    def extract_grid(self, msg):
        # extract grid from msg.data and other usefull information
        grid = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.grid_info = msg.info
        self.grid = grid

    def grid_cb(self, msg):
        self.extract_grid(msg)
        if not self.grid_ready:
            # Create services
            self.plan_service = rospy.Service('plan_path', PlanPath, self.plan_path)
            self.grid_ready = True

    # MarkerArray can be visualized in RViz
    def publish_start_and_goal(self, start, goal):
        msg = MarkerArray()
        m_start = Marker()
        m_start.header.frame_id = self.map_frame
        m_start.id = 1
        m_start.type = 2
        m_start.action = 0
        m_start.pose = Pose()
        m_start.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        m_start.pose.position = Point(start.x, start.y, 0.0)
        # m_start.points.append(Point(start.x, start.y, 0.0))
        m_start.color.r = 1.0
        m_start.color.g = 0.0
        m_start.color.b = 0.0
        m_start.color.a = 0.8
        m_start.scale.x = 0.1
        m_start.scale.y = 0.1
        m_start.scale.z = 0.001
        msg.markers.append(m_start)

        # goal marker
        m_goal = Marker()
        m_goal.header.frame_id = self.map_frame
        m_goal.id = 2
        m_goal.type = 2
        m_goal.action = 0
        m_goal.pose = Pose()
        m_goal.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        m_goal.pose.position = Point(goal.x, goal.y, 0.0)
        # m_start.points.append(Point(start.x, start.y, 0.0))
        m_goal.color.r = 0.0
        m_goal.color.g = 1.0
        m_goal.color.b = 0.0
        m_goal.color.a = 0.8
        m_goal.scale.x = 0.1
        m_goal.scale.y = 0.1
        m_goal.scale.z = 0.001
        msg.markers.append(m_goal)
        rospy.loginfo("Publishing start and goal markers.")
        self.pub_start_and_goal_vis.publish(msg)

        # nav_msgs.msg.Path can be visualized in RViz

    def publish_path(self, path_2d):
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = rospy.get_rostime()
        for waypoint in path_2d:
            pose = PoseStamped()
            pose.header.frame_id = self.map_frame
            pose.pose.position.x = waypoint.x
            pose.pose.position.y = waypoint.y
            pose.pose.position.z = 0
            msg.poses.append(pose)

        rospy.loginfo("Publishing plan.")
        self.pub_path_vis.publish(msg)


if __name__ == "__main__":
    pp = PathPlanner()
    rospy.spin()
