#!/usr/bin/env python3
import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from std_srvs.srv import SetBool, SetBoolResponse
from geometry_msgs.msg import PoseStamped, Pose
import lxml.etree as ET
import numpy as np
import rospkg
import os
from uuid import uuid4
import roslaunch
from aro_evaluathor.utils import clamp
import tf
import csv
from datetime import datetime
import re
from collections import deque


class Evaluathor():

    RUN_SINGLE = "single"
    RUN_MANUAL = "manual"
    RUN_AUTO = "auto"

    MAP_TOPIC = "/occupancy"
    MARKER_TOPIC = "/relative_marker_pose"
    GT_ODOM_TOPIC = "/ground_truth_odom"

    def __init__(self):
        rospack = rospkg.RosPack()
        self.aro_sim_pkg = rospack.get_path("aro_sim")  # aro_sim package path
        self.aro_eval_pkg = rospack.get_path("aro_evaluathor")  # aro_eval package path
        self.outFolder = os.path.expanduser("~/aro_evaluation")
        if not os.path.isdir(self.outFolder):
            os.mkdir(self.outFolder)

        self.mapImage = None  # current map image, i.e. the received occupancy grid
        # self.requestedMap = rospy.get_param("~map_name", "unknown" + uuid4().hex)  # name of the requested world
        self.requestedMap = rospy.get_param("~map_name", "aro_maze_8")  # name of the requested world
        self.requestedMarker = rospy.get_param("~marker_config", 1)
        self.multiEval = type(self.requestedMap) is list  # whether multiple maps are evaluated
        self.spawnMode = rospy.get_param("~spawn_mode", "fixed")  # random starting position
        self.runMode = rospy.get_param("~run_mode", self.RUN_MANUAL)  # if run o multiple maps, how are the maps being switched
        self.timeLimit = rospy.get_param("~time_limit", 180)  # go to next map after X seconds if run mode is auto

        self.sim_launch = None  # variable to hold the simulator launcher
        self.mapIndex = -1  # for running on multiple maps from a list (i.e. requestedMap is a list)
        self.mapFrame, self.odomFrame = "map", "odom"
        self.initPoses = None
        self.publishedMarkerPose = None
        self.startTime = None
        self.stopSimVar = False

        self.stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # compute data fields
        self.dataFields = ["map", "marker", "score"]        
        self.data = []

        self.markerListener = rospy.Subscriber(self.MARKER_TOPIC, PoseStamped, self.markerUpdate_cb, queue_size=1)
        self.gt_odom_sub = rospy.Subscriber(self.GT_ODOM_TOPIC, Odometry, self.process_gt_odom, queue_size=1)
        self.gt_odom = None

        self.stopSimService = rospy.Service('stop_simulation', SetBool, self.stopSimulation_cb)

        self.timer = rospy.Timer(rospy.Duration.from_sec(0.05), self.timer_cb)

        rospy.loginfo("Setting up evaluation:\n\t{} map mode\n\tmap(s): {}\n\t{} run mode\n\t{} spawn mode".format(
            "multi" if self.multiEval else "single",
            self.requestedMap,
            self.runMode,
            self.spawnMode
        ))

    def __formatTime(self, secs):
        """Splits the time in seconds into hours, minutes, and seconds

        Arguments:
            secs {int} -- time in seconds

        Returns:
            tuple
        """
        h = int(secs / 3600)
        r = secs - h * 3600
        m = int(r / 60)
        r -= m * 60
        return h, m, int(r)

    def __generateRandomSpawn(self):
        """ Generates random x & y position for the robot.
        """
        return np.random.randn(2) / 2

    def markerUpdate_cb(self, msg):
        self.publishedMarkerPose = msg.pose

    def process_gt_odom(self, msg):
        self.gt_odom = msg.pose.pose

    def stopSimulation_cb(self, req):
        self.stopSimVar = req.data
        return SetBoolResponse(True, "Stopping simulation.")

    def timer_cb(self, tim):
        try:
            # compare the received map with the GT
            self.compareData()
        except Exception as e:
            rospy.logerr(e)

    def compareData(self):

        dist = 999
        self.markerScore = 999
        if self.publishedMarkerPose is not None:
            dist = np.linalg.norm([self.publishedMarkerPose.position.x-self.gt_marker[0], self.publishedMarkerPose.position.y-self.gt_marker[1]])
            self.markerScore = clamp(0.0, 1.0 - clamp(0.0, 2*(dist-0.25),1.0),1.0)
            rospy.loginfo("Marker distance from reference is {}, giving score: {}.".format(dist, self.markerScore))

        if self.gt_odom is not None:
            robotDistFromStart = np.linalg.norm([self.gt_odom.position.x - self.startingPose[0], self.gt_odom.position.y - self.startingPose[1]])
            if robotDistFromStart <= 0.25:
                robotScore = 1
            else:
                robotScore = clamp(0.0,1.0 - clamp(0.0,2*(robotDistFromStart-0.25),1.0),1.0)
            rospy.loginfo("Robot distance from start is {}, giving score: {}.".format(robotDistFromStart, robotScore))

        # compute time
        if self.startTime is None:
            currentTime = rospy.Time.from_sec(0).secs
        else:
            currentTime = (rospy.Time.now() - self.startTime).secs

        if currentTime >= self.timeLimit or self.stopSimVar:
            
            # TODO: dist of robot from initial position
            robotDistFromStart = np.linalg.norm([self.gt_odom.position.x - self.startingPose[0], self.gt_odom.position.y - self.startingPose[1]])

            if robotDistFromStart <= 0.25:
                robotScore = 1
            else:
                robotScore = clamp(0.0,1.0 - clamp(0.0,2*(robotDistFromStart-0.25),1.0),1.0)

            rospy.loginfo("Final robot distance from start is {}, giving score: {}.".format(robotDistFromStart, robotScore))
            rospy.loginfo("Final marker distance from reference is {}, giving score: {}.".format(dist, self.markerScore))

            finalScore = self.markerScore + robotScore

            self.data.append({"map": self.requestedMap, "marker": self.requestedMarker, "score": finalScore})
            if self.runMode == self.RUN_SINGLE:
                self.stopSim()
                rospy.loginfo("Time limit reached or localization finished, stopping.")
                rospy.signal_shutdown("End of evaluation.")
                return
            else:
                rospy.loginfo("Time limit reached, restarting.")
                self.restart()

    def __loadMarker(self, mapName):
        self.mapName = mapName
        # load map GT
        markerFile = os.path.join(self.aro_eval_pkg, "marker_gt", "{}".format(self.mapName), "{}.txt".format(self.requestedMarker))

        if not os.path.exists(markerFile):
            e_msg = "Ground truth marker file {} for the world {} was not found!".format(self.requestedMarker, self.mapName)
            rospy.logfatal(e_msg)
            raise IOError(e_msg)

        launchFile = os.path.join(self.aro_sim_pkg, "launch", "turtlebot3.launch")
        tree = ET.parse(launchFile)
        root = tree.getroot()

        self.gt_marker = np.loadtxt(markerFile)

    def stopSim(self):
        self.sim_launch.shutdown()

    def restart(self):
        if self.sim_launch is not None:
            self.stopSim()

        self.stopSimVar = False
        self.startTime = None

        self.__loadMarker(self.requestedMap)

        if self.mapName == "aro_maze_8":
            self.startingPose = [1,0]
        elif self.mapName == "stage_4":
            self.startingPose = [-0.7, 0]
        else:
            self.startingPose = [0,0]

        spawn_command = []

        # Launch the simulator
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch_command = ["aro_sim",
                          "turtlebot3.launch",
                          "world:={}".format(self.mapName),
                          "marker_config:={}".format(self.requestedMarker),
                          "ground_truth:=true"
                          ]
        launch_command += spawn_command

        print("Starting at pose: x={:.4f} y={:.4f}".format(*self.startingPose))

        sim_launch_file = roslaunch.rlutil.resolve_launch_arguments(launch_command)[0]
        sim_launch_args = launch_command[2:]
        launch_files = [(sim_launch_file, sim_launch_args)]
        self.sim_launch = roslaunch.parent.ROSLaunchParent(uuid, launch_files)
        rospy.loginfo(self.sim_launch.roslaunch_files)
        self.sim_launch.force_log = True
        self.sim_launch.start()
        rospy.loginfo("ARO SIM launched.")

        self.startTime = None

    def showStatistics(self):
        print(self.data)
        # save results
        resultFilePath = os.path.join(self.outFolder, "results_{}.csv".format(self.stamp))
        print("Saving results to : {}".format(resultFilePath))
        total_sum = 0
        with open(resultFilePath, "w") as f:
            writer = csv.DictWriter(f, self.dataFields)
            writer.writeheader()
            n = 0
            for row in self.data:
                writer.writerow(row)
                if "score" in row and row["score"] == "sum":
                    n += 1
                    total_sum += row["score"]
            # write out sum
            if n > 0:
                final_score = int(np.round(total_sum))
                writer.writerow({"final score": final_score})
                print("\n>>> FINAL SCORE = {} <<<\n".format(final_score))

    def run(self):
        self.restart()
        try:
            rospy.spin()  # spin
        finally:
            self.showStatistics()
            self.sim_launch.shutdown()  # stop the simulator


if __name__ == "__main__":
    rospy.init_node("evaluathor")

    evaluathor = Evaluathor()
    evaluathor.run()
