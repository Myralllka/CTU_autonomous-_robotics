from __future__ import print_function

import copy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
if not hasattr(time, "perf_counter"):
    time.perf_counter = time.clock
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.sparse import vstack

try:
    from utils import coords_to_transform_matrix, find_nearest_index_for_new_measurement, \
        merge_dicts, sec_to_nsec, transform_matrix_to_coords, nsec_to_sec
except (ValueError, ModuleNotFoundError):
    from aro_localization.utils import coords_to_transform_matrix, find_nearest_index_for_new_measurement, \
        merge_dicts, sec_to_nsec, transform_matrix_to_coords, nsec_to_sec


class FactorGraph(object):
    """Factorgraph implementation for 2D localization using relative motion measurements, absolute and relative
    markers.

    In description of dimensions of various arrays, we use N to denote the number of measurements that have been added
    to the factorgraph (i.e. the number of successful calls to self.add_z()). The estimated trajectory (self.x) is one
    element longer (it includes a starting pose), os its length is N+1.
    """

    def __init__(self, ma=(12.0, 40.0, 0.0), mr_gt=None, time_step=sec_to_nsec(0.2), fuse_icp=False,
                 solver_options=None, icp_yaw_scale=1.0, marker_yaw_scale=1.0):
        """
        :param list ma: Pose of the absolute marker in the world frame (3-tuple x, y, yaw).
        :param mr_gt: Ground truth pose of the relative marker in the world frame (3-tuple x, y, yaw).
        :type mr_gt: list or None
        :param int time_step: Approximate time step between two time instants in the factor graph (in nanoseconds).
        :param bool fuse_icp: Whether to fuse ICP SLAM odometry. Set to False for HW 03.
        :param solver_options: Options for the non-linear least squares solver. They will override the defaults.
        :type solver_options: dict or None
        :param float icp_yaw_scale: Additional scale applied to ICP odometry yaw residuals.
        :param float marker_yaw_scale: Additional scale applied to marker yaw residuals.
        """
        # Trajectory estimate
        self.x = [[0.0, 0.0, 0.0]]
        """The best estimate of the robot trajectory in world coordinates. List (N+1)x3."""
        self.x_odom = copy.copy(self.x)
        """Trajectory of the robot constructed just by integrating the input odom measurements. Should be very similar
        to odom frame pose. List (N+1)x3."""
        self.x_icp_odom = copy.copy(self.x)
        """Trajectory of the robot constructed just by integrating the input ICP odom measurements. Should be very
        similar to ICP odom frame pose. List (N+1)x3."""
        self.last_optimized_idx = 0
        """The last index into self.x that has been optimized. All values after this index are just initial guesses."""

        # Absolute marker
        self.ma = np.array([[ma[0]], [ma[1]], [ma[2]]])
        """Pose of the absolute marker (should be given as task input). Numpy array 3x1."""
        self.z_ma = [[np.nan, np.nan, np.nan]]
        """Measurements of the absolute marker pose (in current robot frame). This array should have the same number
        of elements as self.x. Each time instant where no measurement is available should contain NaNs. List (N+1)x3."""
        self.c_ma = [0.0]
        """Costs for the absolute marker pose residuals. List N+1."""
        self.seen_ma = False
        """Whether the absolute marker has already been seen."""

        # Relative marker
        self.mr = np.nan * np.zeros_like(self.ma)  # Initialization with NaNs
        """Current best estimate of the relative marker pose. Numpy array 3x1."""
        self.mr_gt = None if mr_gt is None else np.array([[mr_gt[0]], [mr_gt[1]], [mr_gt[2]]])
        """Ground truth pose of the relative marker (can be given as task input). Numpy array 3x1."""
        self.z_mr = [[np.nan, np.nan, np.nan]]
        """Measurements of the relative marker pose (in current robot frame). This array should have the same number
        of elements as self.x. Each time instant where no measurement is available should contain NaNs. List (N+1)x3."""
        self.c_mr = [0.0]
        """Costs for the relative marker pose residuals. List N+1."""
        self.seen_mr = False
        """Whether the relative marker has already been seen."""
        self.marker_yaw_scale = marker_yaw_scale

        # Odom
        self.z_odom = []
        """Odometry measurements (relative motion in robot body frame). Each time instant contains a valid (non-NaN)
        measurement. Measurement at index t moves the robot from x[t] to x[t+1]. List Nx3."""
        self.c_odom = []
        """Costs for odometry motion residuals. List N."""
        self.t_odom = []
        """Timestamps of odometry measurements (int, nanoseconds). List N."""
        self.time_step = time_step
        """Approximate time step between two time instants in the factor graph (in nanoseconds)."""

        # ICP
        self.fuse_icp = fuse_icp
        """Whether to add ICP odometry measurements to the graph."""
        self.z_icp = []
        """Relative motion estimated from ICP (expressed in robot body frame). This array should have the same number
        of elements as self.z_odom. Each time instant where no measurement is available should contain NaNs. Measurement
        at index t moves the robot from x[t] to x[next t with non-NaN value in z_icp].  List Nx3."""
        self.c_icp = []
        """Costs for ICP motion residuals. List N."""
        self.icp_yaw_scale = icp_yaw_scale
        # helper variables for turning incoming absolute ICP odometry into relative
        self.icp_poses = [np.eye(4)]
        self.icp_ts = [0]
        self.icp_cs = [0.0]

        # Ground truth odometry
        self.z_gt_odom = []
        """Ground truth odometry measurements (absolute poses in world frame). Each time instant where no measurement is
        available should contain NaNs. List Nx3."""

        # Solver
        # ARO homework 3: test with various combinations of loss and f_scale, select the best
        # (loss, f_scale) tested configs: ('linear', 1.0), ('soft_l1', 0.05), ('huber', 1.0), ('cauchy', 1.9)
        self.solver_options = {
            "method": 'trf',
            "x_scale": 1,
            "tr_solver": 'lsmr',
            "tr_options": {
                'maxiter': 10,
                'atol': 1e-4,
                'btol': 1e-4
            },
            #"loss": "linear",
            #"f_scale": 1.0,
            "loss": "soft_l1",
            "f_scale": 0.05,
            #"loss": "huber",
            #"f_scale": 1.0,
            #"loss": "cauchy",
            #"f_scale": 1.9,
            "max_nfev": 40,
            "verbose": 1,
            "ftol": 1e-3,
            "gtol": 1e-8,
            "xtol": 1e-4,
        }
        if solver_options is not None:
            merge_dicts(self.solver_options, solver_options)
        print("Solver options are: " + str(self.solver_options))

        self._figure_shown = False

    def add_z(self, z_odom, c_odom, z_mr, c_mr, z_ma, c_ma, z_icp, c_icp, z_gt_odom,
              t_odom=None, t_markers=None, t_icp=None):
        """Add measurements to the factorgraph.

        :param tuple z_odom: Odometry relative motion measurement. Should not contain NaNs. List 3.
        :param float c_odom: Cost of the odometry relative motion residual.
        :param tuple z_mr: Measurement of the relative marker pose in robot body frame (NaNs if not observed). List 3.
        :param float c_mr: Cost of the relative marker pose measurement residual.
        :param tuple z_ma: Measurement of the absolute marker pose in robot body frame (NaNs if not observed). List 3.
        :param float c_ma: Cost of the absolute marker pose measurement residual.
        :param list z_icp: ICP poses in ICP frame observed since last measurement update. List of 4x4 numpy arrays.
        :param list c_icp: Costs of the ICP measurement residual.
        :param tuple z_gt_odom: Measurement of ground truth absolute pose in world frame (NaNs if not observed). List 3.
        :param t_odom: Timestamp of the odometry measurement in nanoseconds.
        :type t_odom: int or None
        :param t_markers: Timestamp of the marker observations in nanoseconds.
        :type t_markers: int or None
        :param t_icp: Timestamps of the ICP measurements in nanoseconds.
        :type t_icp: list or None
        """
        if not self.seen_ma and np.all(np.isnan(z_ma)):
            print("Waiting for absolute marker", file=sys.stderr)
            return
        elif not self.seen_ma:
            # Initialize self.x[0] according to the first measurement of absolute marker. This aligns the coordinate
            # frame of the factorgraph with the coordinate frame of the world.
            ma_in_body = coords_to_transform_matrix(z_ma)
            ma_in_world = coords_to_transform_matrix(self.ma)
            body_in_ma = np.linalg.inv(ma_in_body)
            body_in_world = np.matmul(ma_in_world, body_in_ma)
            self.x[0] = transform_matrix_to_coords(body_in_world)
            self.x_odom[0] = copy.copy(self.x[0])
            self.x_icp_odom[0] = copy.copy(self.x[0])
            self.icp_poses[0] = body_in_world
            self.icp_ts[0] = t_odom
            self.seen_ma = True

        # Add odometry relative motion measurement
        self.z_odom.append(z_odom)
        self.c_odom.append(c_odom)
        self.t_odom.append(t_odom)
        timestamps = copy.copy(self.t_odom)

        # Add relative and absolute marker measurements
        self.z_ma.append(np.nan * np.zeros_like(z_ma))
        self.c_ma.append(c_ma)
        self.z_mr.append(np.nan * np.zeros_like(z_mr))
        self.c_mr.append(c_mr)

        markers_i = find_nearest_index_for_new_measurement(timestamps, t_markers)
        if markers_i is not None:
            if not np.isnan(z_ma[0]):
                self.z_ma[markers_i] = z_ma

            if not np.isnan(z_mr[0]):
                self.z_mr[markers_i] = z_mr
                self.seen_mr = True

        # Add ICP odometry measurement (add it even if self.fuse_icp is False - e.g. for visualization)
        self.z_icp.append(np.nan * np.zeros_like(z_odom))
        self.c_icp.append(0.0)
        self.icp_poses.append(None)
        self.icp_ts.append(None)
        self.icp_cs.append(None)
        self.x_icp_odom.append([np.nan, np.nan, np.nan])

        # Integrate ICP odometry into the factor graph. This code is a bit complicated because the odometry comes as
        # absolute measurements and we want relative ones - and it is not guaranteed that we will receive some odometry
        # for every time step. Moreover, it comes with a slight time delay so we do not necessarily get a measurement
        # for the latest time instant. So we want to differentiate the absolute pose against a previous one, but it is
        # not guaranteed that the one at index-1 is present. It can also happen that we will receive an older
        # measurement later and we will need to recompute the differences.
        #
        # The basic idea is to create a "pool" of absolute pose measurements and their timestamps and select the
        # best measurement from this pool for each time instant in the graph. Once the measurements are selected,
        # we compute the differences between them to get the relative odometry.

        if t_icp is not None and len(t_icp) > 0:  # If there are some ICP odometry measurements
            # Figure out from which time instant we need to update the graph
            min_t_icp = min(t_icp)
            min_icp_idx = find_nearest_index_for_new_measurement(timestamps, min_t_icp)
            # If there already is a measurement at min_icp_idx with earlier stamp, add it to the update pool
            if self.t_odom[min_icp_idx] < min_t_icp:
                min_icp_idx = max(min_icp_idx - 1, 0)

            # If there already is a measurement at a time instant, add it to the update pool so that it can be changed
            for idx in range(min_icp_idx, len(self.t_odom)):
                if self.icp_poses[idx] is not None:
                    z_icp += [self.icp_poses[idx]]
                    t_icp += [self.icp_ts[idx]]
                    c_icp += [self.icp_cs[idx]]

            # Find the best measurements from the update pool for each time instant in the graph
            used_idxs = set()
            for idx in range(min_icp_idx, len(self.t_odom)):
                t = self.t_odom[idx]
                # Compute time differences of all unassigned measurements to the selected time instant.
                # Used measurements have +inf so that they are never selected by the minimization.
                t_diffs = [abs(t - t_icp[i]) if i not in used_idxs else float('inf') for i in range(len(t_icp))]
                closest_value = min(t_diffs)
                if closest_value > self.time_step * 4 / 3:  # If the closest value is too far, do not assign it
                    continue
                closest_idx = np.argmin(t_diffs)
                used_idxs.add(closest_idx)

                self.icp_poses[idx + 1] = z_icp[closest_idx]
                self.icp_ts[idx + 1] = t_icp[closest_idx]
                self.icp_cs[idx + 1] = c_icp[closest_idx]

            # With the most suitable measurements assigned, compute the ICP pose differences to get the relative
            # odometry
            for idx in range(min_icp_idx, len(self.t_odom)):
                prev_pose = self.icp_poses[idx]
                if prev_pose is None:
                    continue
                gap = 0  # Count the number of "gaps" (time instants) between the current and previous pose
                for i in range(idx, len(self.t_odom)):
                    gap += 1
                    if self.icp_poses[i + 1] is not None:  # Found a measurement, compute the difference
                        new_pose = self.icp_poses[i + 1]
                        pose_diff = np.matmul(np.linalg.inv(prev_pose), new_pose)
                        # Compute the pose difference divided by the size of the gap. This gives the relative pose
                        # change per time instant (note that this simple division only works for differences in yaw
                        # smaller than pi).
                        dx, dy, dw = np.array(transform_matrix_to_coords(pose_diff)) / gap
                        # Fill all the gaps with the computed pose difference
                        for t in range(idx, i + 1):
                            self.z_icp[t] = (dx, dy, dw)
                            self.c_icp[t] = self.icp_cs[idx + 1]

                            # Update x_icp_odom. The first few values except [0] may be NaN, but all other should be
                            # filled. So if we encounter NaN in the previous value, we substitute it with the absolute
                            # measurement.
                            if np.any(np.isnan(self.x_icp_odom[t])):
                                self.x_icp_odom[t] = transform_matrix_to_coords(
                                    # transform the ICP pose to a frame relative to the first measurement in world frame
                                    np.matmul(self.icp_poses[0], self.icp_poses[t]))
                            self.x_icp_odom[t + 1] =\
                                self.get_world_pose(self.z_icp[t], self.x_icp_odom[t]).ravel().tolist()
                        break

        self.z_gt_odom.append(z_gt_odom)

        # Update self.x_odom by integrating odometry velocity
        self.x_odom.append(self.get_world_pose(z_odom, self.x_odom[-1]).ravel().tolist())

        # Initialize new x with NaNs. It will be initialized from the best odometry source right before optimization.
        self.x.append([np.nan] * 3)

    def optimize(self, x, mr, ma, z_odom, z_mr, z_ma, z_icp, c_odom, c_mr, c_ma, c_icp):
        """Optimize the factor graph given by the parameters.

        :param np.ndarray x: Robot trajectory estimate. Numpy array Nx3.
        :param np.ndarray mr: Relative marker pose estimate. Numpy array 3x1.
        :param np.ndarray ma: Absolute marker pose. Numpy array 3x1.
        :param np.ndarray z_odom: Odometry measurements (relative motion in robot body frame). Numpy array 3xN.
        :param np.ndarray z_mr: Relative marker pose measurements (in robot body frame). Numpy array 3xN.
        :param np.ndarray z_ma: Absolute marker pose measurements (in robot body frame). Numpy array 3xN.
        :param np.ndarray z_icp: ICP odometry measurements (relative motion in robot body frame). Numpy array 3xN.
        :param np.ndarray c_odom: Costs of odometry residuals. Numpy array N.
        :param np.ndarray c_mr: Costs of relative marker pose estimate residuals. Numpy array N.
        :param np.ndarray c_ma: Costs of absolute marker pose estimate residuals. Numpy array N.
        :param np.ndarray c_icp: Costs of ICP odometry residuals. Numpy array N.
        :return: Optimized `x` (robot trajectory estimate, list) and `mr` (relative marker pose estimate,
                 3x1 Numpy array).
        :rtype: tuple
        """
        idx_ma = np.where((~np.isnan(z_ma)).all(axis=0))[0]
        idx_mr = np.where((~np.isnan(z_mr)).all(axis=0))[0]
        idx_icp = np.where((~np.isnan(z_icp)).all(axis=0))[0]

        # Initialize the not yet optimized part of the trajectory by either ICP or odom measurements
        for i in range(x.shape[0] - 1):
            if i >= self.last_optimized_idx - 1 or np.any(np.isnan(x[i + 1, :])):
                # Select the best available odometry source
                z = z_icp[:, i] if i in idx_icp and c_icp[i] != 0 else z_odom[:, i]
                x[i + 1, :] = self.get_world_pose(z, x[i, :])[:, 0]

        mr_is_nan = np.any(np.isnan(mr))
        if mr_is_nan:
            # If this is the first time the relative marker has been seen, set its pose estimate to the observed one
            if idx_mr.shape[0] > 0:
                mr = self.get_world_pose(z_mr[:, idx_mr[0]], x[idx_mr[0]])
            # If it has not been seen yet, set it to zeros because the optimizer refuses to work with NaNs
            else:
                mr = np.zeros_like(mr)

        x_ini = np.hstack((x.reshape(-1), mr.reshape(-1)))   # concatenation [x, mr]

        tic = time.perf_counter()
        print("x %s, mr %s, ma %s" % (str(x[-1]), str(np.ravel(mr)), str(np.ravel(self.ma))))

        sol = least_squares(self.compute_residuals_opt, x_ini,
                            self.compute_residuals_jacobian_opt,
                            args=(ma, z_odom, z_mr, z_ma, z_icp, idx_mr, idx_ma, idx_icp, c_odom, c_mr, c_ma, c_icp),
                            **self.solver_options)

        # "un-concatenate" x, mr
        x, mr = sol.x[0:-3].reshape(x.shape[0], x.shape[1]), sol.x[-3:].reshape(3, 1)
        x[:, 2] = np.mod(x[:, 2] + np.pi, 2 * np.pi) - np.pi  # Wrap the computed yaw between -pi and pi
        mr[2, :] = np.mod(mr[2, :] + np.pi, 2 * np.pi) - np.pi  # Wrap the computed marker yaw between -pi and pi

        # If mr has not been seen yet, turn the result back to NaN
        if mr_is_nan and idx_mr.shape[0] == 0:
            mr *= np.nan

        self.last_optimized_idx = x.shape[0] - 1

        print("time %0.4f seconds, size %i, x %s, mr %s, ma %s" % (
            time.perf_counter() - tic, self.last_optimized_idx, str(x[-1]), str(np.ravel(mr)), str(np.ravel(self.ma))))

        return x.tolist(), mr

    @staticmethod
    def get_world_pose(z, x):
        """Get world pose of the given measurements in robot body frame (or apply relative odometry).

        :param z: The measurements or odometry in robot body frame. List or numpy array 3 or 3xN.
        :type z: list or np.ndarray
        :param x: Robot world pose estimate in times corresponding to each measurement.
                  List or numpy array 3 or 3xN or 3x(N+1) (last column ignored).
        :type x: list or np.ndarray
        :return: World poses of the measurements computed using the robot pose estimates. Numpy array 3xN.
        :rtype: np.ndarray
        """
        # convenience for passing lists
        x = np.array(x)
        z = np.array(z)

        # convenience to passing single measurements as 1D arrays
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        if len(z.shape) == 1:
            z = z[:, np.newaxis]

        # convenience for passing self.x which has one element more on the end
        if x.shape[1] == z.shape[1] + 1:
            x = x[:, :-1]

        x_t = np.zeros(x.shape)
        for t in range(x.shape[1]):
            R = np.array([[np.cos(x[2, t]), -np.sin(x[2, t])], [np.sin(x[2, t]), np.cos(x[2, t])]])  # Rotation matrix
            x_t[0:2, t] = np.matmul(R, z[0:2, t]) + x[0:2, t]
            x_t[2, t] = np.mod(z[2, t] + x[2, t] + np.pi, 2 * np.pi) - np.pi  # Wrap yaw between -pi and pi
        return x_t

    @staticmethod
    def compute_pose_residual(observed_x, x):
        """Compute the residuals of the given poses.

        :param np.ndarray observed_x: Observations (Numpy array 3x(N+1)).
        :param np.ndarray x: Estimates (Numpy array 3x(N+1)).
        :return: The residuals (Numpy array 3x(N+1)).
        :rtype: np.ndarray
        """
        # ARO homework 3: compute pose residuals
        res = observed_x - x

        # After computing the residuals, make sure that for yaw, we compute the shortest angular diff
        res[2, :] = np.mod(res[2, :] + np.pi, 2 * np.pi) - np.pi
        res[2, res[2, :] < -np.pi] = res[2, res[2, :] < -np.pi] + 2 * np.pi
        res[2, res[2, :] > np.pi] = res[2, res[2, :] > np.pi] - 2 * np.pi
        return res

    def compute_residuals(self, x, mr, ma, z_odom, z_mr, z_ma, z_icp, idx_mr, idx_ma, idx_icp, c_odom, c_mr, c_ma, c_icp):
        """Compute residuals (errors) of all measurements.

        :param np.ndarray x: Fused robot poses. Numpy array 3x(N+1).
        :param np.ndarray mr: Estimated relative marker pose. Numpy array 3x1.
        :param np.ndarray ma: Absolute marker pose. Numpy array 3x1.
        :param np.ndarray z_odom: Odometry measurements (relative motion in robot body frame). Numpy array 3xN.
        :param np.ndarray z_mr: Relative marker pose measurements (in robot body frame). Numpy array 3xN.
        :param np.ndarray z_ma: Absolute marker pose measurements (in robot body frame). Numpy array 3xN.
        :param np.ndarray z_icp: ICP odometry measurements (relative motion in robot body frame). Numpy array 3xN.
        :param np.ndarray idx_mr: Indices of z_mr with valid (non-NaN) measurements. Numpy array <N.
        :param np.ndarray idx_ma: Indices of z_ma with valid (non-NaN) measurements. Numpy array <N.
        :param np.ndarray idx_icp: Indices of z_icp with valid (non-NaN) measurements. Numpy array <N.
        :param np.ndarray c_odom: Costs of odometry residuals. Numpy array N.
        :param np.ndarray c_mr: Costs of relative marker pose estimate residuals. Numpy array N.
        :param np.ndarray c_ma: Costs of absolute marker pose estimate residuals. Numpy array N.
        :param np.ndarray c_icp: Costs of ICP odometry residuals. Numpy array N.
        :return: The residuals of odometry, relative marker, absolute marker and (possibly) ICP.
                 Tuple of numpy arrays Qx3, Q <= N.
        :rtype: tuple
        """
        # Odom measurements residuals
        observed_x = self.get_world_pose(z_odom, x)
        res_odom = self.compute_pose_residual(observed_x, x[:, 1:])
        res_odom = np.repeat(np.atleast_2d(c_odom), res_odom.shape[0], axis=0) * res_odom  # apply cost

        # Relative marker measurements residuals
        observed_mr = self.get_world_pose(z_mr[:, idx_mr], x[:, idx_mr])
        res_mr = self.compute_pose_residual(observed_mr, mr)
        # res_mr[np.abs(res_mr) < 0.1] = 0  # HACK
        res_mr = np.repeat(np.atleast_2d(c_mr)[:, idx_mr], res_mr.shape[0], axis=0) * res_mr  # apply cost
        res_mr[2, :] *= self.marker_yaw_scale

        # Absolute marker measurements residuals
        observed_ma = self.get_world_pose(z_ma[:, idx_ma], x[:, idx_ma])
        res_ma = self.compute_pose_residual(observed_ma, ma)
        # res_ma[np.abs(res_ma) < 0.1] = 0  # HACK
        res_ma = np.repeat(np.atleast_2d(c_ma)[:, idx_ma], res_ma.shape[0], axis=0) * res_ma  # apply cost
        res_ma[2, :] *= self.marker_yaw_scale

        # TODO; after homework 4: integrate ICP odometry
        res_icp = None
        if self.fuse_icp:
            print("="*15)
            observed_icp = self.get_world_pose(z_icp[:, idx_icp], x[:, idx_icp])
            res_icp = self.compute_pose_residual(observed_icp, x[:, idx_icp])
            res_icp = np.repeat(np.atleast_2d(c_icp)[:, idx_icp], res_icp.shape[0], axis=0) * res_icp  # apply cost
            print("="*15)

        return res_odom, res_mr, res_ma, res_icp

    def compute_residuals_opt(self, xmr, ma, z_odom, z_mr, z_ma, z_icp, idx_mr, idx_ma, idx_icp, c_odom, c_mr, c_ma, c_icp):
        """Like compute_residuals, but adapted shapes to work with the optimizer."""
        # unpack x, mr from the concatenated state
        x, mr = xmr[0:-3].reshape(3, int((xmr.shape[0] - 3) / 3), order='F'), xmr[-3:].reshape(3, 1)

        res_odom, res_mr, res_ma, res_icp = self.compute_residuals(
            x, mr, ma, z_odom, z_mr, z_ma, z_icp, idx_mr, idx_ma, idx_icp, c_odom, c_mr, c_ma, c_icp)

        # Concatenate and linearize the residuals
        if not self.fuse_icp:
            return np.hstack([
                res_odom.reshape(-1, order='F'),
                res_mr.reshape(-1, order='F'),
                res_ma.reshape(-1, order='F'),
            ])

        return np.hstack([
            res_odom.reshape(-1, order='F'),
            res_mr.reshape(-1, order='F'),
            res_ma.reshape(-1, order='F'),
            res_icp.reshape(-1, order='F'),
        ])

    def compute_residuals_jacobian(
            self, x, mr, ma, z_odom, z_mr, z_ma, z_icp, idx_mr, idx_ma, idx_icp, c_odom, c_mr, c_ma, c_icp):
        """Compute Jacobian of the residuals. This is used to steer the optimization in a good direction.

        :param np.ndarray x: Fused robot poses. Numpy array 3x(N+1).
        :param np.ndarray mr: Estimated relative marker pose. Numpy array 3x1.
        :param np.ndarray ma: Absolute marker pose. Numpy array 3x1.
        :param np.ndarray z_odom: Odometry measurements (relative motion in robot body frame). Numpy array 3xN.
        :param np.ndarray z_mr: Relative marker pose measurements (in robot body frame). Numpy array 3xN.
        :param np.ndarray z_ma: Absolute marker pose measurements (in robot body frame). Numpy array 3xN.
        :param np.ndarray z_icp: ICP odometry measurements (relative motion in robot body frame). Numpy array 3xN.
        :param np.ndarray idx_mr: Indices of z_mr with valid (non-NaN) measurements. Numpy array N.
        :param np.ndarray idx_ma: Indices of z_ma with valid (non-NaN) measurements. Numpy array N.
        :param np.ndarray idx_icp: Indices of z_icp with valid (non-NaN) measurements. Numpy array N.
        :param np.ndarray c_odom: Costs of odometry residuals. Numpy array N.
        :param np.ndarray c_mr: Costs of relative marker pose estimate residuals. Numpy array N.
        :param np.ndarray c_ma: Costs of absolute marker pose estimate residuals. Numpy array N.
        :param np.ndarray c_icp: Costs of ICP odometry residuals. Numpy array N.
        :return: 4-tuple of Jacobians of the residuals.
                 Tuple of Numpy arrays 3Nx(3N+3), Q1x(3N+3), Q2x(3N+3), Q3x(3N+3), Q <= 3N.
        :rtype: tuple
        """
        num_positions, dim, num_rel_markers, num_abs_markers = x.shape[1], x.shape[0], mr.shape[1], ma.shape[1]
        num_variables = dim * (num_positions + num_rel_markers)

        # Precompute sin(yaw) and cos(yaw) values to speed up computations
        sx = np.sin(x[2, :])
        cx = np.cos(x[2, :])

        # lil_matrix is a sparse matrix implementation. We use it here to speed up the computations.

        # Odom measurements
        J_odom = lil_matrix((dim * (num_positions - 1), num_variables), dtype=np.float32)
        for t in range(num_positions - 1):
            # ARO homework 3: compute the odom part of the Jacobian in J and J1, i.e.
            #       differentiate res_odom[:, t] w.r.t x[t] and x[t+1]
            
            J1 = -np.eye(3)

            J = np.eye(3)
            J[0, 2] = -sx[t] * z_odom[0, t] - cx[t] * z_odom[1, t]
            J[1, 2] = cx[t] * z_odom[0, t] - sx[t] * z_odom[1, t]

            J = c_odom[t] * J  # apply cost
            J1 = c_odom[t] * J1  # apply cost
            j = t * dim  # index of t-th pose in xmr
            J_odom[j:(j + 3), j:(j + 3)] = J  # derivative w.r.t x[t]
            J_odom[j:(j + 3), (j + 3):(j + 6)] = J1  # derivative w.r.t. x[t+1]

        # Relative marker measurements
        J_mr = lil_matrix((dim * idx_mr.shape[0], num_variables), dtype=np.float32)
        k = 0
        for t in idx_mr:
            # ARO homework 3: compute the relative marker part of the Jacobian in J, i.e.
            #       differentiate res_mr[:, t] w.r.t x[t] and mr

            J = np.eye(3)
            J[0, 2] = -sx[t] * z_mr[0, t] - cx[t] * z_mr[1, t]
            J[1, 2] =  cx[t] * z_mr[0, t] - sx[t] * z_mr[1, t]

            Jm = -np.eye(3)

            J = c_mr[t] * J  # apply cost
            J[2, :] *= self.marker_yaw_scale
            Jm = c_mr[t] * Jm  # apply cost
            Jm[2, :] *= self.marker_yaw_scale
            j = t * dim  # index of t-th pose in xmr
            J_mr[k:(k + 3), j:(j + 3)] = J  # derivative w.r.t. x[t]
            J_mr[k:(k + 3), dim * num_positions:] = Jm  # derivative w.r.t mr
            k = k + 3

        # Absolute marker measurements
        J_ma = lil_matrix((dim * idx_ma.shape[0], num_variables), dtype=np.float32)
        k = 0
        for t in idx_ma:
            # ARO homework 3: compute the absolute marker part of the Jacobian in J, i.e.
            #       differentiate res_ma[:, t] w.r.t x[t]

            J = np.eye(3)
            J[0, 2] = -sx[t] * z_ma[0, t] - cx[t] * z_ma[1, t]
            J[1, 2] =  cx[t] * z_ma[0, t] - sx[t] * z_ma[1, t]

            J = c_ma[t] * J  # apply cost
            J[2, :] *= self.marker_yaw_scale
            j = t * dim  # index of t-th pose in xmr
            J_ma[k:(k + 3), j:(j + 3)] = J  # derivative w.r.t x[t]
            k = k + 3

        # ICP odom measurements
        J_icp = lil_matrix((dim * idx_icp.shape[0], num_variables), dtype=np.float32)
        k = 0
        for t in idx_icp:
            # after homework 4: compute the ICP odom part of the Jacobian in J and J1, i.e.
            #       differentiate res_icp_odom[:, t] w.r.t x[t] and x[t+1]
            J1 = -np.eye(3)

            J = np.eye(3)
            J[0, 2] = -sx[t] * z_icp[0, t] - cx[t] * z_icp[1, t]
            J[1, 2] =  cx[t] * z_icp[0, t] - sx[t] * z_icp[1, t]

            J = c_icp[t] * J  # apply cost
            J[2, :] *= self.icp_yaw_scale
            J1 = c_icp[t] * J1  # apply cost
            J1[2, :] *= self.icp_yaw_scale
            j = t * dim  # index of t-th pose in xmr
            J_icp[k:(k + 3), j:(j + 3)] = J  # derivative w.r.t x[t]
            J_icp[k:(k + 3), (j + 3):(j + 6)] = J1  # derivative w.r.t. x[t+1]
            k = k + 3

        return J_odom, J_mr, J_ma, J_icp

    def compute_residuals_jacobian_opt(
            self, xmr, ma, z_odom, z_mr, z_ma, z_icp, idx_mr, idx_ma, idx_icp, c_odom, c_mr, c_ma, c_icp):
        """Like compute_residuals_jacobian() but formatted for the optimizer."""
        # unpack x, mr from the concatenated state
        x, mr = xmr[0:-3].reshape(3, int((xmr.shape[0] - 3) / 3), order='F'), xmr[-3:].reshape(3, 1)

        J_odom, J_mr, J_ma, J_icp = self.compute_residuals_jacobian(
            x, mr, ma, z_odom, z_mr, z_ma, z_icp, idx_mr, idx_ma, idx_icp, c_odom, c_mr, c_ma, c_icp)

        if not self.fuse_icp:
            return vstack((
                J_odom,
                J_mr,
                J_ma,
            ))

        return vstack((
            J_odom,
            J_mr,
            J_ma,
            J_icp,
        ))

    def visu(self, x, mr, ma, z_odom, z_mr, z_ma, z_icp, z_gt_odom, last_optimized_idx=None, only_main=False):
        # this happens right after the start before we have the first odom measurement
        if len(z_odom.shape) < 2 or z_odom.shape[1] == 0:
            return

        if last_optimized_idx is None:
            last_optimized_idx = x.shape[1] - 1

        x = x.transpose()
        x_odom = np.array(self.x_odom).transpose()
        x_icp_odom = np.array(self.x_icp_odom).transpose()

        idx_mr = np.where((~np.isnan(z_mr)).all(axis=0))[0]
        idx_ma = np.where((~np.isnan(z_ma)).all(axis=0))[0]
        idx_gt = np.where((~np.isnan(z_gt_odom)).all(axis=0))[0]
        if (isinstance(z_gt_odom, np.ndarray) and z_gt_odom.shape[0] == 0) or (isinstance(z_gt_odom, list) and len(z_gt_odom) == 0):
            idx_gt = np.array([])
        # idx_mr_gt = np.where((~np.isnan(z_gt_odom + z_mr[:, :-1])).all(axis=0))[0]
        # idx_ma_gt = np.where((~np.isnan(z_gt_odom + z_ma[:, :-1])).all(axis=0))[0]
        x_icp_odom_idx = np.where((~np.isnan(x_icp_odom)).all(axis=0))[0]

        z_mr_wcf = self.get_world_pose(z_mr[:, idx_mr], x[:, idx_mr])
        z_ma_wcf = self.get_world_pose(z_ma[:, idx_ma], x[:, idx_ma])
        # z_mr_gt_wcf = self.get_world_pose(z_mr[:, idx_mr_gt], z_gt_odom[:, idx_mr_gt])
        # z_ma_gt_wcf = self.get_world_pose(z_ma[:, idx_ma_gt], z_gt_odom[:, idx_ma_gt])
        gt_odom = np.array(z_gt_odom)

        nplots = 5  # update this when more subplots are added
        if idx_gt.shape[0] > 0:
            nplots += 2  # update this when more GT subplots are added
        if only_main:
            nplots = 2
        plot_index = 0

        fig = plt.figure(1, figsize=(10, 13))
        plt.clf()

        plot_index += 1
        main_ax = plt.subplot(nplots, 1, (plot_index, plot_index + 1))
        plot_index += 1
        for k, j in enumerate(idx_ma):
            main_ax.plot([x[0, j], z_ma_wcf[0, k]], [x[1, j], z_ma_wcf[1, k]], 'x:', color='k', linewidth=1, mew=2)
            main_ax.plot([ma[0, 0], z_ma_wcf[0, k]], [ma[1, 0], z_ma_wcf[1, k]], ':', color='k', linewidth=1, mew=2)
        # for k, j in enumerate(idx_ma_gt):
        #     main_ax.plot([z_gt_odom[0, j], z_ma_gt_wcf[0, k]], [z_gt_odom[1, j], z_ma_gt_wcf[1, k]], 'x:', color='r', linewidth=1, mew=2)
        #     main_ax.plot([ma[0, 0], z_ma_gt_wcf[0, k]], [ma[1, 0], z_ma_gt_wcf[1, k]], ':', color='r', linewidth=1, mew=2)

        for k, j in enumerate(idx_mr):
            main_ax.plot([x[0, j], z_mr_wcf[0, k]], [x[1, j], z_mr_wcf[1, k]], 'x:', color='c', linewidth=1, mew=2)
            main_ax.plot([mr[0, 0], z_mr_wcf[0, k]], [mr[1, 0], z_mr_wcf[1, k]], ':', color='c', linewidth=1, mew=2)
        # for k, j in enumerate(idx_mr_gt):
        #     main_ax.plot([z_gt_odom[0, j], z_mr_gt_wcf[0, k]], [z_gt_odom[1, j], z_mr_gt_wcf[1, k]], 'x:', color='r', linewidth=1, mew=2)
        #     main_ax.plot([mr[0, 0], z_mr_gt_wcf[0, k]], [mr[1, 0], z_mr_gt_wcf[1, k]], ':', color='r', linewidth=1, mew=2)

        if idx_gt.shape[0] > 0:
            main_ax.plot(gt_odom[0, :(last_optimized_idx+1)], gt_odom[1, :(last_optimized_idx+1)], '-', color='r', linewidth=1, mew=2, label="GT odom")
            main_ax.plot(gt_odom[0, last_optimized_idx:], gt_odom[1, last_optimized_idx:], '-', color=(1.0, 0.5, 0.5), linewidth=1, mew=2)
        main_ax.plot(x_odom[0, :(last_optimized_idx+1)], x_odom[1, :(last_optimized_idx+1)], '-', color='y', linewidth=1, mew=2, label="Integrated odom")
        main_ax.plot(x_odom[0, last_optimized_idx:], x_odom[1, last_optimized_idx:], '-', color=(1.0, 1.0, 0.5), linewidth=1, mew=2)
        main_ax.plot(x_icp_odom[0, x_icp_odom_idx], x_icp_odom[1, x_icp_odom_idx], '-', color='g', linewidth=1, mew=2, label="Integrated ICP odom")
        main_ax.plot(x_icp_odom[0, x_icp_odom_idx[x_icp_odom_idx <= last_optimized_idx]], x_icp_odom[1, x_icp_odom_idx[x_icp_odom_idx <= last_optimized_idx]], '-', color='g', linewidth=1, mew=2, label="Integrated ICP odom")
        main_ax.plot(x_icp_odom[0, x_icp_odom_idx[x_icp_odom_idx >= last_optimized_idx]], x_icp_odom[1, x_icp_odom_idx[x_icp_odom_idx >= last_optimized_idx]], '-', color=(0.5, 1.0, 0.5), linewidth=1, mew=2)
        main_ax.plot(x[0, :(last_optimized_idx+1)], x[1, :(last_optimized_idx+1)], '-', color='b', linewidth=2, mew=2, label="Fused odom")
        main_ax.plot(x[0, last_optimized_idx:], x[1, last_optimized_idx:], '-', color=(0.5, 0.5, 1.0), linewidth=2, mew=2)
        main_ax.scatter(x_odom[0, -1], x_odom[1, -1], color='y', s=40)
        main_ax.scatter(x_icp_odom[0, x_icp_odom_idx[-1]], x_icp_odom[1, x_icp_odom_idx[-1]], color='g', s=40)
        main_ax.scatter(x[0, -1], x[1, -1], color='b', s=80)
        if idx_gt.shape[0] > 0:
            main_ax.scatter(gt_odom[0, -1], gt_odom[1, -1], color='r', s=40)

        if self.mr_gt is not None:
            main_ax.plot(self.mr_gt[0, 0], self.mr_gt[1, 0], 's', color='y', linewidth=1, mew=6, label="Rel.m. GT")
        if not np.any(np.isnan(mr)):
            main_ax.plot(mr[0, :], mr[1, :], 's', color='c', linewidth=1, mew=6, label="Rel. marker")
        else:
            main_ax.plot([0], [0], 's', color='c', linewidth=1, mew=6, label="Rel. marker")
        main_ax.plot(ma[0, :], ma[1, :], 's', color='k', linewidth=1, mew=6, label="Abs. marker")

        # main_ax.axis('equal')
        main_ax.set_xlabel('x')
        main_ax.set_ylabel('y')
        main_ax.set_xlim(-2.5, 2.5)
        main_ax.set_ylim(-2.5, 2.5)
        main_ax.grid()
        main_ax.legend()

        if only_main:
            return

        stamps = np.array([(t - self.t_odom[0]) / 1e9 for t in self.t_odom[:x.shape[1]]])

        plot_index += 1
        yaw_ax = plt.subplot(nplots, 1, plot_index)

        if idx_gt.shape[0] > 0:
            yaw_ax.plot(stamps[idx_gt[idx_gt <= last_optimized_idx]], gt_odom[2, idx_gt[idx_gt <= last_optimized_idx]], '-', color='r', linewidth=1, mew=2, label="GT odom")
            yaw_ax.plot(stamps[idx_gt[idx_gt >= last_optimized_idx]], gt_odom[2, idx_gt[idx_gt >= last_optimized_idx]], '-', color=(1.0, 0.5, 0.5), linewidth=1, mew=2)
        yaw_ax.plot(stamps[:(last_optimized_idx+1)], x_odom[2, :(last_optimized_idx+1)], '-', color='y', linewidth=1, mew=2, label="Integrated odom")
        yaw_ax.plot(stamps[last_optimized_idx:], x_odom[2, last_optimized_idx:], '-', color=(1.0, 1.0, 0.5), linewidth=1, mew=2)
        yaw_ax.plot(stamps[x_icp_odom_idx[x_icp_odom_idx <= last_optimized_idx]], x_icp_odom[2, x_icp_odom_idx[x_icp_odom_idx <= last_optimized_idx]], '-', color='g', linewidth=1, mew=2, label="Integrated ICP odom")
        yaw_ax.plot(stamps[x_icp_odom_idx[x_icp_odom_idx >= last_optimized_idx]], x_icp_odom[2, x_icp_odom_idx[x_icp_odom_idx >= last_optimized_idx]], '-', color=(0.5, 1.0, 0.5), linewidth=1, mew=2)
        yaw_ax.plot(stamps[:(last_optimized_idx+1)], x[2, :(last_optimized_idx+1)], '-', color='b', linewidth=1, mew=2, label="Fused odom")
        yaw_ax.plot(stamps[last_optimized_idx:], x[2, last_optimized_idx:], '-', color=(0.5, 0.5, 1.0), linewidth=1, mew=2)

        yaw_ax.set_ylabel('yaw')
        yaw_ax.set_yticks(np.arange(-1.5*np.pi, 1.5*np.pi, np.pi/2))
        yaw_ax.grid()

        plot_index += 1
        res_ax = plt.subplot(nplots, 1, plot_index)

        idx_ma = np.where((~np.isnan(z_ma)).all(axis=0))[0]
        idx_mr = np.where((~np.isnan(z_mr)).all(axis=0))[0]
        idx_icp = np.where((~np.isnan(z_icp)).all(axis=0))[0]
        c_odom = np.ones((x.shape[1] - 1,))
        c_mr = np.ones((max(idx_mr) + 1,)) if len(idx_mr) > 0 else np.array([])
        c_ma = np.ones((max(idx_ma) + 1,)) if len(idx_ma) > 0 else np.array([])
        c_icp = np.ones((max(idx_icp) + 1,)) if len(idx_icp) > 0 else np.array([])
        res_odom, res_mr, res_ma, res_icp = self.compute_residuals(
            x, mr, ma, z_odom, z_mr, z_ma, z_icp, idx_mr, idx_ma, idx_icp, c_odom, c_mr, c_ma, c_icp)

        res_ax.plot(stamps[1:][:(last_optimized_idx+1)], np.linalg.norm(res_odom, axis=0)[:(last_optimized_idx+1)], label="res_odom", color='k')
        res_ax.plot(stamps[1:][last_optimized_idx:], np.linalg.norm(res_odom, axis=0)[last_optimized_idx:], color=(0.5, 0.5, 0.5))
        res_ax.plot(stamps[idx_mr[idx_mr <= last_optimized_idx]], np.linalg.norm(res_mr, axis=0)[idx_mr <= last_optimized_idx], label="res_mr", color='r')
        res_ax.plot(stamps[idx_mr[idx_mr >= last_optimized_idx]], np.linalg.norm(res_mr, axis=0)[idx_mr >= last_optimized_idx], color=(1.0, 0.5, 0.5))
        res_ax.plot(stamps[idx_ma[idx_ma <= last_optimized_idx]], np.linalg.norm(res_ma, axis=0)[idx_ma <= last_optimized_idx], label="res_ma", color='g')
        res_ax.plot(stamps[idx_ma[idx_ma >= last_optimized_idx]], np.linalg.norm(res_ma, axis=0)[idx_ma >= last_optimized_idx], color=(0.5, 1.0, 0.5))
        if self.fuse_icp:
            res_ax.plot(stamps[idx_icp[idx_icp <= last_optimized_idx]], np.linalg.norm(res_icp, axis=0)[idx_icp <= last_optimized_idx], label="res_icp", color='b')
            res_ax.plot(stamps[idx_icp[idx_icp >= last_optimized_idx]], np.linalg.norm(res_icp, axis=0)[idx_icp >= last_optimized_idx], color=(0.5, 0.5, 1.0))

        res_ax.set_ylabel('unscaled residuals')
        res_ax.grid()
        res_ax.legend(loc='upper left', frameon=False)

        plot_index += 1
        res_scaled_ax = plt.subplot(nplots, 1, plot_index)

        c_odom = np.array(self.c_odom)[1:x.shape[1]]
        c_mr = np.array(self.c_mr)[:(max(idx_mr) + 1)] if len(idx_mr) > 0 else np.array([])
        c_ma = np.array(self.c_ma)[:(max(idx_ma) + 1)] if len(idx_ma) > 0 else np.array([])
        c_icp = np.array(self.c_icp)[:(max(idx_icp) + 1)] if len(idx_icp) > 0 else np.array([])
        res_odom, res_mr, res_ma, res_icp = self.compute_residuals(
            x, mr, ma, z_odom, z_mr, z_ma, z_icp, idx_mr, idx_ma, idx_icp, c_odom, c_mr, c_ma, c_icp)

        res_scaled_ax.plot(stamps[1:][:(last_optimized_idx+1)], np.linalg.norm(res_odom, axis=0)[:(last_optimized_idx+1)], label="res_odom", color='k')
        res_scaled_ax.plot(stamps[1:][last_optimized_idx:], np.linalg.norm(res_odom, axis=0)[last_optimized_idx:], color=(0.5, 0.5, 0.5))
        res_scaled_ax.plot(stamps[idx_mr[idx_mr <= last_optimized_idx]], np.linalg.norm(res_mr, axis=0)[idx_mr <= last_optimized_idx], label="res_mr", color='r')
        res_scaled_ax.plot(stamps[idx_mr[idx_mr >= last_optimized_idx]], np.linalg.norm(res_mr, axis=0)[idx_mr >= last_optimized_idx], color=(1.0, 0.5, 0.5))
        res_scaled_ax.plot(stamps[idx_ma[idx_ma <= last_optimized_idx]], np.linalg.norm(res_ma, axis=0)[idx_ma <= last_optimized_idx], label="res_ma", color='g')
        res_scaled_ax.plot(stamps[idx_ma[idx_ma >= last_optimized_idx]], np.linalg.norm(res_ma, axis=0)[idx_ma >= last_optimized_idx], color=(0.5, 1.0, 0.5))
        if self.fuse_icp:
            res_scaled_ax.plot(stamps[idx_icp[idx_icp <= last_optimized_idx]], np.linalg.norm(res_icp, axis=0)[idx_icp <= last_optimized_idx], label="res_icp", color='b')
            res_scaled_ax.plot(stamps[idx_icp[idx_icp >= last_optimized_idx]], np.linalg.norm(res_icp, axis=0)[idx_icp >= last_optimized_idx], color=(0.5, 0.5, 1.0))

        if idx_gt.shape[0] == 0:
            res_scaled_ax.set_xlabel('t')
        res_scaled_ax.set_ylabel('scaled residuals')
        res_scaled_ax.grid()
        res_scaled_ax.legend(loc='upper left', frameon=False)

        if idx_gt.shape[0] > 0:
            plot_index += 1
            gt_ax = plt.subplot(nplots, 1, plot_index)
            errors_x = x[0, 1:] - gt_odom[0, :]
            errors_y = x[1, 1:] - gt_odom[1, :]
            errors_yaw = (x[2, 1:] - gt_odom[2, :] + np.pi) % (2 * np.pi) - np.pi
            errors = np.linalg.norm(np.vstack((errors_x, errors_y, errors_yaw)), axis=0)
            stamps = [nsec_to_sec(t - self.t_odom[0]) for t in self.t_odom[1:x.shape[1]]]
            gt_ax.plot(stamps[:(last_optimized_idx+1)], errors[:(last_optimized_idx+1)], label="FG localization error", color='k')
            gt_ax.plot(stamps[last_optimized_idx:], errors[last_optimized_idx:], color=(0.5, 0.5, 0.5))
            gt_ax.plot(stamps[:(last_optimized_idx+1)], errors_x[:(last_optimized_idx+1)], label="X error", color='r')
            gt_ax.plot(stamps[last_optimized_idx:], errors_x[last_optimized_idx:], color=(1.0, 0.5, 0.5))
            gt_ax.plot(stamps[:(last_optimized_idx+1)], errors_y[:(last_optimized_idx+1)], label="Y error", color='g')
            gt_ax.plot(stamps[last_optimized_idx:], errors_y[last_optimized_idx:], color=(0.5, 1.0, 0.5))
            gt_ax.plot(stamps[:(last_optimized_idx+1)], errors_yaw[:(last_optimized_idx+1)], label="Yaw error", color='b')
            gt_ax.plot(stamps[last_optimized_idx:], errors_yaw[last_optimized_idx:], color=(0.5, 0.5, 1.0))
            gt_ax.set_ylabel('fused error')
            ylim_min = np.nanmin(np.hstack((errors_x, errors_y, errors_yaw, [-1]))) * 1.1
            ylim_max = np.nanmax(np.hstack((errors_x, errors_y, errors_yaw, errors, [-1]))) * 1.1
            gt_ax.set_ylim(ylim_min, ylim_max)
            gt_ax.grid()
            gt_ax.legend(loc='upper left', frameon=False)

            plot_index += 1
            gt_odom_ax = plt.subplot(nplots, 1, plot_index)
            x_odom2 = x_odom[:, :(gt_odom.shape[1]+1)]
            odom_and_gt_idxs = np.where((~np.isnan(x_odom2[:, 1:]) & ~np.isnan(gt_odom)).all(axis=0))[0]
            if odom_and_gt_idxs.shape[0] > 0:
                odom_errors_x = x_odom2[0, 1:] - gt_odom[0, :]
                odom_errors_y = x_odom2[1, 1:] - gt_odom[1, :]
                odom_errors_yaw = (x_odom2[2, 1:] - gt_odom[2, :] + np.pi) % (2 * np.pi) - np.pi
                odom_errors = np.linalg.norm(np.vstack((odom_errors_x, odom_errors_y, odom_errors_yaw)), axis=0)
                gt_odom_ax.plot(stamps[:(last_optimized_idx+1)], odom_errors[:(last_optimized_idx+1)], label="Odom localization error", color='m')
                gt_odom_ax.plot(stamps[last_optimized_idx:], odom_errors[last_optimized_idx:], color=(0.5, 1.0, 1.0))
                gt_odom_ax.plot(stamps[:(last_optimized_idx+1)], odom_errors_x[:(last_optimized_idx+1)], label="X error", color='r')
                gt_odom_ax.plot(stamps[last_optimized_idx:], odom_errors_x[last_optimized_idx:], color=(1.0, 0.5, 0.5))
                gt_odom_ax.plot(stamps[:(last_optimized_idx+1)], odom_errors_y[:(last_optimized_idx+1)], label="Y error", color='g')
                gt_odom_ax.plot(stamps[last_optimized_idx:], odom_errors_y[last_optimized_idx:], color=(0.5, 1.0, 0.5))
                gt_odom_ax.plot(stamps[:(last_optimized_idx+1)], odom_errors_yaw[:(last_optimized_idx+1)], label="Yaw error", color='b')
                gt_odom_ax.plot(stamps[last_optimized_idx:], odom_errors_yaw[last_optimized_idx:], color=(0.5, 0.5, 1.0))
                ylim_min = np.nanmin(np.hstack((odom_errors_x, odom_errors_y, odom_errors_yaw, [-1]))) * 1.1
                ylim_max = np.nanmax(np.hstack((odom_errors_x, odom_errors_y, odom_errors_yaw, odom_errors, [-1]))) * 1.1
                gt_odom_ax.set_ylim(ylim_min, ylim_max)
            gt_odom_ax.set_xlabel('t')
            gt_odom_ax.set_ylabel('odom error')
            gt_odom_ax.grid()
            gt_odom_ax.legend(loc='upper left', frameon=False)

        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.01)
        plt.savefig("/tmp/fig%04i.png" % (x.shape[1],))

        # plt.pause() brings the plot window into foreground on each redraw, which is not convenient, so we only call it
        # the first time and then we use a workaround that hopefully works on all plotting backends (but not sure!)
        if not self._figure_shown:
            plt.pause(0.001)
            self._figure_shown = True
        else:
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.001)


if __name__ == '__main__':
    # This code is for playing with the optimization of factorgraph. It generates a sample trajectory and runs the
    # factorgraph localization on it.
    # np.random.seed(1)
    def sim(trajectory_length=10, v=0.3, w=0.1,
            x0=np.array([0, 0, 0]), mr=np.array([[2], [1], [0]]), ma=np.array([[1], [0], [0]]),
            noise_odom=0.1, noise_mr=0.1, noise_ma=0.1):
        """Generate a spiral trajectory with the given parameters.

        :param int trajectory_length: Number of steps of the trajectory.
        :param float v: Linear velocity.
        :param float w: Angular velocity.
        :param np.ndarray x0: Initial pose estimate. Numpy array 3.
        :param np.ndarray mr: Relative marker pose. Numpy array 3x1.
        :param np.ndarray ma: Absolute marker pose. Numpy array 3x1.
        :param float noise_odom: Noise of odometry measurements.
        :param float noise_mr: Noise of relative marker measurements.
        :param float noise_ma: Noise of absolute marker measurements.
        :return: x0, x, mr, ma, z_odom, z_ma, z_mr
        """
        def forward(u, x0):
            K = u.shape[1]
            pos_x, pos_y, phi = x0[0], x0[1], x0[2]
            x = [np.stack((pos_x, pos_y, phi))]

            for k in range(K):
                pos_x = pos_x + u[0, k] * np.cos(phi)
                pos_y = pos_y + u[0, k] * np.sin(phi)
                phi = phi + u[2, k]
                x.append(np.stack((pos_x, pos_y, phi)))
            return np.hstack(x)

        np.set_printoptions(formatter={'float_kind': "{:.4f}".format})

        # ground truth robot positions x and marker mr
        x0 = x0.reshape(3, 1)  # initial position
        u = np.ones((3, trajectory_length-1), dtype=float)
        u[0, :] = u[0, :] * v
        u[1, :] = 0
        u[2, :] = u[2, :] * w
        x = forward(u, x0)  # generate ground truth trajectory x (based on ground truth velocity control u)
        
        z_odom = u + np.random.randn(3, trajectory_length-1)*noise_odom
        z_ma, z_mr = np.zeros((3, x.shape[1] - 1)), np.zeros((3, x.shape[1] - 1))
        for t in range(1,x.shape[1]):
            R = np.array([[np.cos(x[2,t]), -np.sin(x[2,t])], [np.sin(x[2,t]), np.cos(x[2,t])]])
            z_ma[0:2, t - 1] = np.matmul(R.transpose(), ma[0:2,0]) - np.matmul(R.transpose(), x[0:2, t]) + np.random.randn(2) * noise_ma
            z_ma[2, t - 1] = ma[2] - x[2, t] + np.random.randn() * noise_ma
            z_mr[0:2, t - 1] = np.matmul(R.transpose(), mr[0:2,0]) - np.matmul(R.transpose(), x[0:2, t]) + np.random.randn(2) * noise_mr
            z_mr[2, t - 1] = mr[2] - x[2, t] + np.random.randn() * noise_mr

        return x0, x, mr, ma, z_odom, z_ma, z_mr

    # Generate a test trajectory
    x0, X, Mr, Ma, Z_odom, Z_ma, Z_mr = sim(
        trajectory_length=30, v=0.3, w=0.3, noise_odom=0.1, noise_mr=0.1, noise_ma=0.1)

    # Set costs for the measurements
    C_odom = 1.0 * np.ones((Z_odom.shape[1],))
    C_mr = 1.0 * np.ones((Z_mr.shape[1],))
    C_ma = 1.0 * np.ones((Z_ma.shape[1],))

    fg = FactorGraph(ma=Ma.ravel().tolist(), mr_gt=Mr.ravel().tolist())

    opt_step = 3  # After how many measurements we perform optimization
    mr_keep_every_nth = 4
    ma_keep_every_nth = 2

    for i in range(X.shape[1] - 1):
        gt = X[:, i].tolist()
        z_odom = Z_odom[:, i].tolist()
        z_mr = Z_mr[:, i].tolist() if i % mr_keep_every_nth == 0 else [np.nan, np.nan, np.nan]
        z_ma = Z_ma[:, i].tolist() if i % ma_keep_every_nth == 0 else [np.nan, np.nan, np.nan]
        t = int((i + 1) * 1e9)

        # Add the measurement
        fg.add_z(z_odom, C_odom[i], z_mr, C_mr[i], z_ma, C_ma[i], [], [], gt, t, t, None)

        # Reoptimize and visualize only after a few measurements
        if i % opt_step != (opt_step - 1):
            continue

        # Data conversion from Python lists to Numpy arrays necessary for the library to work
        x = np.array(fg.x)
        mr = np.array(fg.mr)
        ma = np.array(fg.ma)
        z_odom = np.array(fg.z_odom).transpose()
        z_mr = np.array(fg.z_mr).transpose()
        z_ma = np.array(fg.z_ma).transpose()
        z_icp = np.array(fg.z_icp).transpose()
        c_odom = np.array(fg.c_odom)
        c_ma = np.array(fg.c_ma)
        c_mr = np.array(fg.c_mr)
        c_icp = np.array(fg.c_icp)

        # Run optimization
        res = fg.optimize(x, mr, ma, z_odom, z_mr, z_ma, z_icp, c_odom, c_mr, c_ma, c_icp)

        # Store results of the optimization
        fg.x = res[0]
        fg.mr = res[1]

        print("Relative marker localization error: " + str(np.linalg.norm(fg.mr - Mr)))

        # Visualize the factorgraph
        fg.visu(np.array(fg.x), fg.mr, ma, z_odom, z_mr, z_ma, z_icp, np.array(fg.z_gt_odom).transpose(),
                fg.last_optimized_idx, only_main=True)
        plt.tight_layout()
        plt.pause(1)
        image_name = 'traj_%02i.png' % i
        image_path = os.path.join('/tmp', image_name)
        plt.savefig(image_path)
    plt.pause(10)
