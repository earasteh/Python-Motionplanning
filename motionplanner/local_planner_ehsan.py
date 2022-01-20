# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Author: Ryan De Iaco
# Additional Comments: Carlos Wang
# Date: October 29, 2018

import numpy as np
import copy
from motionplanner import path_optimizer
from motionplanner import collision_checker
from motionplanner import velocity_planner
from math import sin, cos, pi, sqrt

# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.



def get_closest_index(waypoints, ego_state):
    """Gets closest index a given list of waypoints to the vehicle position.

    args:
        waypoints: current waypoints to track. (global frame)
            length and speed in m and m/s.
            (includes speed to track at each x,y location.)
            format: [[x0, y0, v0],
                     [x1, y1, v1],
                     ...
                     [xn, yn, vn]]
            example:
                waypoints[2][1]:
                returns the 3rd waypoint's y position

                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)

    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """
    closest_len = float('Inf')
    closest_index = 0
    # Done
    # ------------------------------------------------------------------
    diff = []
    for i in range(len(waypoints)):
        dxi = waypoints[i][0] - ego_state[0]
        dyi = waypoints[i][1] - ego_state[1]
        diff = np.sqrt(dxi ** 2 + dyi ** 2)
        if diff <= closest_len:
            closest_len = diff
            closest_index = i
    # ------------------------------------------------------------------

    return closest_len, closest_index

class LocalPlanner:
    def __init__(self, lookahead, num_paths, path_offset, circle_offsets, circle_radii,
                 path_select_weight, time_gap, a_max, slow_speed,
                 stop_line_buffer):
        self._lookahead = lookahead
        self._num_paths = num_paths
        self._path_offset = path_offset
        self._prev_best_path = []
        self._path_optimizer = path_optimizer.PathOptimizer()
        self._collision_checker = \
            collision_checker.CollisionChecker(circle_offsets,
                                               circle_radii,
                                               path_select_weight)
        self._velocity_planner = \
            velocity_planner.VelocityPlanner(time_gap, a_max, slow_speed,
                                             stop_line_buffer)

    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
        """Gets the goal index for the vehicle.

        Set to be the earliest waypoint that has
        accumulated arc length (including closest_len) that is greater than or
        equal to self._lookahead.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]:
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
        """
        # Find the farthest point along the path that is within the
        # lookahead distance of the ego vehicle.
        # Take the distance from the ego vehicle to the closest waypoint into
        # consideration.
        arc_length = closest_len
        wp_index = closest_index

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        # Otherwise, find our next waypoint.
        # Done
        # ------------------------------------------------------------------
        while wp_index < len(waypoints) - 1:
            dx = waypoints[wp_index][0] - waypoints[wp_index + 1][0]
            dy = waypoints[wp_index][1] - waypoints[wp_index + 1][1]
            arc_length += np.sqrt(dx ** 2 + dy ** 2)
            if arc_length > self._lookahead:
                break
            wp_index += 1
        # ------------------------------------------------------------------
        return wp_index
    # Computes the goal state set from a given goal position. This is done by
    # laterally sampling offsets from the goal location along the direction
    # perpendicular to the goal yaw of the ego vehicle.
    def get_goal_state_set(self, goal_index, goal_state, waypoints, ego_state):
        """Gets the goal states given a goal position.

        Gets the goal states given a goal position. The states

        args:
            goal_index: Goal index for the vehicle to reach
                i.e. waypoints[goal_index] gives the goal waypoint
            goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal], in units [m, m, m/s]
            waypoints: current waypoints to track. length and speed in m and m/s.
                (includes speed to track at each x,y location.) (global frame)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]:
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle, in the global frame.
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
        returns:
            goal_state_set: Set of goal states (offsetted laterally from one
                another) to be used by the local planner to plan multiple
                proposal paths. This goal state set is in the vehicle frame.
                format: [[x0, y0, t0, v0],
                         [x1, y1, t1, v1],
                         ...
                         [xm, ym, tm, vm]]
                , where m is the total number of goal states
                  [x, y, t] are the position and yaw values at each goal
                  v is the goal speed at the goal point.
                  all units are in m, m/s and radians
        """
        # Compute the final heading based on the next index.
        # If the goal index is the last in the set of waypoints, use
        # the previous index instead.
        # To do this, compute the delta_x and delta_y values between
        # consecutive waypoints, then use the np.arctan2() function.
        # ------------------------------------------------------------------
        if goal_index < len(waypoints) - 1:
            delta_x = waypoints[goal_index + 1][0] - waypoints[goal_index][0]
            delta_y = waypoints[goal_index + 1][1] - waypoints[goal_index][1]
        else:
            delta_x = waypoints[goal_index][0] - waypoints[goal_index - 1][0]
            delta_y = waypoints[goal_index][1] - waypoints[goal_index - 1][1]
        heading = np.arctan2(delta_y, delta_x)
        # ------------------------------------------------------------------

        # Compute the center goal state in the local frame using
        # the ego state. The following code will transform the input
        # goal state to the ego vehicle's local frame.
        # The goal state will be of the form (x, y, t, v).
        goal_state_local = copy.copy(goal_state)

        # Translate so the ego state is at the origin in the new frame.
        # This is done by subtracting the ego_state from the goal_state_local.
        # ------------------------------------------------------------------
        goal_state_local[0] -= ego_state[0]
        goal_state_local[1] -= ego_state[1]
        # ------------------------------------------------------------------

        # Rotate such that the ego state has zero heading in the new frame.
        # Recall that the general rotation matrix is [cos(theta) -sin(theta)
        #                                             sin(theta)  cos(theta)]
        # and that we are rotating by -ego_state[2] to ensure the ego vehicle's
        # current yaw corresponds to theta = 0 in the new local frame.
        # ------------------------------------------------------------------
        theta = -ego_state[2]
        # rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
        #                        [np.sin(theta), np.cos(theta)]])
        # ego_vec = np.array([ego_state[0], ego_state[1]])
        # goal_x, goal_y = np.matmul(rot_matrix, ego_vec)
        goal_x = goal_state_local[0] * cos(theta) - goal_state_local[1] * sin(theta)
        goal_y = goal_state_local[0] * sin(theta) + goal_state_local[1] * cos(theta)
        # ------------------------------------------------------------------

        # Compute the goal yaw in the local frame by subtracting off the
        # current ego yaw from the heading variable.
        # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
        # ------------------------------------------------------------------
        goal_t = heading - ego_state[2]
        # ------------------------------------------------------------------

        # Velocity is preserved after the transformation.
        goal_v = goal_state[2]

        # Keep the goal heading within [-pi, pi] so the optimizer behaves well.
        if goal_t > pi:
            goal_t -= 2 * pi
        elif goal_t < -pi:
            goal_t += 2 * pi

        # Compute and apply the offset for each path such that
        # all of the paths have the same heading of the goal state,
        # but are laterally offset with respect to the goal heading.
        goal_state_set = []
        for i in range(self._num_paths):
            # Compute offsets that span the number of paths set for the local
            # planner. Each offset goal will be used to generate a potential
            # path to be considered by the local planner.
            offset = (i - self._num_paths // 2) * self._path_offset

            # Compute the projection of the lateral offset along the x
            # and y axis. To do this, multiply the offset by cos(goal_theta + pi/2)
            # and sin(goal_theta + pi/2), respectively.
            # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
            # ------------------------------------------------------------------
            x_offset = offset * np.cos(goal_t + pi / 2)
            y_offset = offset * np.sin(goal_t + pi / 2)
            # ------------------------------------------------------------------

            goal_state_set.append([goal_x + x_offset,
                                   goal_y + y_offset,
                                   goal_t,
                                   goal_v])

        return goal_state_set
