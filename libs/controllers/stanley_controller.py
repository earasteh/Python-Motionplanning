import numpy as np

from libs.utils.normalise_angle import normalise_angle


class StanleyController:

    def __init__(self, control_gain=2.5, softening_gain=1.0, yaw_rate_gain=0.0, steering_damp_gain=0.0,
                 max_steer=np.deg2rad(24), wheelbase=0.0,
                 waypoints=None):
        """
        Stanley Controller

        At initialisation
        :param control_gain:                (float) time constant [1/s]
        :param softening_gain:              (float) softening gain [m/s]
        :param yaw_rate_gain:               (float) yaw rate gain [rad]
        :param steering_damp_gain:          (float) steering damp gain
        :param max_steer:                   (float) vehicle's steering limits [rad]
        :param wheelbase:                   (float) vehicle's wheelbase [m]
        :param path_x:                      (numpy.ndarray) list of x-coordinates along the path
        :param path_y:                      (numpy.ndarray) list of y-coordinates along the path
        :param path_yaw:                    (numpy.ndarray) list of discrete yaw values along the path
        :param dt:                          (float) discrete time period [s]

        At every time step
        :param x:                           (float) vehicle's x-coordinate [m]
        :param y:                           (float) vehicle's y-coordinate [m]
        :param yaw:                         (float) vehicle's heading [rad]
        :param target_velocity:             (float) vehicle's velocity [m/s]
        :param steering_angle:              (float) vehicle's steering angle [rad]

        :return limited_steering_angle:     (float) steering angle after imposing steering limits [rad]
        :return target_index:               (int) closest path index
        :return crosstrack_error:           (float) distance from closest path index [m]
        """

        self.k = control_gain
        self.k_soft = softening_gain
        self.k_yaw_rate = yaw_rate_gain
        self.k_damp_steer = steering_damp_gain
        self.max_steer = max_steer
        self.L = wheelbase

        self._waypoints = waypoints
        self._lookahead_distance = 4.0
        self.cross_track_deadband = 0.01

        # self.px = path_x
        # self.py = path_y
        # self.pyaw = path_yaw

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_lookahead_index(self, x, y, lookahead_distance):
        min_idx = 0
        min_dist = float("inf")
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                self._waypoints[i][0] - x,
                self._waypoints[i][1] - y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        total_dist = min_dist
        lookahead_idx = min_idx
        for i in range(min_idx + 1, len(self._waypoints)):
            if total_dist >= lookahead_distance:
                break
            total_dist += np.linalg.norm(np.array([
                self._waypoints[i][0] - self._waypoints[i - 1][0],
                self._waypoints[i][1] - self._waypoints[i - 1][1]]))
            lookahead_idx = i
        return lookahead_idx

    def stanley_control(self, x, y, yaw, current_velocity):
        """
        :param x:
        :param y:
        :param yaw:
        :param current_velocity:
        :return: steering output, target index, crosstrack error
        """
        crosstrack_error = float("inf")
        crosstrack_vector = np.array([float("inf"), float("inf")])

        ce_idx = self.get_lookahead_index(x, y, self._lookahead_distance)
        crosstrack_vector = np.array([self._waypoints[ce_idx][0] - \
                                      x - self._lookahead_distance * np.cos(yaw),
                                      self._waypoints[ce_idx][1] - \
                                      y - self._lookahead_distance * np.sin(yaw)])
        crosstrack_error = np.linalg.norm(crosstrack_vector)

        if crosstrack_error < self.cross_track_deadband:
            crosstrack_error = 0

        # Compute the sign of the crosstrack error
        crosstrack_heading = np.arctan2(crosstrack_vector[1], crosstrack_vector[0])
        crosstrack_heading_error = crosstrack_heading - yaw
        crosstrack_heading_error = (crosstrack_heading_error + np.pi) % (2 * np.pi) - np.pi
        crosstrack_sign = np.sign(crosstrack_heading_error)

        # Compute heading relative to trajectory (heading error)
        # First ensure that we are not at the last index. If we are,
        # flip back to the first index (loop the waypoints)
        if ce_idx < len(self._waypoints) - 1:
            vect_wp0_to_wp1 = np.array(
                [self._waypoints[ce_idx + 1][0] - self._waypoints[ce_idx][0],
                 self._waypoints[ce_idx + 1][1] - self._waypoints[ce_idx][1]])
            trajectory_heading = np.arctan2(vect_wp0_to_wp1[1],
                                            vect_wp0_to_wp1[0])
        else:
            vect_wp0_to_wp1 = np.array(
                [self._waypoints[0][0] - self._waypoints[-1][0],
                 self._waypoints[0][1] - self._waypoints[-1][1]])
            trajectory_heading = np.arctan2(vect_wp0_to_wp1[1],
                                            vect_wp0_to_wp1[0])

        heading_error = trajectory_heading - yaw
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        steering_output = heading_error + \
                          np.arctan(self.k * crosstrack_sign * crosstrack_error / \
                                    (current_velocity + self.k_soft))

        limited_steering_angle = np.clip(steering_output, -self.max_steer, self.max_steer)

        return limited_steering_angle, ce_idx, crosstrack_error


class LongitudinalController:
    def __init__(self, p_gain=1, integral_gain=0, derivative_gain=0):
        self.kp = p_gain
        self.ki = integral_gain
        self.kd = derivative_gain

    def long_control(self, desired_velocity, current_velocity, prev_velocity, v_total_error, dt):
        """
        Longitudinal controller using a simple PID control
        :param desired_velocity: The target velocity that we want to follow
        :param current_velocity: current forward velocity of the vehicle
        :param prev_velocity: previous forward velocity of the vehicle
        :param v_total_error:
        :param dt:
        :return:
        """

        vel_error = desired_velocity - current_velocity
        v_total_error_new = v_total_error + vel_error * dt
        p = self.kp * vel_error
        i = self.ki * v_total_error_new
        d = self.kd * (current_velocity - prev_velocity) / dt
        tau = p + i + d

        if current_velocity <= 0.01:
            tau = abs(tau)

        return v_total_error_new, [tau, tau, 0, 0]


def main():
    print("This script is not meant to be executable, and should be used as a library.")


if __name__ == "__main__":
    main()
