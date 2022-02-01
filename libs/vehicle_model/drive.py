import os
import numpy as np
from libs.vehicle_model.vehicle_model import VehicleModel
from libs.vehicle_model.vehicle_model import VehicleParameters
from libs.controllers.stanley_controller import StanleyController
from libs.controllers.stanley_controller import LongitudinalController
from libs.motionplanner.local_planner import LocalPlanner
from libs.utils.env import world

###
# Frame rate = 0.1
# Vehicle simulation time = 1e-4
# Controller time = 1e-2 - 1e-3
# Motion planner  = 3e-2 - 3e-3
###

Veh_SIM_NUM = 100  # Number of times vehicle simulation (Simulation_resolution  = sim.dt/Veh_SIM_NUM)
Control_SIM_NUM = Veh_SIM_NUM / 10

## Motion planner constants
NUM_PATHS = 7
BP_LOOKAHEAD_BASE = 8.0  # m
BP_LOOKAHEAD_TIME = 2.0  # s
PATH_OFFSET = 1.6  # m
CIRCLE_OFFSETS = [-1.0, 1.0, 3.0]  # m
CIRCLE_RADII = [1.5, 1.5, 1.5]  # m
TIME_GAP = 1.0  # s
PATH_SELECT_WEIGHT = 10
A_MAX = 1.5  # m/s^2
SLOW_SPEED = 2.0  # m/s
STOP_LINE_BUFFER = 3.5  # m
LEAD_VEHICLE_LOOKAHEAD = 20.0  # m
LP_FREQUENCY_DIVISOR = 2  # Frequency divisor to make the local planner operate at a lower
# frequency than the controller (which operates at the simulation frequency). Must be a natural number.
LOOKAHEAD = 30

p = VehicleParameters()


class Car:

    def __init__(self, init_x, init_y, init_yaw, px, py, pyaw, dt):
        # Variable to log all the data
        self.DataLog = np.zeros((Veh_SIM_NUM * 2500, 45))
        # Model parameters
        init_vel = 20.0
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.prev_vel = self.v = init_vel
        self.target_vel = 20.0
        self.total_vel_error = 0
        self.delta = 0.0
        self.omega = 0.0
        self.wheelbase = 2.906
        self.max_steer = np.deg2rad(30)
        self.dt = dt
        self.c_r = 0.01
        self.c_a = 2.0
        self.ax_prev = 0
        self.ay_prev = 0

        # self.state = [init_vel, 0, 0, init_yaw, init_x, init_y] (these were for the bicycle model states)
        self.state = [init_vel, 0, 0, init_vel / p.rw, init_vel / p.rw, init_vel / p.rw, init_vel / p.rw, init_yaw,
                      init_x, init_y]

        # Lateral Tracker parameters
        self.px = px
        self.py = py
        self.pyaw = pyaw
        self.k = 8
        self.ksoft = 1.0
        self.kyaw = 0
        self.ksteer = 0
        self.crosstrack_error = None
        self.target_id = None
        self.x_del = [0]
        self._prev_paths = 0

        # Longitudinal Tracker parameters
        self.k_v = 1000
        self.k_i = 100
        self.k_d = 0
        self.torque_vec = [0, 0, 0, 0]

        # Description parameters
        self.overall_length = 4.97
        self.overall_width = 1.964
        self.tyre_diameter = 0.4826
        self.tyre_width = 0.2032
        self.axle_track = 1.662
        self.rear_overhang = (self.overall_length - self.wheelbase) / 2
        self.colour = 'black'

        self.local_motion_planner = LocalPlanner(LOOKAHEAD,
                                                 NUM_PATHS,
                                                 PATH_OFFSET,
                                                 CIRCLE_OFFSETS,
                                                 CIRCLE_RADII,
                                                 PATH_SELECT_WEIGHT,
                                                 TIME_GAP,
                                                 A_MAX,
                                                 SLOW_SPEED,
                                                 STOP_LINE_BUFFER)
        self.lateral_tracker = StanleyController(self.k, self.ksoft, self.kyaw, self.ksteer, self.max_steer,
                                                 self.wheelbase)
        self.kbm = VehicleModel(self.wheelbase, self.max_steer, self.dt)
        self.long_tracker = LongitudinalController(self.k_v, self.k_i, self.k_d)

    def drive(self, frame):
        for i in range(Veh_SIM_NUM):
            # Motion Planner:
            if i % 40 == 0:
                paths, best_index, best_path = \
                    self.local_motion_planner.MotionPlanner(self.px, self.py, self.target_vel,
                                                            self.x, self.y, self.yaw, self.v,
                                                            self.lateral_tracker, world.obstacle_xy)
                self._prev_paths = paths
                if paths is None:
                    paths = self._prev_paths
                    best_index = 0
                    best_path = 0
            ## Motion Controllers:
            if i % 10 == 0:
                self.delta, self.target_id, self.crosstrack_error = \
                    self.lateral_tracker.stanley_control(self.x, self.y, self.yaw, self.v)
                self.total_vel_error, self.torque_vec = \
                    self.long_tracker.long_control(self.target_vel, self.v, self.prev_vel,
                                                   self.total_vel_error, self.dt)
                self.prev_vel = self.v

                # Filter the delta output
                self.x_del.append((1 - 1e-4 / (2*0.001)) * self.x_del[-1] + 1e-4 / (2*0.001) * self.delta)
                self.delta = self.x_del[-1]

            ## Vehicle model
            self.state, self.x, self.y, self.yaw, self.v, state_dot, outputs, self.ax_prev, self.ay_prev = \
                self.kbm.planar_model_RK4(self.state, self.torque_vec, [1.0, 1.0, 1.0, 1.0],
                                          [self.delta, self.delta, 0, 0], p, self.ax_prev, self.ay_prev)

            self.DataLog[frame * Veh_SIM_NUM + i, 0] = (frame * Veh_SIM_NUM + i) * self.kbm.dt
            self.DataLog[frame * Veh_SIM_NUM + i, 1:11] = self.state
            self.DataLog[frame * Veh_SIM_NUM + i, 11:21] = state_dot
            self.DataLog[frame * Veh_SIM_NUM + i, 21] = self.delta
            self.DataLog[frame * Veh_SIM_NUM + i, 22:26] = self.torque_vec
            self.DataLog[frame * Veh_SIM_NUM + i, 26:44] = outputs
            self.DataLog[frame * Veh_SIM_NUM + i, 44] = self.crosstrack_error

        os.system('cls' if os.name == 'nt' else 'clear')
        return paths, best_index, best_path
