import collections
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rand

from vehicle_model import VehicleModel
from vehicle_model import VehicleParameters
from matplotlib.animation import FuncAnimation
from libs.stanley_controller import StanleyController
from libs.stanley_controller import LongitudinalController
from libs.car_description import Description
from libs.cubic_spline_interpolator import generate_cubic_spline
from env import world  # Importing road definition
from motionplanner.local_planner_ehsan import LocalPlanner, get_closest_index, motionplanner_datatranslation, transform_paths

class Simulation:

    def __init__(self):
        fps = 100.0

        self.dt = 1 / fps
        self.map_size = 40
        self.frames = 250
        self.loop = False

log_time = [0]
log_U = []
log_V = []
log_wz = []
log_wFL = []
log_wFR = []
log_wRL = []
log_wRR = []
log_yaw = []
log_x = []
log_y = []
log_delta = []
log_tau_FL = []
log_tau_FR = []
log_tau_RL = []
log_tau_RR = []
log_sFL = []
log_sFR = []
log_sRL = []
log_sRR = []
log_Fx_FL = []
log_Fx_FR = []
log_Fx_RL = []
log_Fx_RR = []
log_Fy_FL = []
log_Fy_FR = []
log_Fy_RL = []
log_Fy_RR = []
log_Fz_FL = []
log_Fz_FR = []
log_Fz_RL = []
log_Fz_RR = []

log_yaw_rate = []
log_latac = []

p = VehicleParameters()


class Car:

    def __init__(self, init_x, init_y, init_yaw, px, py, pyaw, dt):
        # Model parameters
        init_vel = 15.0
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.prev_vel = self.v = init_vel
        self.target_vel = 15.0
        self.total_vel_error = 0
        self.delta = 0.0
        self.omega = 0.0
        self.wheelbase = 2.906
        self.max_steer = np.deg2rad(30)
        self.dt = dt
        self.c_r = 0.01
        self.c_a = 2.0

        # self.state = [init_vel, 0, 0, init_yaw, init_x, init_y] (these were for the bicycle model states)
        self.state = [init_vel, 0, 0, init_vel / p.rw, init_vel / p.rw, init_vel / p.rw, init_vel / p.rw, init_yaw,
                      init_x, init_y]

        # Lateral Tracker parameters
        self.px = px
        self.py = py
        self.pyaw = pyaw
        self.k = 8.0
        self.ksoft = 1.0
        self.kyaw = 0.01
        self.ksteer = 0
        self.crosstrack_error = None
        self.target_id = None

        # Longitudinal Tracker parameters
        self.k_v = 1000
        self.k_i = 100
        self.k_d = 0

        # Description parameters
        self.overall_length = 4.97
        self.overall_width = 1.964
        self.tyre_diameter = 0.4826
        self.tyre_width = 0.2032
        self.axle_track = 1.662
        self.rear_overhang = (self.overall_length - self.wheelbase) / 2
        self.colour = 'black'

        self.local_motion_planner = LocalPlanner(10, 5, 1.5, 1, 1, 1, 1, 1, 1, 1)
        self.lateral_tracker = StanleyController(self.k, self.ksoft, self.kyaw, self.ksteer, self.max_steer,
                                                 self.wheelbase,
                                                 self.px, self.py, self.pyaw)
        self.kbm = VehicleModel(self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)
        self.long_tracker = LongitudinalController(self.k_v, self.k_i, self.k_d)

    def drive(self):
        # Motion Planner:
        waypoints, ego_state = motionplanner_datatranslation(self.px, self.py, self.target_vel,
                                                                         self.x, self.y, self.yaw, self.v)
        closest_len, closest_index = get_closest_index(waypoints, ego_state)
        goal_index = self.local_motion_planner.get_goal_index(waypoints, ego_state, closest_len, closest_index)
        goal_state = waypoints[goal_index]
        goal_state_set = self.local_motion_planner.get_goal_state_set(goal_index, goal_state, waypoints, ego_state)
        paths, path_validity = self.local_motion_planner.plan_paths(goal_state_set)
        paths = transform_paths(paths, ego_state)

        # Motion Controllers:
        self.delta, self.target_id, self.crosstrack_error = self.lateral_tracker.stanley_control(self.x, self.y,
                                                                                                 self.yaw,
                                                                                                 self.v, self.delta)
        self.total_vel_error, torque_vec = self.long_tracker.long_control(self.target_vel, self.v, self.prev_vel,
                                                                          self.total_vel_error, self.dt)
        self.prev_vel = self.v
        # Vehicle model:
        for i in range(10):
            state_dot, _, _, _, _, outputs = self.kbm.planar_model(self.state, torque_vec, [1.0, 1.0, 1.0, 1.0],
                                                                   [self.delta, self.delta, 0, 0], p)
            self.state, self.x, self.y, self.yaw, self.v = self.kbm.planar_model_RK4(self.state, torque_vec,
                                                                                     [1.0, 1.0, 1.0, 1.0],
                                                                                     [self.delta, self.delta, 0, 0], p)

            U, V, wz, wFL, wFR, wRL, wRR, yaw, x, y = self.state
            U_dot, V_dot, wz_dot, wFL_dot, wFR_dot, wRL_dot, wRR_dot, yaw_dot, x_dot, y_dot = state_dot
            fFLx, fFRx, fRLx, fRRx, fFLy, fFRy, fRLy, fRRy, fFLz, fFRz, fRLz, fRRz, sFL, sFR, sRL, sRR = outputs

            log_time.append(log_time[-1] + self.kbm.dt)

            log_U.append(U)
            log_V.append(V)
            log_wz.append(wz)

            log_wFL.append(wFL)
            log_wFR.append(wFR)
            log_wRL.append(wRL)
            log_wRR.append(wRR)

            log_yaw.append(yaw)
            log_x.append(x)
            log_y.append(y)

            log_tau_FL.append(torque_vec[0])
            log_tau_FR.append(torque_vec[1])
            log_tau_RL.append(torque_vec[2])
            log_tau_RR.append(torque_vec[3])

            log_Fx_FL.append(fFLx)
            log_Fx_FR.append(fFRx)
            log_Fx_RL.append(fRLx)
            log_Fx_RR.append(fRRx)

            log_Fy_FL.append(fFLy)
            log_Fy_FR.append(fFRy)
            log_Fy_RL.append(fRLy)
            log_Fy_RR.append(fRRy)

            log_Fz_FL.append(fFLz)
            log_Fz_FR.append(fFRz)
            log_Fz_RL.append(fRLz)
            log_Fz_RR.append(fRRz)

            log_sFL.append(sFL)
            log_sFR.append(sFR)
            log_sRL.append(sRL)
            log_sRR.append(sRR)

            log_yaw_rate.append(yaw_dot)
            log_latac.append(V_dot)

            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Cross-track term: {self.crosstrack_error}")
        return paths


def main():
    sim = Simulation()
    path = world.path

    car = Car(path.px[0], path.py[0], path.pyaw[0], path.px, path.py, path.pyaw, sim.dt)
    desc = Description(car.overall_length, car.overall_width, car.rear_overhang, car.tyre_diameter, car.tyre_width,
                       car.axle_track, car.wheelbase)

    interval = sim.dt * 10 ** 2

    fig = plt.figure()
    ax = plt.axes()
    ax.set_aspect('equal')

    # road = plt.Circle((0, 0), 50, color='gray', fill=False, linewidth=30)
    road = plt.fill(np.append(world.bound_xr, world.bound_xl[::-1]), np.append(world.bound_yr, world.bound_yl[::-1]),
                    color='gray')
    ax.plot(world.bound_xl, world.bound_yl, color='black')
    ax.plot(world.bound_xr, world.bound_yr, color='black')
    ax.plot(path.px, path.py, '--', color='gold')

    annotation = ax.annotate(f'{car.x:.1f}, {car.y:.1f}', xy=(car.x, car.y + 5), color='black', annotation_clip=False)
    target, = ax.plot([], [], '+r')
    CLP1, = ax.plot([], [], 'g-.', color='green')
    CLP2, = ax.plot([], [], 'g-.', color='green')
    CLP3, = ax.plot([], [], 'g-.', color='green')
    CLP4, = ax.plot([], [], 'g-.', color='green')
    CLP5, = ax.plot([], [], 'g-.', color='green')

    outline, = ax.plot([], [], color=car.colour)
    fr, = ax.plot([], [], color=car.colour)
    rr, = ax.plot([], [], color=car.colour)
    fl, = ax.plot([], [], color=car.colour)
    rl, = ax.plot([], [], color=car.colour)
    rear_axle, = ax.plot(car.x, car.y, '+', color=car.colour, markersize=2)

    plt.grid()


    def animate(frame):
        # Camera tracks car
        ax.set_xlim(car.x - sim.map_size, car.x + sim.map_size)
        ax.set_ylim(car.y - sim.map_size, car.y + sim.map_size)

        # Drive and draw car
        paths = car.drive()
        paths = np.array(paths)
        # print(paths[0, :, :])

        outline_plot, fr_plot, rr_plot, fl_plot, rl_plot = desc.plot_car(car.x, car.y, car.yaw, car.delta)
        outline.set_data(*outline_plot)
        fr.set_data(*fr_plot)
        rr.set_data(*rr_plot)
        fl.set_data(*fl_plot)
        rl.set_data(*rl_plot)
        rear_axle.set_data(car.x, car.y)
        # Show car's target
        target.set_data(path.px[car.target_id], path.py[car.target_id])

        CLP1.set_data(paths[-2, 0, :], paths[-2, 1, :])
        CLP2.set_data(paths[-1, 0, :], paths[-1, 1, :])
        CLP3.set_data(paths[0, 0, :], paths[0, 1, :])
        CLP4.set_data(paths[1, 0, :], paths[1, 1, :])
        CLP5.set_data(paths[2, 0, :], paths[2, 1, :])


        # Annotate car's coordinate above car
        annotation.set_text(f'{car.x:.1f}, {car.y:.1f}')
        annotation.set_position((car.x, car.y + 5))
        plt.title(f'{sim.dt * frame:.2f}s', loc='right')
        plt.xlabel(f'Speed: {car.v:.2f} m/s', loc='left')
        # plt.savefig(f'fig/visualisation_{frame:04}.png')

        return outline, fr, rr, fl, rl, rear_axle, target, CLP1, CLP2, CLP3, CLP4, CLP5

    _ = FuncAnimation(fig, animate, frames=sim.frames, interval=interval, repeat=sim.loop)
    # anim.save('resources/animation.gif', fps=100)   #Uncomment to save the animation
    plt.show()

    plt.figure()
    plt.title('Forward Velocity')
    plt.plot(log_time[:-1], log_U)
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Forward Velocity (m/s)')

    plt.figure()
    plt.title('Lateral Velocity')
    plt.plot(log_time[:-1], log_V)
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Lateral Velocity (m/s)')

    plt.figure()
    plt.title('Yaw rate')
    plt.plot(log_time[:-1], log_yaw_rate)
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Yaw rate (rad/sec)')

    plt.figure()
    plt.title('Lateral Acceleration')
    plt.plot(log_time[:-1], log_latac)
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Lateral Acceleration (m/s^2)')

    # plt.figure()
    # plt.title('FxFL')
    # plt.plot(log_time[:-1], log_Fx_FL)

    plt.figure()
    plt.title('Normal Forces')
    plt.plot(log_time[:-1], log_Fz_FL)
    plt.plot(log_time[:-1], log_Fz_FR)
    plt.plot(log_time[:-1], log_Fz_RL)
    plt.plot(log_time[:-1], log_Fz_RR)
    plt.legend(['FL', 'FR', 'RL', 'RR'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Normal Force (N)')

    plt.figure()
    plt.title('Torque at each wheel')
    plt.plot(log_time[:-1], log_tau_FL)
    plt.plot(log_time[:-1], log_tau_FR)
    plt.plot(log_time[:-1], log_tau_RL)
    plt.plot(log_time[:-1], log_tau_RR)
    plt.legend(['FL', 'FR', 'RL', 'RR'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Wheel Torque (N.m)')

    # plt.show()

if __name__ == '__main__':
    main()
