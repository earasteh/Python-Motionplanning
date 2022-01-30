import os
import matplotlib.pyplot as plt
import numpy as np

from vehicle_model import VehicleModel
from vehicle_model import VehicleParameters
from matplotlib.animation import FuncAnimation
from libs.controllers.stanley_controller import StanleyController
from libs.controllers.stanley_controller import LongitudinalController
from libs.utils.car_description import Description
from libs.utils.env import world  # Importing road definition
from libs.motionplanner.local_planner import LocalPlanner
from libs.utils.plots import data_cleaning, plot_results

###
# Frame rate = 0.1
# Vehicle simulation time = 1e-4
# Controller time = 1e-2 - 1e-3
# Motion planner  = 3e-2 - 3e-3
###

Veh_SIM_NUM = 1000  # Number of times vehicle simulation (Simulation_resolution  = sim.dt/Veh_SIM_NUM)
Control_SIM_NUM = Veh_SIM_NUM / 10


# MP_SIM_NUM

class Simulation:

    def __init__(self):
        fps = 10.0

        self.frame_dt = 1 / fps
        self.veh_dt = self.frame_dt / Veh_SIM_NUM
        self.controller_dt = self.frame_dt / Control_SIM_NUM
        self.map_size = 40
        self.frames = 10
        self.loop = False


p = VehicleParameters()

## Motion planner constants
NUM_PATHS = 7
BP_LOOKAHEAD_BASE = 8.0  # m
BP_LOOKAHEAD_TIME = 2.0  # s
PATH_OFFSET = 1.5  # m
CIRCLE_OFFSETS = [-1.0, 1.0, 3.0]  # m
CIRCLE_RADII = [1.5, 1.5, 1.5]  # m
TIME_GAP = 1.0  # s
PATH_SELECT_WEIGHT = 10
A_MAX = 1.5  # m/s^2
SLOW_SPEED = 2.0  # m/s
STOP_LINE_BUFFER = 3.5  # m
LEAD_VEHICLE_LOOKAHEAD = 20.0  # m
LP_FREQUENCY_DIVISOR = 2  # Frequency divisor to make the

LOOKAHEAD = 10


# local planner operate at a lower
# frequency than the controller
# (which operates at the simulation
# frequency). Must be a natural
# number.

class Car:

    def __init__(self, init_x, init_y, init_yaw, px, py, pyaw, dt):
        # Variable to log all the data
        self.DataLog = np.zeros((Veh_SIM_NUM * 2500, 37))
        # Model parameters
        init_vel = 25.0
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.prev_vel = self.v = init_vel
        self.target_vel = 25.0
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
        self.k = 8.0
        self.ksoft = 1.0
        self.kyaw = 0.01
        self.ksteer = 0
        self.crosstrack_error = None
        self.target_id = None
        self.x_del = [0]

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
                                                 self.wheelbase,
                                                 self.px, self.py, self.pyaw)
        self.kbm = VehicleModel(self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)
        self.long_tracker = LongitudinalController(self.k_v, self.k_i, self.k_d)

    def drive(self, frame):
        for i in range(Veh_SIM_NUM):
            ## Motion Planner:
            # if i % 20 == 0:
            #     paths, best_index, best_path = self.local_motion_planner.MotionPlanner(self.px, self.py,
            #                                                                            self.target_vel,
            #                                                                            self.x, self.y, self.yaw, self.v,
            #                                                                            self.lateral_tracker)
            paths = 0
            best_index = 0
            best_path = 0
            ## Motion Controllers:
            ## git test
            if i % 10 == 0:
                self.delta, self.target_id, self.crosstrack_error = self.lateral_tracker.stanley_control(self.x, self.y,
                                                                                                         self.yaw,
                                                                                                         self.v,
                                                                                                         self.delta)
                self.total_vel_error, self.torque_vec = self.long_tracker.long_control(self.target_vel, self.v,
                                                                                       self.prev_vel,
                                                                                       self.total_vel_error, self.dt)
                self.prev_vel = self.v

                # Filter the delta output
                self.x_del.append((1 - 1e-4/0.001) * self.x_del[-1] + 1e-4/0.001 * self.delta)
                self.delta = self.x_del[-1]

            ## Vehicle model
            self.state, self.x, self.y, self.yaw, self.v, state_dot, outputs, self.ax_prev, self.ay_prev = \
                self.kbm.planar_model_RK4(self.state,self.torque_vec, [1.0, 1.0, 1.0, 1.0],
                                          [self.delta, self.delta, 0, 0], p, self.ax_prev, self.ay_prev)

            U, V, wz, wFL, wFR, wRL, wRR, yaw, x, y = self.state
            U_dot, V_dot, wz_dot, wFL_dot, wFR_dot, wRL_dot, wRR_dot, yaw_dot, x_dot, y_dot = state_dot
            fFLx, fFRx, fRLx, fRRx, fFLy, fFRy, fRLy, fRRy, fFLz, fFRz, fRLz, fRRz, sFL, sFR, sRL, sRR, fFLxt, fFLyt = outputs

            self.DataLog[frame * Veh_SIM_NUM + i, :] = [(frame * Veh_SIM_NUM + i) * self.kbm.dt, U, V, wz,
                                                        wFL, wFR, wRL, wRR, yaw, x, y, self.delta,
                                                        self.torque_vec[0], self.torque_vec[1], self.torque_vec[2],
                                                        self.torque_vec[3],
                                                        sFL, sFR, sRL, sRR, fFLx, fFRx, fRLx, fRRx, fFLy, fFRy, fRLy,
                                                        fRRy,
                                                        fFLz,
                                                        fFRz, fRLz, fRRz, yaw_dot, V_dot, self.crosstrack_error,
                                                        fFLxt, fFLyt]

        os.system('cls' if os.name == 'nt' else 'clear')
        return paths, best_index, best_path


def main():
    sim = Simulation()
    path = world.path

    car = Car(path.px[1800], path.py[1800], path.pyaw[1800], path.px, path.py, path.pyaw, sim.veh_dt)
    desc = Description(car.overall_length, car.overall_width, car.rear_overhang, car.tyre_diameter, car.tyre_width,
                       car.axle_track, car.wheelbase)

    interval = sim.frame_dt * 10 ** -8  # * 10 ** 2

    fig = plt.figure()
    ax = plt.axes()
    ax.set_aspect('equal')

    # road = plt.Circle((0, 0), 50, color='gray', fill=False, linewidth=30)
    _ = plt.fill(np.append(world.bound_xr, world.bound_xl[::-1]), np.append(world.bound_yr, world.bound_yl[::-1]),
                 color='gray')
    ax.plot(world.bound_xl, world.bound_yl, color='black')
    ax.plot(world.bound_xr, world.bound_yr, color='black')
    _ = plt.fill(world.obstacle_x, world.obstacle_y, color='red')

    ax.plot(path.px, path.py, '--', color='gold')

    annotation = ax.annotate(f'{car.x:.1f}, {car.y:.1f}', xy=(car.x, car.y + 5), color='black', annotation_clip=False)
    target, = ax.plot([], [], '+r')

    CLP1, = ax.plot([], [], 'k-.')
    CLP2, = ax.plot([], [], 'k-.')
    CLP3, = ax.plot([], [], 'k-.')
    CLP4, = ax.plot([], [], 'k-.')
    CLP5, = ax.plot([], [], 'k-.')
    CLP_best, = ax.plot([], [], 'g-.')

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
        paths, best_index, best_path = car.drive(frame)
        # paths = car.drive(frame)
        paths = np.array(paths)

        outline_plot, fr_plot, rr_plot, fl_plot, rl_plot = desc.plot_car(car.x, car.y, car.yaw, car.delta)
        outline.set_data(*outline_plot)
        fr.set_data(*fr_plot)
        rr.set_data(*rr_plot)
        fl.set_data(*fl_plot)
        rl.set_data(*rl_plot)
        rear_axle.set_data(car.x, car.y)
        # Show car's target
        target.set_data(path.px[car.target_id], path.py[car.target_id])

        try:
            CLP1.set_data(paths[-2, 0, :], paths[-2, 1, :])
            CLP2.set_data(paths[-1, 0, :], paths[-1, 1, :])
            CLP3.set_data(paths[0, 0, :], paths[0, 1, :])
            CLP4.set_data(paths[1, 0, :], paths[1, 1, :])
            CLP5.set_data(paths[2, 0, :], paths[2, 1, :])
            CLP_best.set_data(paths[best_index, 0, :], paths[best_index, 1, :])
        except IndexError:
            CLP1.set_data([0, 0, 0], [0, 0, 0])
            CLP2.set_data([0, 0, 0], [0, 0, 0])
            CLP3.set_data([0, 0, 0], [0, 0, 0])
            CLP4.set_data([0, 0, 0], [0, 0, 0])
            CLP5.set_data([0, 0, 0], [0, 0, 0])
            CLP_best.set_data([0, 0, 0], [0, 0, 0])

        # Annotate car's coordinate above car
        annotation.set_text(f'{car.x:.1f}, {car.y:.1f}')
        annotation.set_position((car.x, car.y + 5))
        plt.title(f'{sim.frame_dt * frame:.2f}s', loc='right')
        plt.xlabel(f'Speed: {car.v:.2f} m/s', loc='left')
        # plt.savefig(f'fig/visualisation_{frame:04}.png')

        return outline, fr, rr, fl, rl, rear_axle, target, CLP1, CLP2, CLP3, CLP4, CLP5, CLP_best
        # return outline, fr, rr, fl, rl, rear_axle, target,

    _ = FuncAnimation(fig, animate, frames=sim.frames, interval=interval, repeat=sim.loop)
    # anim.save('resources/animation.gif', fps=100)   #Uncomment to save the animation
    plt.show()

    plot_results(data_cleaning(car.DataLog))

if __name__ == '__main__':
    main()
