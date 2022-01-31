import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from libs.utils.car_description import Description
from libs.utils.env import world  # Importing road definition
from libs.vehicle_model.drive import Veh_SIM_NUM, Control_SIM_NUM, Car, NUM_PATHS
from libs.utils.plots import plot_results, data_cleaning

class Simulation:

    def __init__(self):
        fps = 10.0

        self.frame_dt = 1 / fps
        self.veh_dt = self.frame_dt / Veh_SIM_NUM
        self.controller_dt = self.frame_dt / Control_SIM_NUM
        self.map_size = 40
        self.frames = 100
        self.loop = False

def main():
    sim = Simulation()
    path = world.path

    car = Car(path.px[800], path.py[800], path.pyaw[800], path.px, path.py, path.pyaw, sim.veh_dt)
    desc = Description(car.overall_length, car.overall_width, car.rear_overhang, car.tyre_diameter, car.tyre_width,
                       car.axle_track, car.wheelbase)

    interval = sim.frame_dt * 10 ** -8  # * 10 ** 2

    fig = plt.figure()
    ax = plt.axes()
    ax.set_aspect('equal')

    _ = plt.fill(np.append(world.bound_xr, world.bound_xl[::-1]), np.append(world.bound_yr, world.bound_yl[::-1]),
                 color='gray')
    ax.plot(world.bound_xl, world.bound_yl, color='black')
    ax.plot(world.bound_xr, world.bound_yr, color='black')
    _ = plt.fill(world.obstacle_x, world.obstacle_y, color='red')

    ax.plot(path.px, path.py, '--', color='gold')

    annotation = ax.annotate(f'{car.x:.1f}, {car.y:.1f}', xy=(car.x, car.y + 5), color='black', annotation_clip=False)

    CLP_ax, = ax.plot([], [], 'k-.')
    CLP_best, = ax.plot([], [], 'g-.')

    CLP_axes = [CLP_ax] * NUM_PATHS

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

        try:
            for i in range(NUM_PATHS):
                if i == best_index:
                    CLP_best.set_data(paths[i, 0, :], paths[i, 1, :])
                else:
                    CLP_axes[i].set_data(paths[i, 0, :], paths[i, 1, :])
            # CLP1.set_data(paths[-3, 0, :], paths[-3, 1, :])
            # CLP2.set_data(paths[-2, 0, :], paths[-2, 1, :])
            # CLP3.set_data(paths[-1, 0, :], paths[-1, 1, :])
            # CLP4.set_data(paths[0, 0, :], paths[0, 1, :])
            # CLP5.set_data(paths[1, 0, :], paths[1, 1, :])
            # CLP6.set_data(paths[2, 0, :], paths[2, 1, :])
            # CLP7.set_data(paths[3, 0, :], paths[3, 1, :])
        except IndexError:
            # CLP1.set_data([0, 0, 0], [0, 0, 0])
            # CLP2.set_data([0, 0, 0], [0, 0, 0])
            # CLP3.set_data([0, 0, 0], [0, 0, 0])
            # CLP4.set_data([0, 0, 0], [0, 0, 0])
            # CLP5.set_data([0, 0, 0], [0, 0, 0])
            # CLP6.set_data(paths[2, 0, :], paths[2, 1, :])
            # CLP7.set_data(paths[3, 0, :], paths[3, 1, :])
            CLP_best.set_data([0, 0, 0], [0, 0, 0])

        CLP1, CLP2, CLP3, CLP4, CLP5, CLP6, CLP7 = CLP_axes
        # Annotate car's coordinate above car
        annotation.set_text(f'{car.x:.1f}, {car.y:.1f}')
        annotation.set_position((car.x, car.y + 5))
        plt.title(f'{sim.frame_dt * frame:.2f}s', loc='right')
        plt.xlabel(f'Speed: {car.v:.2f} m/s', loc='left')

        return outline, fr, rr, fl, rl, rear_axle, CLP1, CLP2, CLP3, CLP4, CLP5, CLP6, CLP7, CLP_best

    _ = FuncAnimation(fig, animate, frames=sim.frames, interval=interval, repeat=sim.loop)
    # anim.save('resources/animation.gif', fps=10)   #Uncomment to save the animation
    plt.show()

    plot_results(data_cleaning(car.DataLog))

if __name__ == '__main__':
    main()
