import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from libs.utils.env import world

"""
This file cleans and labels the data and then plots all the results
"""


# U, V, wz, wFL, wFR, wRL, wRR, yaw, x, y = self.state
# U_dot, V_dot, wz_dot, wFL_dot, wFR_dot, wRL_dot, wRR_dot, yaw_dot, x_dot, y_dot = state_dot
# fFLx, fFRx, fRLx, fRRx, fFLy, fFRy, fRLy, fRRy, fFLz, fFRz, fRLz, fRRz, sFL, sFR, sRL, sRR, fFLxt, fFLyt = outputs

def data_cleaning(DataLog):
    DataLog_nz = DataLog[~np.all(DataLog == 0, axis=1)]  # only plot the rows that are not zeros
    DataLog_pd = pd.DataFrame(DataLog_nz, columns=['time',
                                                   'U', 'V', 'wz', 'wFL', 'wFR', 'wRL', 'wRR', 'yaw', 'x', 'y',
                                                   'U_dot', 'V_dot', 'wz_dot', 'wFL_dot', 'wFR_dot', 'wRL_dot',
                                                   'wRR_dot', 'yaw_dot', 'x_dot', 'y_dot',
                                                   'delta', 'tau_FL', 'tau_FR', 'tau_RL', 'tau_RR',
                                                   'Fx_FL', 'Fx_FR', 'Fx_RL', 'Fx_RR',
                                                   'Fy_FL', 'Fy_FR', 'Fy_RL', 'Fy_RR',
                                                   'Fz_FL', 'Fz_FR', 'Fz_RL', 'Fz_RR',
                                                   'sFL', 'sFR', 'sRL', 'sRR', 'Fxt_FL', 'Fyt_FL', 'crosstrack'])
    return DataLog_pd


def plot_results(DataLog_pd):
    plt.figure()
    plt.title('Forward Velocity')
    plt.plot(DataLog_pd['time'], DataLog_pd['U'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Forward Velocity (m/s)')
    plt.savefig('results/u.png')

    plt.figure()
    fig_name = 'Lateral Velocity'
    plt.title(fig_name)
    plt.plot(DataLog_pd['time'], DataLog_pd['V'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Lateral Velocity (m/s)')
    plt.savefig('results/'+fig_name+'.png')

    plt.figure()
    fig_name = 'Yaw rate'
    plt.title(fig_name)
    plt.plot(DataLog_pd['time'], DataLog_pd['yaw_dot'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Yaw rate (rad/sec)')
    plt.savefig('results/'+fig_name+'.png')

    plt.figure()
    fig_name = 'Lateral Acceleration'
    plt.title(fig_name)
    plt.plot(DataLog_pd['time'], DataLog_pd['V_dot'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Lateral Acceleration (m/s^2)')

    # plt.figure()
    # plt.title('Longitudinal Tire Forces')
    # plt.plot(DataLog_pd['time'], DataLog_pd['Fx_FL'])
    # plt.plot(DataLog_pd['time'], DataLog_pd['Fx_FR'])
    # plt.plot(DataLog_pd['time'], DataLog_pd['Fx_RL'])
    # plt.plot(DataLog_pd['time'], DataLog_pd['Fx_RR'])
    # plt.legend(['FL', 'FR', 'RL', 'RR'])
    # plt.xlabel('Time (Sec.)')
    # plt.ylabel('Force (N)')

    plt.figure()
    fig_name ='Lateral Tire Forces'
    plt.title(fig_name)
    plt.plot(DataLog_pd['time'], DataLog_pd['Fy_FL'])
    plt.plot(DataLog_pd['time'], DataLog_pd['Fy_FR'])
    plt.plot(DataLog_pd['time'], DataLog_pd['Fy_RL'])
    plt.plot(DataLog_pd['time'], DataLog_pd['Fy_RR'])
    plt.legend(['FL', 'FR', 'RL', 'RR'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Force (N)')
    plt.savefig('results/'+fig_name+'.png')

    plt.figure()
    fig_name ='Normal Tire Forces'
    plt.title(fig_name)
    plt.plot(DataLog_pd['time'], DataLog_pd['Fz_FL'])
    plt.plot(DataLog_pd['time'], DataLog_pd['Fz_FR'])
    plt.plot(DataLog_pd['time'], DataLog_pd['Fz_RL'])
    plt.plot(DataLog_pd['time'], DataLog_pd['Fz_RR'])
    plt.legend(['FL', 'FR', 'RL', 'RR'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Force (N)')
    plt.savefig('results/'+fig_name+'.png')

    # plt.figure()
    # plt.title('Torque at each wheel')
    # plt.subplot(2, 2, 1)
    # plt.plot(DataLog_pd['time'], DataLog_pd['tau_FL'])
    # plt.legend(['FL'])
    # plt.xlabel('Time (Sec.)')
    # plt.ylabel('Wheel Torque (N.m)')
    # plt.subplot(2, 2, 2)
    # plt.plot(DataLog_pd['time'], DataLog_pd['tau_FR'])
    # plt.legend(['FR'])
    # plt.xlabel('Time (Sec.)')
    # plt.ylabel('Wheel Torque (N.m)')
    # plt.subplot(2, 2, 3)
    # plt.plot(DataLog_pd['time'], DataLog_pd['tau_RL'])
    # plt.legend(['RL'])
    # plt.xlabel('Time (Sec.)')
    # plt.ylabel('Wheel Torque (N.m)')
    # plt.subplot(2, 2, 4)
    # plt.plot(DataLog_pd['time'], DataLog_pd['tau_RR'])
    # plt.legend(['RR'])
    # plt.xlabel('Time (Sec.)')
    # plt.ylabel('Wheel Torque (N.m)')
    #
    # plt.figure()
    # plt.title('Angular velocity at each wheel')
    # plt.subplot(2, 2, 1)
    # plt.plot(DataLog_pd['time'], DataLog_pd['wFL'])
    # plt.legend(['FL'])
    # plt.xlabel('Time (Sec.)')
    # plt.ylabel('Angular Vel. (rad/Sec)')
    # plt.subplot(2, 2, 2)
    # plt.plot(DataLog_pd['time'], DataLog_pd['wFR'])
    # plt.legend(['FR'])
    # plt.xlabel('Time (Sec.)')
    # plt.ylabel('Angular Vel. (rad/Sec)')
    # plt.subplot(2, 2, 3)
    # plt.plot(DataLog_pd['time'], DataLog_pd['wRL'])
    # plt.legend(['RL'])
    # plt.xlabel('Time (Sec.)')
    # plt.ylabel('Angular Vel. (rad/Sec)')
    # plt.subplot(2, 2, 4)
    # plt.plot(DataLog_pd['time'], DataLog_pd['wRR'])
    # plt.legend(['RR'])
    # plt.xlabel('Time (Sec.)')
    # plt.ylabel('Angular Vel. (rad/Sec)')

    plt.figure()
    fig_name ='Steering Angle'
    plt.title(fig_name)
    plt.plot(DataLog_pd['time'], DataLog_pd['delta'] * 180 / np.pi)
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Steering angle (deg)')
    plt.savefig('results/'+fig_name+'.png')

    fig, ax = plt.subplots()
    ax.set_xlim(0.1, 5)
    fig_name ='Cross Track Error'
    plt.title(fig_name)
    plt.plot(DataLog_pd['time'], DataLog_pd['crosstrack'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Cross Track Error (m)')
    plt.savefig('results/'+fig_name+'.png')

    plt.figure()
    fig_name = 'Combined slip'
    plt.title(fig_name)
    plt.subplot(2, 2, 1)
    plt.plot(DataLog_pd['time'], DataLog_pd['sFL'])
    plt.legend(['FL'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Slip')
    plt.subplot(2, 2, 2)
    plt.plot(DataLog_pd['time'], DataLog_pd['sFR'])
    plt.legend(['FR'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Slip')
    plt.subplot(2, 2, 3)
    plt.plot(DataLog_pd['time'], DataLog_pd['sRL'])
    plt.legend(['RL'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Slip')
    plt.subplot(2, 2, 4)
    plt.plot(DataLog_pd['time'], DataLog_pd['sRR'])
    plt.legend(['RR'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Slip')
    plt.savefig('results/'+fig_name+'.png')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.grid()
    fig_name ='Trajectory of the vehicle'
    plt.title(fig_name)
    plt.plot(DataLog_pd['x'], DataLog_pd['y'], color='green')
    plt.plot(world.xm, world.ym, 'k--')
    plt.fill(world.obstacle_x, world.obstacle_y, color='red', zorder=2)
    ax.set_xlim(-10, 100)
    ax.set_ylim(-5, 25)
    ax.set_aspect('equal')
    ax.legend(['Generated Path', 'Ref'])
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.savefig('results/'+fig_name+'.png')

    plt.show()
