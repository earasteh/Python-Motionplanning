#!/usr/bin/env python
from numpy import cos, sin, tan, clip, abs, sqrt, arctan, pi
from libs.normalise_angle import normalise_angle
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class KinematicBicycleModel:

    def __init__(self, wheelbase=1.0, max_steer=0.7, dt=0.05, c_r=0.0, c_a=0.0):
        """
        2D Kinematic Bicycle Model

        At initialisation
        :param wheelbase:       (float) vehicle's wheelbase [m]
        :param max_steer:       (float) vehicle's steering limits [rad]
        :param dt:              (float) discrete time period [s]
        :param c_r:             (float) vehicle's coefficient of resistance 
        :param c_a:             (float) vehicle's aerodynamic coefficient
    
        At every time step  
        :param x:               (float) vehicle's x-coordinate [m]
        :param y:               (float) vehicle's y-coordinate [m]
        :param yaw:             (float) vehicle's heading [rad]
        :param velocity:        (float) vehicle's velocity in the x-axis [m/s]
        :param throttle:        (float) vehicle's accleration [m/s^2]
        :param delta:           (float) vehicle's steering angle [rad]
    
        :return x:              (float) vehicle's x-coordinate [m]
        :return y:              (float) vehicle's y-coordinate [m]
        :return yaw:            (float) vehicle's heading [rad]
        :return velocity:       (float) vehicle's velocity in the x-axis [m/s]
        :return delta:          (float) vehicle's steering angle [rad]
        :return omega:          (float) vehicle's angular velocity [rad/s]
        """
        self.dt = dt
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.c_r = c_r
        self.c_a = c_a

    def kinematic_model(self, x, y, yaw, velocity, throttle, delta):
        # Compute the local velocity in the x-axis
        f_load = velocity * (self.c_r + self.c_a * velocity)
        velocity += self.dt * (throttle - f_load)

        # Compute the radius and angular velocity of the kinematic bicycle model
        delta = clip(delta, -self.max_steer, self.max_steer)

        # Compute the state change rate
        x_dot = velocity * cos(yaw)
        y_dot = velocity * sin(yaw)
        omega = velocity * tan(delta) / self.wheelbase

        # Compute the final state using the discrete time model
        x += x_dot * self.dt
        y += y_dot * self.dt
        yaw += omega * self.dt
        yaw = normalise_angle(yaw)

        return x, y, yaw, velocity, delta, omega


class VehicleParameters:
    def __init__(self, mf=987.89, mr=869.93, mus=50, L=7 * 0.3048, ab_ratio=0.85, T=1.536, hg=0.55419, Jw=1,
                 kf=26290, kr=25830,
                 Efront=0.0376, Erear=0, LeverArm=0.13256,
                 BFL=20.6357, CFL=1.5047, DFL=1.1233):
        self.rr = 0.329  # rolling radius
        self.mus = mus  # unsprung mass (one wheel)
        self.mf = mf  # front axle mass
        self.mr = mr  # rear axle mass
        self.m = mf + mr  # mass
        self.L = L  # length of the car
        self.ab_ratio = ab_ratio
        self.b = self.L / (1 + self.ab_ratio)
        self.a = self.L - self.b
        self.Izz = 0.5 * self.m * self.a * self.b  # moment of ineria z-axis
        self.Jw = Jw  # wheel's inertia
        self.hg = hg  # Height of mass centre above ground (m)
        self.T = T  # track width
        self.kf = kf  # front suspension stiffness
        self.kr = kr  # rear suspension stiffness
        self.rw = self.rr - (self.mf / 2 + self.mus) / self.kf

        ## Tire parameters
        self.BFL = BFL  # MagicFormula
        self.CFL = CFL
        self.DFL = DFL
        self.BFR = self.BFL
        self.CFR = self.CFL
        self.DFR = self.DFL

        self.BRL = self.BFL
        self.CRL = self.CFL
        self.DRL = self.DFL

        self.BRR = self.BFL
        self.CRR = self.CFL
        self.DRR = self.DFL

        self.Efront = Efront  # Trail/Front Wheel (m)
        self.Erear = Erear  # Trail/Rear Wheel (m)
        self.E = [self.Efront, self.Efront, self.Erear, self.Erear]
        self.LeverArm = LeverArm
        self.wL = self.T / 2
        self.wR = self.T / 2


class vehicle_models:
    def __init__(self):
        p = VehicleParameters()  # parameters

    def planar_model(self, t, state, tire_torques, mu_max, delta, p):
        # Unpacking the state-space and inputs
        U, V, wz, wFL, wFR, wRL, wRR, yaw, x, y = state
        deltaFL, deltaFR, deltaRL, deltaRR = delta
        TFL, TFR, TRL, TRR = tire_torques
        mumaxFL, mumaxFR, mumaxRL, mumaxRR = mu_max

        # Parameters
        g = 9.81
        p.BFL = mumaxFL
        p.CFL = mumaxFL
        p.DFL = mumaxFL

        p.BFR = mumaxFR
        p.CFR = mumaxFR
        p.DFR = mumaxFR

        p.BRL = mumaxRL
        p.CRL = mumaxRL
        p.DRL = mumaxRL

        p.BRR = mumaxRR
        p.CRR = mumaxRR
        p.DRR = mumaxRR

        ## Normal forces (static forces)
        fFLz0 = p.b / (p.a + p.b) * p.m * g / 2
        fFRz0 = p.b / (p.a + p.b) * p.m * g / 2
        fRLz0 = p.a / (p.a + p.b) * p.m * g / 2
        fRRz0 = p.a / (p.a + p.b) * p.m * g / 2

        DfzxL = p.m * p.hg * p.wR / ((p.a + p.b) * (p.wL + p.wR))
        DfzxR = p.m * p.hg * p.wL / ((p.a + p.b) * (p.wL + p.wR))
        DfzyF = p.m * p.hg * p.b / ((p.a + p.b) * (p.wL + p.wR))
        DfzyR = p.m * p.hg * p.a / ((p.a + p.b) * (p.wL + p.wR))

        # TODO: Algebraic loop for ax and ay needs to be fixed later
        fFLz = fFLz0 #- DfzxL * ax_prev - DfzyF * ay_prev
        fFRz = fFRz0 #- DfzxR * ax_prev + DfzyF * ay_prev
        fRLz = fRLz0 #+ DfzxL * ax_prev - DfzyR * ay_prev
        fRRz = fRRz0 #+ DfzxR * ax_prev + DfzyR * ay_prev

        ## Compute tire slip Wheel velocities
        vFLxc = U - p.T * wz / 2
        vFLyc = V + p.a * wz
        vFRxc = U + p.T * wz / 2
        vFRyc = V + p.a * wz
        vRLxc = U - p.T * wz / 2
        vRLyc = V - p.b * wz
        vRRxc = U + p.T * wz / 2
        vRRyc = V - p.b * wz

        # Rotate to obtain velocities in the tire frame
        vFLx = vFLxc * cos(deltaFL) + vFLyc * sin(deltaFL)
        vFLy = -vFLxc * sin(deltaFL) + vFLyc * cos(deltaFL)
        vFRx = vFRxc * cos(deltaFR) + vFRyc * sin(deltaFR)
        vFRy = -vFRxc * sin(deltaFR) + vFRyc * cos(deltaFR)
        vRLx = vRLxc * cos(deltaRL) + vRLyc * sin(deltaRL)
        vRLy = -vRLxc * sin(deltaRL) + vRLyc * cos(deltaRL)
        vRRx = vRRxc * cos(deltaRR) + vRRyc * sin(deltaRR)
        vRRy = -vRRxc * sin(deltaRR) + vRRyc * cos(deltaRR)

        ## Longitudinal slip
        sFLx = p.rw * wFL / vFLx - 1
        sFRx = p.rw * wFR / vFRx - 1
        sRLx = p.rw * wRL / vRLx - 1
        sRRx = p.rw * wRR / vRRx - 1

        ## Lateral slip
        sFLy = -vFLy / abs(vFLx)
        sFRy = -vFRy / abs(vFRx)
        sRLy = -vRLy / abs(vRLx)
        sRRy = -vRRy / abs(vRRx)

        ## Combined slip
        sFL = sqrt(sFLx ** 2 + sFLy ** 2)
        sFR = sqrt(sFRx ** 2 + sFRy ** 2)
        sRL = sqrt(sRLx ** 2 + sRLy ** 2)
        sRR = sqrt(sRRx ** 2 + sRRy ** 2)

        # Compute tire forces
        ## Combined friction coefficient
        muFL = p.DFL * sin(p.CFL * arctan(p.BFL * sFL))
        muFR = p.DFR * sin(p.CFR * arctan(p.BFR * sFR))
        muRL = p.DRL * sin(p.CRL * arctan(p.BRL * sRL))
        muRR = p.DRR * sin(p.CRR * arctan(p.BRR * sRR))

        ## Longitudinal friction coefficient
        if sFL != 0:
            muFLx = sFLx * muFL / sFL
        else:
            muFLx = p.DFL * sin(p.CFL * arctan(p.BFL * sFLx))

        if sFR != 0:
            muFRx = sFRx * muFR / sFR
        else:
            muFRx = p.DFR * sin(p.CFR * arctan(p.BFR * sFRx))

        if sRL != 0:
            muRLx = sRLx * muRL / sRL
        else:
            muRLx = p.DRL * sin(p.CRL * arctan(p.BRL * sRLx))

        if sRR != 0:
            muRRx = sRRx * muRR / sRR
        else:
            muRRx = p.DRR * sin(p.CRR * arctan(p.BRR * sRRx))

        ## Lateral Friction coefficient
        if sFL != 0:
            muFLy = sFLy * muFL / sFL
        else:
            muFLy = p.DFL * sin(p.CFL * arctan(p.BFL * sFLy))

        if sFR != 0:
            muFRy = sFRy * muFR / sFR
        else:
            muFRy = p.DFR * sin(p.CFR * arctan(p.BFR * sFRy))

        if sRL != 0:
            muRLy = sRLy * muRL / sRL
        else:
            muRLy = p.DRL * sin(p.CRL * arctan(p.BRL * sRLy))

        if sRR != 0:
            muRRy = sRRy * muRR / sRR
        else:
            muRRy = p.DRR * sin(p.CRR * arctan(p.BRR * sRRy))

        ## Compute longitudinal force
        fFLxt = muFLx * fFLz
        fFRxt = muFRx * fFRz
        fRLxt = muRLx * fRLz
        fRRxt = muRRx * fRRz

        ## Compute lateral forces
        fFLyt = muFLy * fFLz
        fFRyt = muFRy * fFRz
        fRLyt = muRLy * fRLz
        fRRyt = muRRy * fRRz

        ## Rotate to obtain forces in the chassis frame
        fFLx = fFLxt * cos(deltaFL) - fFLyt * sin(deltaFL)
        fFLy = fFLxt * sin(deltaFL) + fFLyt * cos(deltaFL)
        fFRx = fFRxt * cos(deltaFR) - fFRyt * sin(deltaFR)
        fFRy = fFRxt * sin(deltaFR) + fFRyt * cos(deltaFR)
        fRLx = fRLxt * cos(deltaRL) - fRLyt * sin(deltaRL)
        fRLy = fRLxt * sin(deltaRL) + fRLyt * cos(deltaRL)
        fRRx = fRRxt * cos(deltaRR) - fRRyt * sin(deltaRR)
        fRRy = fRRxt * sin(deltaRR) + fRRyt * cos(deltaRR)

        # Compute the time derivatives
        U_dot = 1 / p.m * (fFLx + fFRx + fRLx + fRRx) + V * wz
        V_dot = 1 / p.m * (fFLy + fFRy + fRLy + fRRy) - U * wz
        wz_dot = 1 / p.Izz * (p.a * (fFLy + fFRy) - p.b * (fRLy + fRRy) + p.T / 2 * (fFRx - fFLx + fRRx - fRLx))
        wFL_dot = (TFL - p.rw * fFLxt) / p.Jw
        wFR_dot = (TFR - p.rw * fFRxt) / p.Jw
        wRL_dot = (TRL - p.rw * fRLx) / p.Jw
        wRR_dot = (TRR - p.rw * fRRx) / p.Jw
        yaw_dot = wz
        x_dot = U * cos(yaw) - V * sin(yaw)
        y_dot = U * sin(yaw) + V * cos(yaw)

        state_dot = [U_dot, V_dot, wz_dot, wFL_dot, wFR_dot, wRL_dot, wRR_dot, yaw_dot, x_dot, y_dot]

        # Velocities at the wheel contact patch in the global inertial frame
        vFLxg = vFLxc * cos(yaw) - vFLyc * sin(yaw)
        vFRxg = vFRxc * cos(yaw) - vFRyc * sin(yaw)
        vRLxg = vRLxc * cos(yaw) - vRLyc * sin(yaw)
        vRRxg = vRRxc * cos(yaw) - vRRyc * sin(yaw)
        vFLyg = vFLxc * sin(yaw) + vFLyc * cos(yaw)
        vFRyg = vFRxc * sin(yaw) + vFRyc * cos(yaw)
        vRLyg = vRLxc * sin(yaw) + vRLyc * cos(yaw)
        vRRyg = vRRxc * sin(yaw) + vRRyc * cos(yaw)

        # Position of the wheel contact patch in the global inertial frame
        xFL = x + p.a * cos(yaw) - p.T / 2 * sin(yaw)
        yFL = y + p.a * sin(yaw) + p.T / 2 * cos(yaw)
        xFR = x + p.a * cos(yaw) + p.T / 2 * sin(yaw)
        yFR = y + p.a * sin(yaw) - p.T / 2 * cos(yaw)
        xRL = x - p.b * cos(yaw) - p.T / 2 * sin(yaw)
        yRL = y - p.b * sin(yaw) + p.T / 2 * cos(yaw)
        xRR = x - p.b * cos(yaw) + p.T / 2 * sin(yaw)
        yRR = y - p.b * sin(yaw) - p.T / 2 * cos(yaw)

        # Chassis velocity, and acceleration in the global inertial frame
        vx = U * cos(yaw) - V * sin(yaw)
        vy = V * sin(yaw) + U * cos(yaw)

        axc = U_dot - V * wz
        ayc = V_dot + U * wz
        ax = axc * cos(yaw) - ayc * sin(yaw)
        ay = axc * sin(yaw) + ayc * cos(yaw)

        axOut = axc
        ayOut = ayc

        return state_dot
        # return [state_dot, vx, vy, ax, ay]



def main():
    print("This script is not meant to be executable, and should be used as a library.")
    veh = vehicle_models()
            # U,V,wz,wFL,wFR,wRL,wRR,yaw,x,y
    state0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ax_prev = 1
    ay_prev = 1
    tire_torques = [100, 100, 100, 100]
    delta = [0, 0, 0, 0]
    mu_max = [1, 1, 1, 1] # road surface maximum friction
    p1 = VehicleParameters()

    sol = solve_ivp(veh.planar_model, [0, 10], state0, args=(tire_torques, mu_max, delta, p1), dense_output=True)
    t = sol.t
    U, V, wz, wFL, wFR, wRL, wRR, yaw, x, y = sol.y

    plt.figure()
    plt.title('x-y position')
    plt.plot(x, y)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    plt.figure()
    plt.title('yaw rate')
    plt.xlabel('Time (Sec)')
    plt.ylabel('Yaw rate (rad/sec)')
    plt.plot(t, yaw)
    plt.show()






if __name__ == "__main__":
    main()
