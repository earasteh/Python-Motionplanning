#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#
# author: Daniel Kloeser
import numpy as np
import casadi as ca
from libs.utils.normalise_angle import normalise_angle
import matplotlib.pyplot as plt
from libs.vehicle_model.vehicle_model import VehicleParameters
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import scipy.linalg


# params = VehicleParameters()


class StanleyController:

    def __init__(self, control_gain=2.5, softening_gain=1.0, yaw_rate_gain=0.0, steering_damp_gain=0.0,
                 max_steer=np.deg2rad(24), wheelbase=0.0, param=None,
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
        self._lookahead_distance = 5.0
        self.cross_track_deadband = 0.01

        self.px = waypoints[0][:]
        self.py = waypoints[1][:]
        self.pyaw = waypoints[2][:]
        self.params = param  # system parameters

    def update_waypoints(self):
        local_waypoints = self._waypoints

    @staticmethod
    def find_target_path_id(px, py, x, y, yaw, param):
        L = param.L

        # Calculate position of the front axle
        fx = x + L * np.cos(yaw)
        fy = y + L * np.sin(yaw)

        dx = fx - px  # Find the x-axis of the front axle relative to the path
        dy = fy - py  # Find the y-axis of the front axle relative to the path

        d = np.hypot(dx, dy)  # Find the distance from the front axle to the path
        target_index = np.argmin(d)  # Find the shortest distance in the array

        return target_index, dx[target_index], dy[target_index], d[target_index], d

    def stanley_control(self, x, y, yaw, current_velocity):
        """
        :param x:
        :param y:
        :param yaw:
        :param current_velocity:
        :return: steering output, target index, crosstrack error
        """
        target_index, dx, dy, absolute_error, _ = self.find_target_path_id(self.px, self.py, x, y, yaw, self.params)
        yaw_error = normalise_angle(self.pyaw[target_index] - yaw)
        # calculate cross-track error
        front_axle_vector = [np.sin(yaw), -np.cos(yaw)]
        nearest_path_vector = [dx, dy]
        crosstrack_error = np.sign(np.dot(nearest_path_vector, front_axle_vector)) * absolute_error
        crosstrack_steering_error = np.arctan2((self.k * crosstrack_error), (self.k_soft + current_velocity))

        desired_steering_angle = yaw_error + crosstrack_steering_error
        # Constrains steering angle to the vehicle limits
        limited_steering_angle = np.clip(desired_steering_angle, -self.max_steer, self.max_steer)

        return limited_steering_angle, target_index, crosstrack_error

    @property
    def waypoints(self):
        return self._waypoints


class MPC:
    def __init__(self, N, T, param, px, py, pyaw, veh_initial_conditions):
        self.N = N  # dt = T/N
        self.T = T  # Total time
        self.params = param  # system parameters
        self.px = px
        self.py = py
        self.pyaw = pyaw
        self.initial_conditions = veh_initial_conditions
        # acados setup
        self.constraint, self.model, self.acados_solver = self.acados_settings(self.T, self.N)
        # dimensions
        self.nx = self.model.x.size()[0]
        self.nu = self.model.u.size()[0]
        self.ny = self.nx + self.nu

    def bicycle_model(self, initial_condition, param):
        """Compute the right-hand side of the ODEs

        Args:
            states (array-like): State vector
            u (array-like): Input vector
            p (object of class parameter): Parameters
            xs (array-like, optional): steady-state
            us (array-like, optional): steady-state input

        Returns:
            array-like: dx/dt
            :param us:
            :param xs:
            :param states:
            :param param:
        """
        # define structs
        constraint = ca.types.SimpleNamespace()
        model = ca.types.SimpleNamespace()

        model_name = "bicycle_model"

        # parameters of the vehicle:
        m = param.m
        lf = param.a
        lr = param.b
        Izz = param.Izz
        rw = param.rw
        # tire parameters for the pajecka model - front
        Bf = param.BFL
        Cf = param.CFL
        Df = param.DFL
        # tire parameters for the pajecka model - rear
        Br = param.BRL
        Cr = param.CRL
        Dr = param.DRL
        # Longitudinal parameters
        # Cm1 = 0.287
        # Cm2 = 0.0545
        # Cr0 = 0.0518
        # Cr2 = 0.00035

        # states
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        yaw = ca.SX.sym('yaw')
        vx = ca.SX.sym('vx')
        vy = ca.SX.sym('vy')
        omega = ca.SX.sym('omega')
        # u
        tau = ca.SX.sym('tau')
        delta = ca.SX.sym('delta')

        state = ca.vertcat(x, y, yaw, vx, vy, omega, tau, delta)

        # system inputs (Delta U)
        derTau = ca.SX.sym('derTau')
        derDelta = ca.SX.sym('derDelta')
        sym_u = ca.vertcat(derTau, derDelta)

        # xdot
        x_dot = ca.SX.sym('x_dot')
        y_dot = ca.SX.sym('y_dot')
        yaw_dot = ca.SX.sym('yaw_dot')
        vx_dot = ca.SX.sym('vx_dot')
        vy_dot = ca.SX.sym('vy_dot')
        omega_dot = ca.SX.sym('omega_dot')
        tau_dot = ca.SX.sym('tau_dot')
        delta_dot = ca.SX.sym('delta_dot')
        state_dot = ca.vertcat(x_dot, y_dot, yaw_dot, vx_dot, vy_dot, omega_dot, tau_dot, delta_dot)

        # algebraic variables
        z = ca.vertcat([])

        # parameters
        p = ca.vertcat([])

        ## Dynamics
        # Slip angles for the front and rear
        alpha_f = delta - np.arctan((vy + lf * omega) / (vx + 0.01))
        alpha_r = np.arctan((-vy + lr * omega) / (vx + 0.01))

        Nf = lr / (lr + lf) * m * 9.81
        Nr = lf / (lr + lf) * m * 9.81
        # Lateral Forces
        Ffy = 2 * Df * np.sin(Cf * np.arctan(Bf * alpha_f)) * Nf
        Fry = 2 * Dr * np.sin(Cr * np.arctan(Br * alpha_r)) * Nr
        # Frx = (Cm1 - Cm2 * vx) * D - Cr0 - Cr2 * vx**2
        Frx = 1 * tau / rw

        dx = vx * np.cos(yaw) - vy * np.sin(yaw)
        dy = vx * np.sin(yaw) + vy * np.cos(yaw)
        dyaw = omega
        dvx = 1 / m * (Frx - Ffy * np.sin(delta) + m * vy * omega)
        dvy = 1 / m * (Fry + Ffy * np.cos(delta) - m * vx * omega)
        domega = 1 / Izz * (Ffy * lf * np.cos(delta) - Fry * lr)

        expr_f_expl = ca.vertcat(dx, dy, dyaw, dvx, dvy, domega, derTau, derDelta)

        # Constraint on acceleration
        a_long = dvx - vy * omega
        a_lat = dvy + vx * omega

        # # Model bounds
        # model.n_min = -5  # width of the track [m]
        # model.n_max = 5  # width of the track [m]

        # State bounds
        model.delta_min = -8.0 * np.pi / 180
        model.delta_max = +8.0 * np.pi / 180
        model.tau_min = -1000
        model.tau_max = +1000
        # input bounds
        model.ddelta_min = -20 * np.pi / 180 #* self.N / self.T  # minimum change rate of stering angle [rad/s]
        model.ddelta_max = +20 * np.pi / 180 #* self.N / self.T  # maximum change rate of steering angle [rad/s]
        model.dtau_min = -100  # -10.0  # minimum torque change rate
        model.dtau_max = 100  # 10.0  # maximum torque change rate
        # nonlinear constraint
        constraint.alat_min = -10  # maximum lateral force [m/s^2]
        constraint.alat_max = 10  # maximum lateral force [m/s^1]
        constraint.along_min = -40  # maximum lateral force [m/s^2]
        constraint.along_max = 40  # maximum lateral force [m/s^2]

        # Define initial conditions
        # model.x0 = np.array([0, 10, 0, 10, 0, 0, 0, 0])
        model.x0 = initial_condition
        # define constraints struct
        # constraint.alat = ca.Function("a_lat", [state, sym_u], [a_lat])
        constraint.expr = ca.vertcat(a_long, a_lat, tau, delta)

        # Define model struct
        # params = ca.types.SimpleNamespace()

        model.f_impl_expr = state_dot - expr_f_expl
        model.f_expl_expr = expr_f_expl
        model.x = state
        model.xdot = state_dot
        model.u = sym_u
        model.z = z
        model.p = p
        model.name = model_name
        # model.params = params
        return model, constraint

    def acados_settings(self, Tf, N):
        # create render arguments
        ocp = AcadosOcp()

        # export model
        model, constraint = self.bicycle_model(self.initial_conditions, self.params)

        # define acados ODE
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.f_impl_expr
        model_ac.f_expl_expr = model.f_expl_expr
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.z = model.z
        model_ac.p = model.p
        model_ac.name = model.name
        ocp.model = model_ac

        # define constraint
        model_ac.con_h_expr = constraint.expr

        # dimensions
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
        ny_e = nx

        nsbx = 1
        nh = constraint.expr.shape[0]
        # nh = 0
        nsh = nh
        ns = nsh + nsbx

        # discretization
        ocp.dims.N = N

        # set cost
        # Q = np.diag([1e-1, 1e-8, 1e-8, 1e-8, 1e-3, 5e-3])
        Q = np.diag([0, 2000.0, 6000.0, 100.0, 0, 0, 0, 0])

        R = np.eye(nu)
        R[0, 0] = 1e-3
        R[1, 1] = 5e-3
        # R[0, 0] = 0
        # R[1, 1] = 0

        # Qe = np.diag([5e0, 1e1, 1e-8, 1e-8, 5e-3, 2e-3])
        Qe = np.diag([0, 20000.0, 6000.0, 100.0, 0, 0, 0, 0])

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        unscale = N / Tf

        ocp.cost.W = unscale * scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Qe / unscale

        Vx = np.zeros((ny, nx))
        # Vx[:nx, :nx] = np.eye(nx)
        Vx[1, 1] = 1  # y
        Vx[2, 2] = 1  # yaw
        Vx[3, 3] = 0  # forward velocity
        ocp.cost.Vx = Vx

        Vu = np.zeros((ny, nu))
        Vu[6, 0] = 1.0
        Vu[7, 1] = 1.0
        ocp.cost.Vu = Vu

        Vx_e = np.zeros((ny_e, nx))
        Vx_e[:nx, :nx] = np.eye(nx)
        ocp.cost.Vx_e = Vx_e

        # ocp.cost.zl = 100 * np.ones((ns,))
        # ocp.cost.zu = 100 * np.ones((ns,))
        # ocp.cost.Zl = 1 * np.ones((ns,))
        # ocp.cost.Zu = 1 * np.ones((ns,))

        # set intial references
        ocp.cost.yref = np.array([0, 10, 0, 10, 0, 0, 0, 0, 0, -1*np.pi/180])
        ocp.cost.yref_e = np.array([0, 10, 0, 10, 0, 0, 0, 0])

        # setting constraints
        # ocp.constraints.lbx = np.array([model.tau_min, model.delta_min])
        # ocp.constraints.ubx = np.array([model.tau_max, model.delta_max])
        # ocp.constraints.idxbx = np.array([1, 1]) #np.array([[0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])

        ocp.constraints.lbu = np.array([model.dtau_min, model.ddelta_min])
        ocp.constraints.ubu = np.array([model.dtau_max, model.ddelta_max])
        ocp.constraints.Jbu = np.array([[1, 0],
                                        [0, 1]])

        ocp.constraints.Jbx_e = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0, 0, 0]])
        ocp.constraints.lbx_e = np.array([0, -60 * np.pi/180])
        ocp.constraints.ubx_e = np.array([20, +60 * np.pi / 180])

        # Slack variables for state constraints
        # ocp.constraints.lsbx = np.zeros([nsbx])
        # ocp.constraints.usbx = np.zeros([nsbx])
        # ocp.constraints.idxsbx = np.array(range(nsbx))

        ocp.constraints.lh = np.array(
            [
                constraint.along_min,
                constraint.alat_min,
                model.tau_min,
                model.delta_min,
            ]
        )
        ocp.constraints.uh = np.array(
            [
                constraint.along_max,
                constraint.alat_max,
                model.tau_max,
                model.delta_max,
            ]
        )

        # Slack variables for along, alat, delta_max
        # ocp.constraints.lsh = np.zeros(nsh)
        # ocp.constraints.ush = np.zeros(nsh)
        # ocp.constraints.idxsh = np.array(range(nsh))

        # set intial condition
        ocp.constraints.x0 = model.x0

        # set QP solver and integration
        ocp.solver_options.tf = Tf
        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP"
        # ocp.solver_options.levenberg_marquardt = 0.00001
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
        ocp.solver_options.nlp_solver_max_iter = 1000
        ocp.solver_options.tol = 1e-4
        # ocp.solver_options.nlp_solver_tol_comp = 1e-1

        # create solver
        acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
        acados_solver.options_set('qp_warm_start', 2)

        return constraint, model, acados_solver

    def solve_mpc(self, current_state, uk_prev_step, px, py, pyaw):
        """Solve MPC provided the current state, i.e., this
        function is u = h(x), which is the implicit control law of MPC.

        Args:
            current_state (array-like): current state

        Returns:
            tuple: current input and return status pair
            :param uk_prev_step: previous input
            :param current_state:
        """
        #     # unpacking the real signals that come from the system
        x_r, y_r, yaw_r, vx_r, vy_r, omega_r = current_state

        self.acados_solver.set(0, "lbx", np.array([x_r, y_r, yaw_r, vx_r, vy_r, omega_r, uk_prev_step[0], uk_prev_step[1]]))
        self.acados_solver.set(0, "ubx", np.array([x_r, y_r, yaw_r, vx_r, vy_r, omega_r, uk_prev_step[0], uk_prev_step[1]]))

        target_index, _, _, _, _ = StanleyController.find_target_path_id(px, py, x_r, y_r, yaw_r, self.params)
        local_px = px[target_index]
        local_py = py[target_index]
        local_yaw = pyaw[target_index]

        qy = 150 * 1 / 0.01 ** 2
        qyaw = 100 * 1 / (0.1 * np.pi / 180) ** 2
        qDelta = 100 * 1/(0.1 * np.pi/180) ** 2

        Q = np.diag([0, qy, qyaw, 100, 0, 0, 0, qDelta])
        R = np.eye(2)
        R[0, 0] = 0 * 1e-3
        R[1, 1] = 6000 * 1/(0.01 * np.pi/180) ** 2
        W = scipy.linalg.block_diag(Q, R) # self.N / self.T *
        Qe = np.diag([0, 40 * qy, 100 * qyaw, 10, 0, 0, 0, 0]) #/ (self.N / self.T)
        W_e = Qe

        for i in range(self.N):
            self.acados_solver.cost_set(i, 'W', W)
            yref = np.array([0, local_py, local_yaw, 10.0, 0, 0, 0, 0, 0, 0])
            self.acados_solver.set(i, "yref", yref)
        self.acados_solver.cost_set(self.N, 'W', W_e)
        yref_N = np.array([0, local_py, local_yaw, 10.0, 0, 0, 0, 0])

        self.acados_solver.constraints_set(self.N, 'lbx', np.array([local_py - 10 * np.cos(local_yaw), local_yaw - 60 * np.pi/180]))
        self.acados_solver.constraints_set(self.N, 'ubx', np.array([local_py + 10 * np.cos(local_yaw), local_yaw + 60 * np.pi / 180]))
        self.acados_solver.set(self.N, "yref", yref_N)

        # print(f'yref = {yref}')

        # set options
        self.acados_solver.options_set('print_level', 0)
        status = self.acados_solver.solve()

        # get solution
        x0 = self.acados_solver.get(1, "x")  # x1 used for solution
        u0 = self.acados_solver.get(1, "u")
        xN = self.acados_solver.get(self.N, "x")

        # print(f'yr: {y_r}')
        # print(f'local_py:{local_py}')
        # print(f'local_yaw:{local_yaw}')
        # print(f'x = {x0}')

        # print(f'MPC {local_py - xN[1]} vs. real {local_py - y_r}')
        u = x0[6:]

        crosstrack = y_r - local_py

        if status != 0:
            print("acados returned status {}.".format(status))

        # print("cost is {}".format(self.acados_solver.get_cost()))
        # print("u0 is {}".format(u0))
        # print("X0 is {}".format(x0))
        # print("xN is {}".format(self.acados_solver.get(self.N, "x")))
        # print(f'x1 = {x1}')
        # update initial condition

        # self.initial_conditions = x0

        return [u, crosstrack, x0, xN, status]


def main():
    print("This script is not meant to be executable, and should be used as a library.")


if __name__ == "__main__":
    main()
