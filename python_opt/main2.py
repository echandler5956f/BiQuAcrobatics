from casadi import *
import numpy as np
from scipy.spatial.transform import Rotation as rp
from scipy.spatial.transform import Slerp


class Contacts:
    def __init__(self):
        self.num_cons = None
        self.step_list = None
        self.num_steps = None
        self.cum_steps = None
        self.contact_list = None
        self.num_legs = None

    def initialize_contacts(self, step_list, contact_list):
        self.num_cons = len(step_list)
        self.step_list = step_list
        self.num_steps = int(np.sum(step_list))
        self.cum_steps = np.cumsum(step_list)
        self.contact_list = contact_list
        self.num_legs = len(self.contact_list[0])

    def get_current_phase(self, k):
        i = 0
        for j in range(self.num_cons):
            i = i + (k >= self.cum_steps[j])
        return i

    def get_offset(self, leg):
        return leg * 3 * self.num_steps

    def is_new_contact(self, k, leg):
        i = self.get_current_phase(k)
        if self.contact_list[i][leg] and i > 0:
            if not self.contact_list[i - 1][leg] and k - self.cum_steps[i - 1] == 0:
                return True
        return False


class Collocation:
    def __init__(self, degree):
        # Degree of interpolating polynomial
        self.degree = degree

        # Get collocation points
        tau_root = np.append(0, collocation_points(self.degree, 'legendre'))

        # Coefficients of the collocation equation
        self.C = np.zeros((self.degree + 1, self.degree + 1))

        # Coefficients of the continuity equation
        self.D = np.zeros(self.degree + 1)

        # Coefficients of the quadrature function
        self.B = np.zeros(self.degree + 1)

        # Construct polynomial basis
        for j in range(self.degree + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(self.degree + 1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            self.D[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the
            # continuity equation
            p_der = np.polyder(p)
            for r in range(self.degree + 1):
                self.C[j, r] = p_der(tau_root[r])

            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            self.B[j] = pint(1.0)


class Parameters:
    def __init__(self, mass, inertia, r, p_feet_bar, f_max):
        # Contact pattern (Note: This is mutated in MotionProfile to get the final contacts object)
        self.contacts = Contacts()

        # Mass of the SRB
        self.mass = mass

        # Inertia of SRB
        self.inertia = inertia

        # Sphere of radius r that the foot position is constrained to
        self.r = r

        # The sth foot position is constrained in a sphere of radius r to satisfy
        # joint limits. This parameter is the center of the sphere w.r.t the COM
        self.p_feet_bar = p_feet_bar

        # Maximum ground reaction force in the normal direction
        self.f_max = f_max

        # Acceleration due to gravity
        self.g_accel = np.array([0, 0, 9.81])

        # Friction coefficient
        self.mu = 0.7

        # Omega cost weight
        self.eOmega = 5e-5

        # Force cost weight
        self.eF = 1e-6

        # Rotation error cost weight
        self.eR = 1e-3

        # Final position error cost weight
        self.eP = 1e-20

        # Number of steps per phase
        self.step_count = 15

        # Degree of the interpolating polynomial used in the gaussian quadrature function approximation
        self.g_degree = 2

        # Number of cubic Hermite curve segments used to parameterize the contact forces
        self.F_segments = 3

        # Number of cubic Hermite curve segments used to parameterize the foot positions
        self.p_segments = 3

        # Order of Taylor series approximation for matrix exponential
        self.e_terms = 8

        # Order of Taylor series approximation for matrix logarithm
        self.l_terms = 8

        # Just-In-Time compilation options
        jit_options = {"flags": ["-O3"], "verbose": True, "compiler": "gcc"}

        # Solver options
        self.opts = {"expand": True, "detect_simple_bounds": True, "warn_initial_bounds": True,
                     # "jit": True,
                     # "compiler": "shell",
                     # "jit_options": jit_options,
                     "ipopt": {"max_iter": 100,
                               "fixed_variable_treatment": "make_constraint",
                               "hessian_approximation": "limited-memory",
                               "mumps_mem_percent": 10000,
                               "print_level": 5}}


class Constraints:
    def __init__(self, p):
        grf_sym = SX.sym('grf_sym', 3)
        tau_sym = SX.sym('tau_sym', 3)

        p_body_sym = SX.sym('p_body_sym', 3)
        dp_body_sym = SX.sym('dp_body_sym', 3)
        Omega_sym = SX.sym('Omega_sym', 3)
        DOmega_sym = SX.sym('DOmega_sym', 3)

        R_sym = SX.sym('R_sym', 3, 3)
        R_ref_sym = SX.sym('R_ref_sym', 3, 3)

        x = vertcat(p_body_sym, dp_body_sym, Omega_sym, DOmega_sym)
        u = vertcat(grf_sym, tau_sym)

        f_sym_x = SX.sym('f_sym_x')
        f_sym_y = SX.sym('f_sym_y')
        f_sym_z = SX.sym('f_sym_z')
        f_sym = vertcat(f_sym_x, f_sym_y, f_sym_z)

        p_feet_sym = SX.sym('p_feet_sym', 3)
        p_feet_bar_sym = SX.sym('p_feet_bar_sym', 3)

        xdot = vertcat(dp_body_sym, grf_sym / p.mass - p.g_accel, DOmega_sym, mtimes(np.linalg.inv(p.inertia), (
                mtimes(transpose(R_sym), tau_sym) - cross(Omega_sym, mtimes(p.inertia, Omega_sym)))))
        e_R_sym = inv_skew(self.approximate_log_a(mtimes(transpose(R_ref_sym), R_sym), p.l_terms))
        L = 0
        # L = L + p.eOmega * mtimes(transpose(Omega_sym), Omega_sym)
        # L = L + p.eF * mtimes(transpose(grf_sym), grf_sym)
        # L = L + p.eR * mtimes(transpose(e_R_sym), e_R_sym)

        self.fx = Function('fx', [x, u, R_sym], [xdot], ['x', 'u', 'R'], ['xdot'])
        self.fl = Function('fl', [x, u, R_sym, R_ref_sym], [L], ['x', 'u', 'R', 'R_ref'], ['L'])

        self.friction_cone_x = Function('friction_cone_x', [f_sym], [f_sym_x / f_sym_z], ['f'], ['fcx'])
        self.friction_cone_y = Function('friction_cone_y', [f_sym], [f_sym_y / f_sym_z], ['f'], ['fcy'])

        self.kinematic_constraint = Function('kinematic_constraint', [p_body_sym, R_sym, p_feet_sym, p_feet_bar_sym],
                                             [norm_2(mtimes(R_sym, p_feet_sym - p_body_sym) - p_feet_bar_sym)],
                                             ['p_body', 'R', 'p_feet', 'p_feet_bar'], ['kc'])

    @staticmethod
    def approximate_log_a(a, deg):
        log_a = DM(np.zeros((3, 3)))
        for i in range(1, deg):
            log_a = log_a + power(-1, i + 1) * mpower(a - DM(np.eye(3)), i) / i
        return log_a


class LegStorage:
    def __init__(self, num_legs):
        self.num_legs = num_legs
        self.F_syms = {}
        self.F = {}
        self.p_feet_syms = {}
        self.p_feet = {}
        for leg in range(self.num_legs):
            self.F[leg] = []
            self.p_feet[leg] = []


class CubicHermiteSpline:
    def __init__(self, p):
        x0 = SX.sym('chs_x0')
        x1 = SX.sym('chs_x1')
        dx0 = SX.sym('chs_dx0')
        dx1 = SX.sym('chs_dx1')
        t = SX.sym('chs_t')
        DT = SX.sym('chs_DT')
        a0 = x0
        a1 = dx0
        a2 = - (3.0 * (x0 - x1) + DT * (2.0 * dx0 + dx1)) / power(DT, 2)
        a3 = (2.0 * (x0 - x1) + DT * (dx0 + dx1)) / power(DT, 3)
        x = a0 + (a1 * t) + (a2 * power(t, 2)) + (a3 * power(t, 3))
        self.evaluate_at_t = Function('evaluate_at_t', [x0, x1, dx0, dx1, t, DT], [x],
                                      ['x0', 'x1', 'dx0', 'dx1', 't', 'DT'], ['x'])


class BiQuAcrobatics:
    # def __init__(self, p, mp, constraints):
    #     self.collocation = Collocation(p.g_degree)
    #     self.contacts = p.contacts
    #     self.opts = p.opts
    #     self.p = p
    #     self.R0 = mp.R_ref[0]
    #
    #     self.p_body = []
    #     self.dp_body = []
    #     self.Omega = []
    #     self.DOmega = []
    #
    #     self.R = []
    #     self.T = MX.sym('T', self.contacts.num_cons)
    #
    #     self.w = []
    #     self.lbw = []
    #     self.ubw = []
    #     self.w0 = []
    #     self.J = 0
    #     self.g = []
    #     self.lbg = []
    #     self.ubg = []
    #     self.add_g_cons(sum1(self.T), [mp.tMin], [mp.tMax])
    #     self.add_w_cons(self.T, mp.t_b[:, 0], mp.t_b[:, 1], mp.t_g)
    #
    #     n_steps = self.contacts.num_steps
    #     n_legs = self.contacts.num_legs
    #
    #     self.LS = LegStorage(n_legs)
    #
    #     R_k = self.R0
    #     for k in range(n_steps):
    #         i, dt = self.get_dt_k(k)
    #         self.p_body = vertcat(self.p_body, MX.sym('p_body_k{}'.format(k), 3, 1))
    #         self.dp_body = vertcat(self.dp_body, MX.sym('dp_body_k{}'.format(k), 3, 1))
    #         Omega_k = MX.sym('Omega_k{}'.format(k), 3, 1)
    #         self.Omega = vertcat(self.Omega, Omega_k)
    #         self.DOmega = vertcat(self.DOmega, MX.sym('DOmega_k{}'.format(k), 3, 1))
    #         if k != 0:
    #             R_k = mtimes(R_k, self.approximate_exp_a(skew(Omega_k * dt), p.e_terms))
    #         self.R = horzcat(self.R, R_k)
    #
    #     # Lift initial conditions
    #     x_k = self.get_state_k(0)
    #     p_body_k, dp_body_k, Omega_k, DOmega_k = vertsplit(x_k, [0, 3, 6, 9, 12])
    #     R_k = self.get_R_k(0)
    #     R_ref_k = mp.R_ref[0]
    #     self.add_w_cons(x_k, mp.x_g[:, 0], mp.x_g[:, 0], mp.x_g[:, 0])
    #     p_feet_k = []
    #     for leg in range(self.contacts.num_legs):
    #         p_feet_k = horzcat(p_feet_k, self.get_p_feet_k(0, leg))
    #
    #     for k in range(n_steps):
    #         i, dt = self.get_dt_k(k)
    #
    #         grf = np.zeros((3, 1))
    #         tau = np.zeros((3, 1))
    #         for leg in range(self.contacts.num_legs):
    #             offset = self.contacts.get_offset(leg)
    #             F_k = self.get_F_k(k, leg)
    #             self.add_g_cons(constraints.kinematic_constraint(p_body_k, R_k, p_feet_k[:, leg], p.p_feet_bar[leg, :]),
    #                             [0], [p.r])
    #             if self.contacts.contact_list[i][leg]:
    #                 grf = grf + F_k
    #                 tau = tau + cross(p_feet_k[:, leg] - p_body_k, F_k)
    #                 self.add_g_cons(constraints.friction_cone_x(F_k), [-p.mu], [p.mu])
    #                 self.add_g_cons(constraints.friction_cone_y(F_k), [-p.mu], [p.mu])
    #                 self.add_w_cons(F_k, mp.f_b[:, 0], mp.f_b[:, 1], mp.f_g[offset + 3 * k: offset + 3 * (k + 1)])
    #                 if self.contacts.is_new_contact(k, leg) and k != n_steps - 1:
    #                     p_feet_k_leg = self.get_p_feet_k(k, leg)
    #                     p_feet_k[:, leg] = p_feet_k_leg
    #                     self.add_w_cons(p_feet_k_leg, mp.p_feet_b_ground[:, 0], mp.p_feet_b_ground[:, 1],
    #                                     np.vstack((mp.p_feet_g[offset + 3 * k: offset + (3 * k) + 2], 0.0)))
    #             else:
    #                 p_feet_k_leg = self.get_p_feet_k(k, leg)
    #                 p_feet_k[:, leg] = p_feet_k_leg
    #                 self.add_w_cons(p_feet_k_leg, mp.p_feet_b[:, 0], mp.p_feet_b[:, 1],
    #                                 mp.p_feet_g[offset + 3 * k: offset + 3 * (k + 1)])
    #             if k == n_steps - 1:
    #                 p_feet_k_leg = self.get_p_feet_k(k, leg)
    #                 p_feet_k[:, leg] = p_feet_k_leg
    #                 self.add_w_cons(p_feet_k_leg, mp.p_feet_g[offset + 3 * k: offset + 3 * (k + 1)],
    #                                 mp.p_feet_g[offset + 3 * k: offset + 3 * (k + 1)],
    #                                 mp.p_feet_g[offset + 3 * k: offset + 3 * (k + 1)])
    #         u_k = vertcat(grf, tau)
    #
    #         # State at collocation points
    #         x_c = []
    #         for j in range(self.collocation.degree):
    #             x_kj = MX.sym('X_' + str(k) + '_' + str(j), 12)
    #             x_c.append(x_kj)
    #             self.add_w_cons(x_kj, mp.x_b[:, 0], mp.x_b[:, 1], mp.x_g[:, k])
    #
    #         # Loop over collocation points
    #         x_k_end = self.collocation.D[0] * x_k
    #         for j in range(1, self.collocation.degree + 1):
    #             # Expression for the state derivative at the collocation point
    #             x_p = self.collocation.C[0, j] * x_k
    #             for r in range(self.collocation.degree):
    #                 x_p = x_p + self.collocation.C[r + 1, j] * x_c[r]
    #
    #             # Append collocation equations
    #             fj = constraints.fx(x_c[j - 1], u_k, R_k)
    #             qj = constraints.fl(x_c[j - 1], u_k, R_k, R_ref_k)
    #             self.add_g_cons(dt * fj - x_p, np.zeros((12, 1)), np.zeros((12, 1)))
    #
    #             # Add contribution to the end state
    #             x_k_end = x_k_end + self.collocation.D[j] * x_c[j - 1]
    #
    #             # Add contribution to quadrature function
    #             self.J = self.J + self.collocation.B[j] * qj * dt
    #
    #         # New NLP variable for state at end of interval
    #         x_k = self.get_state_k(k)
    #
    #         # Add equality constraint
    #         self.add_g_cons(x_k_end - x_k, np.zeros((12, 1)), np.zeros((12, 1)))
    #
    #         p_body_k, dp_body_k, Omega_k, DOmega_k = vertsplit(x_k, [0, 3, 6, 9, 12])
    #         R_k = self.get_R_k(k)
    #         R_ref_k = mp.R_ref[k]
    #
    #         if k == n_steps - 1:
    #             # Final position constraint
    #             self.add_w_cons(p_body_k, mp.p_body_g[:, k], mp.p_body_g[:, k], mp.p_body_g[:, k])
    #
    #             # Final orientation constraint
    #             self.add_g_cons(reshape(transpose(R_k), (9, 1)), np.reshape(R_ref_k, (9, 1)),
    #                             np.reshape(R_ref_k, (9, 1)))
    #
    #             # The rest of the states only need the typical path constraints
    #             self.add_w_cons(dp_body_k, mp.dp_body_b[:, 0], mp.dp_body_b[:, 1], mp.dp_body_g[:, k])
    #             self.add_w_cons(Omega_k, mp.Omega_b[:, 0], mp.Omega_b[:, 1], mp.Omega_g[:, k])
    #             self.add_w_cons(DOmega_k, mp.DOmega_b[:, 0], mp.DOmega_b[:, 1], mp.DOmega_g[:, k])
    #         elif k != 0:
    #             self.add_w_cons(x_k, mp.x_b[:, 0], mp.x_b[:, 1], mp.x_g[:, k])
    #
    #     # Function to get the optimized x and u at the knot points from w
    #     self.get_opt = Function('get_opt', [self.w],
    #                             [self.T, self.p_body.reshape((3, n_steps)), self.dp_body.reshape((3, n_steps)),
    #                              self.Omega.reshape((3, n_steps)), self.DOmega.reshape((3, n_steps)), self.F,
    #                              self.p_feet], ['w'], ['T', 'p_body', 'dp_body', 'Omega', 'DOmega', 'F', 'p_feet'])

    def __init__(self, p, mp, constraints):
        self.contacts = p.contacts
        self.opts = p.opts
        self.p = p
        self.R0 = mp.R_ref[0]

        self.p_body = []
        self.dp_body = []
        self.Omega = []
        self.DOmega = []

        self.R = []
        self.T = MX.sym('T', self.contacts.num_cons)
        self.F = []
        self.p_feet = []
        self.F_indices = []
        self.p_feet_indices = []
        self.F_length = np.zeros((self.contacts.num_legs, 1), dtype=int)
        self.p_feet_length = np.zeros((self.contacts.num_legs, 1), dtype=int)

        self.w = []
        self.lbw = []
        self.ubw = []
        self.w0 = []
        self.J = 0
        self.g = []
        self.lbg = []
        self.ubg = []
        self.add_g_cons(sum1(self.T), [mp.tMin], [mp.tMax])
        self.add_w_cons(self.T, mp.t_b[:, 0], mp.t_b[:, 1], mp.t_g)

        collocation = Collocation(p.g_degree)

        n_steps = self.contacts.num_steps

        R_k = self.R0
        for k in range(n_steps):
            i, dt = self.get_dt_k(k)
            self.p_body = vertcat(self.p_body, MX.sym('p_body_k{}'.format(k), 3, 1))
            self.dp_body = vertcat(self.dp_body, MX.sym('dp_body_k{}'.format(k), 3, 1))
            Omega_k = MX.sym('Omega_k{}'.format(k), 3, 1)
            self.Omega = vertcat(self.Omega, Omega_k)
            self.DOmega = vertcat(self.DOmega, MX.sym('DOmega_k{}'.format(k), 3, 1))
            if k != 0:
                R_k = mtimes(R_k, self.approximate_exp_a(skew(Omega_k * dt), p.e_terms))
            self.R = horzcat(self.R, R_k)

        for leg in range(self.contacts.num_legs):
            for k in range(n_steps):
                offset = self.contacts.get_offset(leg)
                if self.contacts.contact_list[self.contacts.get_current_phase(k)][leg]:
                    if self.contacts.is_new_contact(k, leg) or k == n_steps - 1:
                        self.p_feet = vertcat(self.p_feet, MX.sym('p_feet_' + str(leg) + '_' + str(k), 3))
                        self.p_feet_indices.extend(np.arange(offset + 3 * k, offset + 3 * (k + 1)))
                        self.p_feet_length[leg] = self.p_feet_length[leg] + 1
                    else:
                        self.p_feet = vertcat(self.p_feet, MX.zeros(3))
                    self.F = vertcat(self.F, MX.sym('F_' + str(leg) + '_' + str(k), 3))
                    self.F_indices.extend(np.arange(offset + 3 * k, offset + 3 * (k + 1)))
                    self.F_length[leg] = self.F_length[leg] + 1
                else:
                    self.p_feet = vertcat(self.p_feet, MX.sym('p_feet_' + str(leg) + '_' + str(k), 3))
                    self.p_feet_indices.extend(np.arange(offset + 3 * k, offset + 3 * (k + 1)))
                    self.p_feet_length[leg] = self.p_feet_length[leg] + 1
                    self.F = vertcat(self.F, MX.zeros(3))

        # Lift initial conditions
        x_k = self.get_state_k(0)
        p_body_k, dp_body_k, Omega_k, DOmega_k = vertsplit(x_k, [0, 3, 6, 9, 12])
        R_k = self.get_R_k(0)
        R_ref_k = mp.R_ref[0]
        self.add_w_cons(x_k, mp.x_g[:, 0], mp.x_g[:, 0], mp.x_g[:, 0])
        p_feet_k = []
        for leg in range(self.contacts.num_legs):
            p_feet_k = horzcat(p_feet_k, self.get_p_feet_k(0, leg))

        for k in range(n_steps):
            i, dt = self.get_dt_k(k)

            grf = np.zeros((3, 1))
            tau = np.zeros((3, 1))
            for leg in range(self.contacts.num_legs):
                offset = self.contacts.get_offset(leg)
                F_k = self.get_F_k(k, leg)
                self.add_g_cons(constraints.kinematic_constraint(p_body_k, R_k, p_feet_k[:, leg], p.p_feet_bar[leg, :]),
                                [0], [p.r])
                if self.contacts.contact_list[i][leg]:
                    grf = grf + F_k
                    tau = tau + cross(p_feet_k[:, leg] - p_body_k, F_k)
                    # self.add_g_cons(constraints.friction_cone_x(F_k), [-p.mu], [p.mu])
                    # self.add_g_cons(constraints.friction_cone_y(F_k), [-p.mu], [p.mu])
                    self.add_w_cons(F_k, mp.f_b[:, 0], mp.f_b[:, 1], mp.f_g[offset + 3 * k: offset + 3 * (k + 1)])
                    if self.contacts.is_new_contact(k, leg) and k != n_steps - 1:
                        p_feet_k_leg = self.get_p_feet_k(k, leg)
                        p_feet_k[:, leg] = p_feet_k_leg
                        self.add_w_cons(p_feet_k_leg, mp.p_feet_b_ground[:, 0], mp.p_feet_b_ground[:, 1],
                                        np.vstack((mp.p_feet_g[offset + 3 * k: offset + (3 * k) + 2], 0.0)))
                else:
                    p_feet_k_leg = self.get_p_feet_k(k, leg)
                    p_feet_k[:, leg] = p_feet_k_leg
                    self.add_w_cons(p_feet_k_leg, mp.p_feet_b[:, 0], mp.p_feet_b[:, 1],
                                    mp.p_feet_g[offset + 3 * k: offset + 3 * (k + 1)])
                if k == n_steps - 1:
                    p_feet_k_leg = self.get_p_feet_k(k, leg)
                    p_feet_k[:, leg] = p_feet_k_leg
                    self.add_w_cons(p_feet_k_leg, mp.p_feet_g[offset + 3 * k: offset + 3 * (k + 1)],
                                    mp.p_feet_g[offset + 3 * k: offset + 3 * (k + 1)],
                                    mp.p_feet_g[offset + 3 * k: offset + 3 * (k + 1)])
            u_k = vertcat(grf, tau)

            # State at collocation points
            x_c = []
            for j in range(collocation.degree):
                x_kj = MX.sym('X_' + str(k) + '_' + str(j), 12)
                x_c.append(x_kj)
                self.add_w_cons(x_kj, mp.x_b[:, 0], mp.x_b[:, 1], mp.x_g[:, k])

            # Loop over collocation points
            x_k_end = collocation.D[0] * x_k
            for j in range(1, collocation.degree + 1):
                # Expression for the state derivative at the collocation point
                x_p = collocation.C[0, j] * x_k
                for r in range(collocation.degree):
                    x_p = x_p + collocation.C[r + 1, j] * x_c[r]

                # Append collocation equations
                fj = constraints.fx(x_c[j - 1], u_k, R_k)
                qj = constraints.fl(x_c[j - 1], u_k, R_k, R_ref_k)
                self.add_g_cons(dt * fj - x_p, np.zeros((12, 1)), np.zeros((12, 1)))

                # Add contribution to the end state
                x_k_end = x_k_end + collocation.D[j] * x_c[j - 1]

                # Add contribution to quadrature function
                self.J = self.J + collocation.B[j] * qj * dt

            # New NLP variable for state at end of interval
            x_k = self.get_state_k(k)

            # Add equality constraint
            self.add_g_cons(x_k_end - x_k, np.zeros((12, 1)), np.zeros((12, 1)))

            p_body_k, dp_body_k, Omega_k, DOmega_k = vertsplit(x_k, [0, 3, 6, 9, 12])
            R_k = self.get_R_k(k)
            R_ref_k = mp.R_ref[k]

            if k == n_steps - 1:
                # Final position constraint
                self.add_w_cons(p_body_k, mp.p_body_g[:, k], mp.p_body_g[:, k], mp.p_body_g[:, k])

                # Final orientation constraint
                self.add_g_cons(reshape(transpose(R_k), (9, 1)), np.reshape(R_ref_k, (9, 1)),
                                np.reshape(R_ref_k, (9, 1)))

                # The rest of the states only need the typical path constraints
                self.add_w_cons(dp_body_k, mp.dp_body_b[:, 0], mp.dp_body_b[:, 1], mp.dp_body_g[:, k])
                self.add_w_cons(Omega_k, mp.Omega_b[:, 0], mp.Omega_b[:, 1], mp.Omega_g[:, k])
                self.add_w_cons(DOmega_k, mp.DOmega_b[:, 0], mp.DOmega_b[:, 1], mp.DOmega_g[:, k])
            elif k != 0:
                self.add_w_cons(x_k, mp.x_b[:, 0], mp.x_b[:, 1], mp.x_g[:, k])

        # self.F_sym = vertcat(*[self.F[i] for i in self.F_indices])
        # self.p_feet_sym = vertcat(*[self.p_feet[i] for i in self.p_feet_indices])

        # Function to get the optimized x and u at the knot points from w
        self.get_opt = Function('get_opt', [self.w],
                                [self.T, self.p_body.reshape((3, n_steps)), self.dp_body.reshape((3, n_steps)),
                                 self.Omega.reshape((3, n_steps)), self.DOmega.reshape((3, n_steps)), self.F,
                                 self.p_feet], ['w'], ['T', 'p_body', 'dp_body', 'Omega', 'DOmega', 'F', 'p_feet'])

    def optimize(self):
        # Initialize an NLP solver
        nlp = {'x': self.w, 'f': self.J, 'g': self.g}

        # Allocate a solver
        solver = nlpsol("solver", "ipopt", nlp, self.opts)

        # Solve the NLP
        sol = solver(x0=self.w0, lbg=self.lbg, ubg=self.ubg, lbx=self.lbw, ubx=self.ubw)
        T, p_body, dp_body, Omega, DOmega, F, p_feet = self.get_opt(sol['x'])
        R = self.integrate_omega_history(self.R0, Omega, T).reshape((self.contacts.num_steps, 9))

        F = np.transpose(np.reshape(F, (-1, 3)))
        p_feet = np.transpose(np.reshape(p_feet, (-1, 3)))
        if self.contacts.num_legs == 2:
            F_fR, F_fL = np.hsplit(F, 2)
            p_fR, p_fL = np.hsplit(p_feet, 2)
            np.savetxt('huron/opt/T_opt.csv', np.transpose(np.array(T)), delimiter=',')
            np.savetxt('huron/opt/p_body_opt.csv', transpose(p_body), delimiter=',')
            np.savetxt('huron/opt/dp_body_opt.csv', transpose(dp_body), delimiter=',')
            np.savetxt('huron/opt/Omega_opt.csv', transpose(Omega), delimiter=',')
            np.savetxt('huron/opt/DOmega_opt.csv', transpose(DOmega), delimiter=',')
            np.savetxt('huron/opt/R_opt.csv', R, delimiter=',')

            np.savetxt('huron/opt/F_fL.csv', F_fL, delimiter=',')
            np.savetxt('huron/opt/F_fR.csv', F_fR, delimiter=',')

            np.savetxt('huron/opt/p_fL.csv', transpose(p_fL), delimiter=',')
            np.savetxt('huron/opt/p_fR.csv', transpose(p_fR), delimiter=',')

            np.savetxt('huron/metadata/step_list.csv', np.array(self.contacts.step_list), delimiter=',')
            np.savetxt('huron/metadata/contact_list.csv', np.array(self.contacts.contact_list), delimiter=',')
            np.savetxt('huron/metadata/p_feet_bar.csv', np.array(transpose(self.p.p_feet_bar)), delimiter=',')
            np.savetxt('huron/metadata/r.csv', [self.p.r], delimiter=',')
        print(self.contacts.contact_list)

    def add_w_cons(self, w_k, lbw_k, ubw_k, w0_k):
        if w_k.size1() == len(lbw_k) and w_k.size1() == len(ubw_k) and len(lbw_k) == len(ubw_k) and w_k.size2() == 1:
            self.w = vertcat(self.w, w_k)
            self.lbw = vertcat(self.lbw, lbw_k)
            self.ubw = vertcat(self.ubw, ubw_k)
            self.w0 = vertcat(self.w0, w0_k)
        else:
            print("Wrong size in design constraints dummy")

    def add_g_cons(self, g_k, lbg_k, ubg_k):
        if g_k.size1() == len(lbg_k) and g_k.size1() == len(ubg_k) and len(lbg_k) == len(ubg_k) and g_k.size2() == 1:
            self.g = vertcat(self.g, g_k)
            self.lbg = vertcat(self.lbg, lbg_k)
            self.ubg = vertcat(self.ubg, ubg_k)
        else:
            print("Wrong size in general constraints dummy")

    def get_state_k(self, k):
        return vertcat(self.p_body[3 * k: 3 * (k + 1)], self.dp_body[3 * k: 3 * (k + 1)],
                       self.Omega[3 * k: 3 * (k + 1)], self.DOmega[3 * k: 3 * (k + 1)])

    def get_R_k(self, k):
        return self.R[:, 3 * k: 3 * (k + 1)]

    def get_dt_k(self, k):
        i = self.contacts.get_current_phase(k)
        return i, self.T[i] / self.contacts.step_list[i]

    def get_F_k(self, k, leg):
        offset = self.contacts.get_offset(leg)
        return self.F[offset + 3 * k: offset + 3 * (k + 1)]

    def get_p_feet_k(self, k, leg):
        offset = self.contacts.get_offset(leg)
        return self.p_feet[offset + 3 * k: offset + 3 * (k + 1)]

    def integrate_omega_history(self, R0, Omega_opt, T_opt):
        R_opt = np.zeros((self.contacts.num_steps, 3, 3))
        R_k = R0
        for k in range(self.contacts.num_steps):
            i = self.contacts.get_current_phase(k)
            dt = T_opt[i] / self.contacts.step_list[i]
            if k != 0:
                R_k = np.matmul(R_k, self.approximate_exp_a(skew(Omega_opt[:, k] * dt), self.p.e_terms))
            R_opt[k, 0:3, 0:3] = R_k
        return R_opt

    @staticmethod
    def approximate_exp_a(a, deg):
        exp_a = DM(np.zeros((3, 3)))
        for i in range(deg):
            exp_a = exp_a + (mpower(a, i) / np.math.factorial(i))
        return exp_a


class MotionProfile:
    def __init__(self, p, mappings, motion_type, x0, R0, p_feet0, xf, Rf, p_feetf, tt_g, sagittal=False, gait_repeat=0):
        self.mappings = mappings
        self.motion_type = motion_type

        # COM position (world in the world frame) bounding constraint
        self.p_body_b = np.array([[-inf, inf],
                                  [-inf, inf],
                                  [0, inf]])

        # COM position time derivative (in the world frame) bounding constraint
        self.dp_body_b = np.array([[-inf, inf],
                                   [-inf, inf],
                                   [-inf, inf]])

        # Angular velocity (w.r.t body) bounding constraint
        self.Omega_b = np.array([[-inf, inf],
                                 [-inf, inf],
                                 [-inf, inf]])

        # Angular velocity time derivative (w.r.t body) bounding constraint
        self.DOmega_b = np.array([[-inf, inf],
                                  [-inf, inf],
                                  [-inf, inf]])

        # Contact force limits
        # TODO change this to a general constraint where the normal force is bounded
        self.f_b = np.array([[-inf, inf],
                             [-inf, inf],
                             [0, p.f_max]])

        # Foot position (in the world frame) bounding constraints
        self.p_feet_b = np.array([[-inf, inf],
                                  [-inf, inf],
                                  [0, inf]])

        # Convenience constraint to force the stance phase feet to stay on the ground
        self.p_feet_b_ground = np.array([[-inf, inf],
                                         [-inf, inf],
                                         [0, 0]])

        # Restrict the decision variables to the sagittal plane, with a tolerance s_b
        if sagittal:
            s_b = [-0.1, 0.1]
            self.p_body_b[1] = s_b
            self.dp_body_b[1] = s_b
            self.Omega_b[2] = s_b
            self.f_b[1] = s_b

        # Convenience variable
        self.x_b = np.vstack((self.p_body_b, self.dp_body_b, self.Omega_b, self.DOmega_b))

        self.t_g, step_list, contact_list = self.generate_contact_pattern(p.step_count, gait_repeat)
        self.t_g = tt_g * np.array(self.t_g)

        p.contacts.initialize_contacts(step_list, contact_list)
        n_steps = p.contacts.num_steps
        n_legs = p.contacts.num_legs

        t_sum = sum(self.t_g)
        self.tMin = t_sum * 0.5
        self.tMax = t_sum * 1.5
        self.t_b = np.hstack((np.ones((p.contacts.num_cons, 1)) * self.tMin / (2 * p.contacts.num_cons),
                              np.ones((p.contacts.num_cons, 1)) * self.tMax))
        self.x_g = np.hstack((x0[:, None], np.zeros((12, n_steps - 2)), xf[:, None]))
        dp_body_avg = (xf[0:3] - x0[0:3]) / t_sum
        pos = x0[0:3]
        for k in range(n_steps - 1):
            i = p.contacts.get_current_phase(k)
            self.x_g[0:3, k] = pos
            dt = self.t_g[i] / p.contacts.step_list[i]
            pos = pos + dt * dp_body_avg

        self.p_body_g, self.dp_body_g, self.Omega_g, self.DOmega_g = np.vsplit(self.x_g, 4)
        slerp = Slerp([0, n_steps], rp.concatenate([rp.from_matrix(R0), rp.from_matrix(Rf)]))
        self.R_ref = slerp(np.linspace(0, n_steps, n_steps)).as_matrix()
        p_feet0 = np.reshape(p_feet0, (n_legs * 3, 1))
        p_feetf = np.reshape(p_feetf, (n_legs * 3, 1))
        self.p_feet_g = np.vstack((p_feet0, np.zeros((n_legs * 3 * (n_steps - 2), 1)),
                                   p_feetf))
        self.f_g = np.zeros((n_legs * 3 * n_steps, 1))
        dp_feet_avg = (p_feetf - p_feet0) / (sum(self.t_g))
        pos = p_feet0
        for k in range(n_steps):
            i = p.contacts.get_current_phase(k)
            if k < n_steps:
                for leg in range(n_legs):
                    offset = p.contacts.get_offset(leg)
                    self.p_feet_g[offset + 3 * k: offset + 3 * (k + 1)] = pos[leg * 3: (leg + 1) * 3]
            dt = self.t_g[i] / p.contacts.step_list[i]
            pos = pos + dt * dp_feet_avg
            clegs = sum(contact_list[i])
            if clegs > 0:
                for leg in range(n_legs):
                    offset = p.contacts.get_offset(leg)
                    self.f_g[offset + 3 * k: offset + 3 * (k + 1)] = np.reshape([0, 0, p.g_accel[2] * p.mass / clegs],
                                                                                (3, 1))

    def generate_contact_pattern(self, step_count, gait_repeat):
        gait_generator = GaitGenerator(self.mappings)
        t_g, step_list, contact_list = gait_generator.get_combos(self.motion_type, step_count, gait_repeat)
        return t_g, step_list, contact_list


class GaitGenerator:
    def __init__(self, mappings):
        self.generator = None
        if mappings == 'biped':
            self.generator = BipedGaitGenerator()
        elif mappings == 'quadruped':
            self.generator = QuadrupedGaitGenerator()
        else:
            print('Invalid mappings')

    def get_combos(self, motion_type, step_count, gait_repeat):
        return self.generator.get_combos(motion_type, step_count, gait_repeat)


class BipedGaitGenerator:
    def __init__(self):
        # [L, R]

        # Flight phase
        self.I = [0, 0]

        # Left support
        self.P = [1, 0]

        # Right support
        self.b = [0, 1]

        # Double support
        self.B = [1, 1]

    def get_combos(self, motion_type, step_count, gait_repeat):
        if motion_type == 'walk':
            t = []
            s = []
            c = []
            t_, s_, c_ = self.get_stride_stand(step_count, 0.2)
            t.extend(t_), s.extend(s_), c.extend(c_)
            for i in range(gait_repeat + 1):
                t_, s_, c_ = self.get_stride_walk(step_count)
                t.extend(t_), s.extend(s_), c.extend(c_)
            t_, s_, c_ = self.get_stride_stand(step_count, 0.2)
            t.extend(t_), s.extend(s_), c.extend(c_)
            return (np.array(t, dtype='f') / sum(t)).tolist(), s, c
        else:
            print('Not implemented yet')
            return 0

    def get_stride_stand(self, step_count, t):
        time_proportion_list = [t]
        step_list = [step_count]
        contact = [self.B]
        return time_proportion_list, step_list, contact

    def get_stride_walk(self, step_count):
        time_proportion_list = [0.3, 0.05, 0.3, 0.05]
        step_list = [step_count, step_count, step_count, step_count]
        contact = [self.b, self.B, self.P, self.B]
        return time_proportion_list, step_list, contact


class QuadrupedGaitGenerator:
    def __init__(self):
        # [LF, RF, LH, RH]

        # Flight phase
        self.II = [0, 0, 0, 0]

        # 1 one stance leg
        self.PI = [0, 0, 1, 0]
        self.bI = [0, 0, 0, 1]
        self.IP = [1, 0, 0, 0]
        self.Ib = [0, 1, 0, 0]

        # 2 stance legs
        self.Pb = [0, 1, 1, 0]
        self.bP = [1, 0, 0, 1]
        self.BI = [0, 0, 1, 1]
        self.IB = [1, 1, 0, 0]
        self.PP = [1, 0, 1, 0]
        self.bb = [0, 1, 0, 1]

        # 3 stance legs
        self.Bb = [0, 1, 1, 1]
        self.BP = [1, 0, 1, 1]
        self.bB = [1, 1, 0, 1]
        self.PB = [1, 1, 1, 0]

        # Stance phase
        self.BB = [1, 1, 1, 1]

    def get_combos(self, motion_type, step_count, gait_repeat):
        if motion_type == 'walk':
            t = []
            s = []
            c = []
            t_, s_, c_ = self.get_stride_stand(step_count, 0.3)
            t.extend(t_), s.extend(s_), c.extend(c_)
            for i in range(gait_repeat + 1):
                t_, s_, c_ = self.get_stride_trot_fly(step_count)
                t.extend(t_), s.extend(s_), c.extend(c_)
            t_, s_, c_ = self.get_stride_trot_fly_end(step_count)
            t.extend(t_), s.extend(s_), c.extend(c_)
            t_, s_, c_ = self.get_stride_stand(step_count, 0.3)
            t.extend(t_), s.extend(s_), c.extend(c_)
            return (np.array(t, dtype='f') / sum(t)).tolist(), s, c
        elif motion_type == 'backflip':
            t = []
            s = []
            c = []
            t_, s_, c_ = self.get_stride_stand(step_count, 0.3)
            t.extend(t_), s.extend(s_), c.extend(c_)
            t_, s_, c_ = self.get_backflip(step_count)
            t.extend(t_), s.extend(s_), c.extend(c_)
            t_, s_, c_ = self.get_stride_stand(step_count, 0.3)
            t.extend(t_), s.extend(s_), c.extend(c_)
            return (np.array(t, dtype='f') / sum(t)).tolist(), s, c
        else:
            print('Not implemented yet')
            return 0

    def get_stride_stand(self, step_count, t):
        time_proportion_list = [t]
        step_list = [step_count]
        contact = [self.BB]
        return time_proportion_list, step_list, contact

    def get_stride_trot_fly(self, step_count):
        time_proportion_list = [0.4, 0.1, 0.4, 0.1]
        step_list = [step_count, step_count, step_count, step_count]
        contact = [self.bP, self.II, self.Pb, self.II]
        return time_proportion_list, step_list, contact

    def get_stride_trot_fly_end(self, step_count):
        time_proportion_list = [0.4]
        step_list = [step_count]
        contact = [self.bP]
        return time_proportion_list, step_list, contact

    def get_backflip(self, step_count):
        time_proportion_list = [0.333, 0.69]
        step_list = [step_count, step_count]
        contact = [self.IB, self.II]
        return time_proportion_list, step_list, contact


def leg_mask(pos, leg, mappings):
    if mappings == 'biped':
        if leg == 0:
            return pos
        elif leg == 1:
            return [pos[0], -pos[1], pos[2]]
        else:
            print('Invalid leg number')
    elif mappings == 'quadruped':
        if leg == 0:
            return pos
        elif leg == 1:
            return [pos[0], -pos[1], pos[2]]
        elif leg == 2:
            return [-pos[0], pos[1], pos[2]]
        elif leg == 3:
            return [-pos[0], -pos[1], pos[2]]
        else:
            print('Invalid leg number')


def foot_positions(p_body, R, p_foot, height, mappings):
    if mappings == 'biped':
        num_legs = 2
    elif mappings == 'quadruped':
        num_legs = 4
    else:
        print('Invalid mapping')
        return
    p_foot = np.append(p_foot, height - p_body[2])
    p_feet = []
    for leg in range(num_legs):
        p_feet.append(p_body + np.matmul(R, leg_mask(p_foot, leg, mappings)))
    return np.array(p_feet)


# Determines the relative leg locations
# mappings = 'biped'
mappings = 'quadruped'

# Mass of the SRB
# mass = 37.0
mass = 2.50000279

# Inertia of SRB
# inertia = np.array([[3.09249-1, 0, 0],
#                     [0, 5.106100-1, 0],
#                     [0, 0, 6.939757-1]])
inertia = np.array([[3.09249e-2, 0, 0],
                    [0, 5.106100e-2, 0],
                    [0, 0, 6.939757e-2]])

# Sphere of radius r that the foot position is constrained to
r = 0.2375

# Foot position of the nominal leg directly below the hip in the x-y direction.
# The nominal leg mappings =
#   for 'biped' it is the left leg
#   for 'quadruped' it is the front left leg
# pbar = np.array([0.0125, 0.0775])
# pbar = np.array([0.075, 0.0775])
pbar = np.array([0.194, 0.1479])

# The sth foot position is constrained in a sphere of radius r to satisfy
# joint limits. This parameter is the center of the sphere w.r.t the COM
# p_feet_bar = foot_positions(np.array([0, 0, 1.0]), np.eye(3), pbar, -1.0, mappings)
p_feet_bar = foot_positions(np.array([0, 0, 0.3]), np.eye(3), pbar, -0.16, mappings)

# Maximum ground reaction force in the normal direction
# f_max = 500.0
f_max = 40.0

# Initial States
# p_body0 = np.array([0, 0, 1.0])
p_body0 = np.array([0, 0, 0.3])
tmp = np.zeros((9,))
x0 = np.hstack((p_body0, np.zeros((9,))))
R0 = rp.from_euler('zxy', [0, 0, 0], True).as_matrix()

# Final States
# p_bodyf = np.array([1.0, 0.0, 1.0])
p_bodyf = np.array([0.2, 0.0, 0.3])
xf = np.hstack((p_bodyf, np.zeros((9,))))
Rf = rp.from_euler('zxy', [0, 0, 0], True).as_matrix()

# Place the feet on the ground and directly below the hip
p_feet0 = foot_positions(p_body0, R0, pbar, 0.0, mappings)
p_feetf = foot_positions(p_bodyf, Rf, pbar, 0.0, mappings)

# Object that holds information about constant parameters
parameters = Parameters(mass, inertia, r, p_feet_bar, f_max)

# Roughly what type of action do you want the robot to take?
# This object calculates the initial guess and assigns tuning parameters to make the program converge better
# Currently supports: ['walk', 'jump', 'spinning_jump', 'diagonal_jump', 'barrel_roll', 'backflip']
motion_profile = MotionProfile(parameters, mappings, 'walk', x0, R0, p_feet0, xf, Rf, p_feetf, 3.0, True, 2)

# Stores constraints and the objective as Casadi functions
constraints = Constraints(parameters)

# Combines the states, constraints, initial guesses, and objective into a nonlinear program solvable by Ipopt
acrobatics = BiQuAcrobatics(parameters, motion_profile, constraints)

# Solve the program
acrobatics.optimize()
