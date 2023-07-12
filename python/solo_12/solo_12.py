from casadi import *
import numpy as np
from scipy.spatial.transform import Rotation as rp
from scipy.spatial.transform import Slerp


class DesignField:
    def __init__(self, field_name, size, split_flag):
        self.field_name = field_name
        self.size = size
        self.split_flag = split_flag
        self.indices = []


class CTConstraints:
    def __init__(self, field_name_list, size_list, split_list):
        self.field_name_list = field_name_list
        self.fields = {field_name: DesignField(field_name, size, split_flag) for (field_name, size, split_flag) in
                       zip(field_name_list, size_list, split_list)}
        self.current_index = 0
        self.w = []
        self.lbw = []
        self.ubw = []
        self.w0 = []
        self.g = []
        self.lbg = []
        self.ubg = []

    def add_general_constraints(self, g_k, lbg_k, ubg_k):
        if g_k.size1() == len(lbg_k) and g_k.size1() == len(ubg_k) and len(lbg_k) == len(ubg_k):
            self.g = vertcat(self.g, g_k)
            self.lbg = vertcat(self.lbg, lbg_k)
            self.ubg = vertcat(self.ubg, ubg_k)
        else:
            print("Wrong size in general constraints dummy")

    def add_design_constraints(self, w_k, lbw_k, ubw_k, w0_k, field_name):
        n = w_k.size1()
        if n == len(lbw_k) and n == len(ubw_k) and len(lbw_k) == len(ubw_k):
            a = self.current_index
            b = self.current_index + n
            f_field = self.fields[field_name]
            self.w = vertcat(self.w, w_k)
            self.lbw = vertcat(self.lbw, lbw_k)
            self.ubw = vertcat(self.ubw, ubw_k)
            self.w0 = vertcat(self.w0, w0_k)
            f_field.indices = vertcat(f_field.indices, range(a, b)).full().flatten()
            self.current_index = b
        else:
            print("Wrong size in design constraints dummy")

    def unpack_opt_indices(self, opt, field_name):
        f_field = self.fields[field_name]
        n = len(f_field.indices)
        s1, s2 = f_field.size
        if f_field.split_flag:
            opt_design_vars = np.zeros((s1, s2, int(floor(n / s2))))
            for i in range(int(floor(n / s2))):
                for j in range(s2):
                    tmp_var = f_field.indices[i:i + s2 - 1]
                    opt_design_vars[:, j, i] = np.reshape(opt[tmp_var], s1)
            return opt_design_vars
        else:
            return np.reshape(opt[f_field.indices], (int(floor(n / (s1 * s2))), s2, s1))

    def unpack_all(self, opt_x, check_violations):
        solution = {field_name: {"opt_x": self.unpack_opt_indices(opt_x, field_name),
                                 "lb_x": self.unpack_opt_indices(self.lbw, field_name),
                                 "ub_x": self.unpack_opt_indices(self.ubw, field_name)}
                    for (field_name) in self.field_name_list}
        if check_violations:
            for field_name in self.field_name_list:
                f_field = self.fields[field_name]
                s1, s2 = f_field.size
                val = solution[field_name]
                n = len(self.fields[field_name].indices)
                for k in range(n):
                    if np.any(val["opt_x"].flatten()[k] < val["lb_x"].flatten()[k]):
                        print(field_name + " violated LOWER bound at iteration " + str(int(floor(k / (s1 * s2)))))
                    if np.any(val["opt_x"].flatten()[k] > val["ub_x"].flatten()[k]):
                        print(field_name + " violated UPPER bound at iteration " + str(int(floor(k / (s1 * s2)))))
        return solution


class Contacts:
    def __init__(self, step_list, contact_list):
        self.num_cons = len(step_list)
        self.step_list = step_list
        self.num_steps = int(np.sum(step_list))
        self.cum_steps = np.cumsum(step_list)
        self.contact_list = contact_list

    def get_current_phase(self, k):
        i = 0
        for j in range(self.num_cons):
            i = i + (k >= self.cum_steps[j])
        return i


class MotionProfile:
    def __init__(self, motion_type, step_list, mass, g_accel, r, p_feet_bar, p_feet0,
                 f_max, p_body0, dp_body0, Omega0, DOmega0, R0, p_bodyf, Rf):
        self.motion_type = motion_type
        self.step_list = step_list

        self.eOmega = None
        self.eF = None
        self.eR = None
        self.beta = None
        self.gamma = None
        self.tMin = None
        self.tMax = None
        self.M = None
        self.cons = None
        self.t_imin = None
        self.t_imax = None
        self.t_guess = None

        self.p_guess = None
        self.dp_guess = None
        self.acc_ref = None
        self.f_ref = None
        self.Omega_guess = None
        self.DOmega_guess = None
        self.R_ref = None

        self.generate_parameters()
        self.generate_guesses(mass, g_accel, r, p_feet_bar, p_feet0, f_max, p_body0, dp_body0, Omega0, DOmega0, R0,
                              p_bodyf, Rf)

    def generate_parameters(self):
        # Defaults

        # Omega cost weight
        eOmega = 5e-5

        # Force cost weight
        eF = 1e-6

        # Rotation error cost weight
        eR = 1e-3

        # Scaling parameter for initial trajectory guess
        beta = -0.1

        # Scaling parameter for initial trajectory guess
        gamma = 0.405

        # Minimum total time
        tMin = 1.0

        # Maximum total time
        tMax = 2.25

        # Rk4 steps per interval
        M = 4

        # Contact pattern
        # TODO make a simple function that, given a motion type and starting and ending position and orientation,
        #  determines the contact list (i.e., automatically choose the feet to barrel roll from the left vs the right)
        contact_list = [[1, 1, 1, 1], [0, 0, 0, 0]]

        match self.motion_type:
            case 'jump':
                tMin = 0.5
                tMax = 2.0
            case 'spinning_jump':
                tMin = 0.5
                tMax = 2.0
            case 'diagonal_jump':
                contact_list = [[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]]
            case 'barrel_roll':
                contact_list = [[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0]]
            case 'backflip':
                contact_list = [[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]]

        # Number of Contacts
        num_cons = len(contact_list)

        # Bounds and guesses for the contact timings. They are *very* sensitive parameters because they are used to
        # generate the initial ballistic trajectory that seeds the solver
        # TODO make a heuristic to get good initial guesses for each motion type
        t_imin = np.ones((num_cons, 1)) * (tMin / (num_cons + 1))
        t_imax = np.ones((num_cons, 1)) * tMax
        t_guess = np.ones((num_cons, 1)) * ((tMax - tMin) / num_cons)

        match self.motion_type:
            case 'jump':
                pass
            case 'spinning_jump':
                pass
            case 'diagonal_jump':
                t_guess = np.array([0.333, 0.666, 0.333])
            case 'barrel_roll':
                pass
            case 'backflip':
                pass

        self.eOmega = eOmega
        self.eF = eF
        self.eR = eR
        self.beta = beta
        self.gamma = gamma
        self.tMin = tMin
        self.tMax = tMax
        self.M = M
        self.cons = Contacts(self.step_list, contact_list)
        self.t_imin = t_imin
        self.t_imax = t_imax
        self.t_guess = t_guess

    def generate_guesses(self, mass, g_accel, r, p_feet_bar, p_feet0, f_max, p_body0, dp_body0, Omega0, DOmega0, R0,
                         p_bodyf, Rf):
        # # Initialize the initial guess matrices
        # self.p_guess = np.hstack((p_body0[:, None], np.zeros((3, self.cons.num_steps - 2)), p_bodyf[:, None]))
        # self.dp_guess = np.hstack((dp_body0[:, None], np.zeros((3, self.cons.num_steps - 1))))
        # self.acc_ref = np.zeros((3, self.cons.num_steps))
        # self.f_ref = np.zeros((3, 4, self.cons.num_steps))
        # self.Omega_guess = np.hstack((Omega0[:, None], np.zeros((3, self.cons.num_steps - 1))))
        # self.DOmega_guess = np.hstack((DOmega0[:, None], np.zeros((3, self.cons.num_steps - 1))))
        #
        # # Find time of liftoff for the initial guess
        # it = None
        # for it in range(self.cons.num_cons):
        #     if self.cons.contact_list[it] == [0, 0, 0, 0]:
        #         break
        # t_lo = self.t_guess.cumsum()[it - 1]
        # lo_steps = np.array(self.cons.step_list).cumsum()[it - 1]
        #
        # DT = SX.sym('DT')
        # x1 = SX.sym('x1_ref')
        # x2 = SX.sym('x2_ref')
        # u1 = SX.sym('u1_ref')
        # x = vertcat(x1, x2)
        # u = vertcat(u1)
        # xdot = vertcat(x2, u1)
        # f1 = Function('f1_ref', [x, u, DT], [xdot, DT])
        # X0 = SX.sym('X0', 2)
        # U = SX.sym('U')
        # X = X0
        # for j in range(self.M):
        #     k1, DT = f1(X, U, DT)
        #     k2, DT = f1(X + DT / 2 * k1, U, DT)
        #     k3, DT = f1(X + DT / 2 * k2, U, DT)
        #     k4, DT = f1(X + DT * k3, U, DT)
        #     X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # F1 = Function('F', [X0, U, DT], [X], ['x0', 'u', 'dt'], ['xf'])
        #
        # # Find z-component of the acceleration reference
        # t = 0
        # for k in range(lo_steps - 1):
        #     i = self.cons.get_current_phase(k)
        #     dt = self.t_guess[i] / self.cons.step_list[i]
        #     p_body_k = self.p_guess[2, k]
        #     dp_body_k = self.dp_guess[2, k]
        #     t = t + dt
        #     grf = 0
        #     clegs = self.cons.contact_list[i][0] + self.cons.contact_list[i][1] + \
        #             self.cons.contact_list[i][2] + self.cons.contact_list[i][3]
        #     grf = grf + clegs * f_max
        #     ddp_body = ((self.beta + (self.gamma * t / t_lo)) * grf / mass)
        #     if k == 0:
        #         self.acc_ref[2, k] = (self.beta * grf / mass)
        #     self.acc_ref[2, k + 1] = ddp_body
        #     Fk1 = F1(x0=[p_body_k, dp_body_k], u=ddp_body, dt=dt / self.M)
        #     self.p_guess[2, k + 1] = Fk1['xf'][0]
        #     self.dp_guess[2, k + 1] = Fk1['xf'][1]
        # self.acc_ref[2, lo_steps: self.cons.num_steps] = \
        #     -np.ones(1, self.cons.num_steps - lo_steps) * g_accel[2]
        # # Find the time-of-flight to reach p_bodyf
        # t_ = np.roots([-(1 / 2) * g_accel[2], self.dp_guess[2, lo_steps - 1],
        #                self.p_guess[2, lo_steps - 1] - p_bodyf[2]])
        # # print(self.p_guess)
        # # print(self.p_guess[2, lo_steps - 1])
        # # print(-p_bodyf[2] + self.p_guess[2, lo_steps - 1])
        # # print(self.dp_guess[2, lo_steps - 1])
        # print(self.acc_ref[2, :])
        # t_fl = t_[0]
        # if t_[0] < 0:
        #     t_fl = t_[1]
        # self.t_guess[it] = t_fl
        # t_la = self.t_guess.sum()
        # avg_acc = np.array(2 * (p_bodyf[0:2] - self.p_guess[0:2, 0] - self.dp_guess[0:2, 0] * t_la) / np.power(t_la, 2))
        # # print(self.t_guess)
        # # print(avg_acc)
        #
        # # r = (2 * avg_acc) / t_la
        # r = avg_acc
        # # print(self.t_guess)
        # # print(avg_acc)
        # # print(r)
        #
        # # Integrate the xy-component of the acceleration references to get the position and velocity reference
        # t = 0
        # for k in range(self.cons.num_steps - 1):
        #     i = self.cons.get_current_phase(k)
        #     dt = self.t_guess[i] / self.cons.step_list[i]
        #     p_body_k = self.p_guess[:, k]
        #     dp_body_k = self.dp_guess[:, k]
        #     # if k < lo_steps:
        #     self.acc_ref[0:2, k] = r
        #     # self.acc_ref[0:2, k] = (t_la / t_lo) * r * t
        #     ddp_body = self.acc_ref[:, k]
        #     t = t + dt
        #     Fk1x = F1(x0=[p_body_k[0], dp_body_k[0]], u=ddp_body[0], dt=dt / self.M)
        #     Fk1y = F1(x0=[p_body_k[1], dp_body_k[1]], u=ddp_body[1], dt=dt / self.M)
        #     Fk1z = F1(x0=[p_body_k[2], dp_body_k[2]], u=ddp_body[2], dt=dt / self.M)
        #     p_body_next = np.reshape([Fk1x['xf'][0], Fk1y['xf'][0], Fk1z['xf'][0]], (3,))
        #     dp_body_next = np.reshape([Fk1x['xf'][1], Fk1y['xf'][1], Fk1z['xf'][1]], (3,))
        #     self.p_guess[:, k + 1] = p_body_next
        #     self.dp_guess[:, k + 1] = dp_body_next
        #
        #     clegs = self.cons.contact_list[i][0] + self.cons.contact_list[i][1] + \
        #             self.cons.contact_list[i][2] + self.cons.contact_list[i][3]
        #     if clegs > 0:
        #         tmp = np.array([0, 0, self.acc_ref[2, k]])
        #         self.f_ref[:, 0, k] = (self.cons.contact_list[i][0] / clegs) * (tmp + g_accel) * mass
        #         self.f_ref[:, 1, k] = (self.cons.contact_list[i][1] / clegs) * (tmp + g_accel) * mass
        #         self.f_ref[:, 2, k] = (self.cons.contact_list[i][2] / clegs) * (tmp + g_accel) * mass
        #         self.f_ref[:, 3, k] = (self.cons.contact_list[i][3] / clegs) * (tmp + g_accel) * mass
        #
        # # print(self.p_guess[0])
        # # print(self.dp_guess)
        # # print(self.acc_ref)
        # # print(self.f_ref[:, 2, :])

        # Reference Trajectory
        slerp = Slerp([0, self.cons.num_steps], rp.concatenate([rp.from_matrix(R0), rp.from_matrix(Rf)]))
        self.R_ref = slerp(np.linspace(0, self.cons.num_steps, self.cons.num_steps)).as_matrix()


def approximate_exp_a(a, deg):
    exp_a = DM(np.zeros((3, 3)))
    for i in range(deg):
        exp_a = exp_a + (mpower(a, i) / np.math.factorial(i))
    return exp_a


def approximate_log_a(a, deg):
    log_a = DM(np.zeros((3, 3)))
    for i in range(1, deg):
        log_a = log_a + power(-1, i + 1) * mpower(a - DM(np.eye(3)), i) / i
    return log_a


def integrate_omega_history(cons, R0, Omega_opt, T_opt, e_terms):
    R_opt = np.zeros((cons.num_steps, 3, 3))
    R_k = R0
    for k in range(cons.num_steps):
        i = cons.get_current_phase(k)
        dt = T_opt[0, 0, i] / cons.step_list[i]
        if k != 0:
            R_k = np.matmul(R_k, approximate_exp_a(skew(Omega_opt[k, :, :] * dt), e_terms))
        R_opt[k, 0:3, 0:3] = transpose(R_k)
    return R_opt


def leg_mask(pos, leg):
    if leg == 1:
        return pos
    elif leg == 2:
        return [pos[0], -pos[1], pos[2]]
    elif leg == 3:
        return [-pos[0], pos[1], pos[2]]
    elif leg == 4:
        return [-pos[0], -pos[1], pos[2]]
    else:
        print('Invalid leg number')
        return 0


def foot_positions(p_body, R, tmp):
    tmp = np.append(tmp, -p_body[2])
    return np.array([p_body + np.matmul(R, leg_mask(tmp, 1)),
                     p_body + np.matmul(R, leg_mask(tmp, 2)),
                     p_body + np.matmul(R, leg_mask(tmp, 3)),
                     p_body + np.matmul(R, leg_mask(tmp, 4))]).transpose()


# n x 2 matrix of lower[1] and upper[2] bounds
def rand_in_bounds(bounds):
    n = len(bounds)
    r = np.zeros((n, 1))
    for i in range(n):
        r[i, 0] = rand_in_range(bounds[i, :])
    return r


def rand_in_range(bound):
    return (bound[1] - bound[0]) * np.random.rand() + bound[0]


# Constant Parameters

# Mass of the Solo_12 SRB
mass = 2.50000279

# Inertia of Solo_12 SRB
inertia = np.array([[3.09249e-2, 0, 0],
                    [0, 5.106100e-2, 0],
                    [0, 0, 6.939757e-2]])

# Acceleration due to gravity
g_accel = np.array([0, 0, 9.81])

# Sphere of radius r that the foot position is constrained to
r = 0.2375

# Friction coefficient
mu = 0.7

# Maximum ground reaction force in the Z direction
# (friction constraints imply the force in the X and Y direction have to be less than this)
f_max = 20

# Degree of Taylor series approximation for matrix exponential
e_terms = 8

# Degree of Taylor series approximation for matrix logarithm
l_terms = 8

# Steps per contact phase
step_list = [30, 30, 30]

# Initial States
p_body0 = np.array([0, 0, 0.3])
dp_body0 = np.array([0, 0, 0])
Omega0 = np.array([0, 0, 0])
DOmega0 = np.array([0, 0, 0])
R0 = rp.from_euler('xyz', [0, 0, 0], True).as_matrix()

# Final States
p_bodyf = np.array([0.4, 0.3, 0.3])
Rf = rp.from_euler('xyz', [0, 0, 45], True).as_matrix()

# Place the feet below the hip
tmp = np.array([0.194, 0.1479])
p_feet0 = foot_positions(p_body0, R0, tmp)
p_feetf = foot_positions(p_bodyf, Rf, tmp)

# The sth foot position is constrained in a sphere of radius r to satisfy
# joint limits. This parameter is the center of the sphere w.r.t the COM
pbar = [0.194, 0.1479, -0.16]
p_feet_bar = np.array([leg_mask(pbar, 1), leg_mask(pbar, 2), leg_mask(pbar, 3), leg_mask(pbar, 4)]).transpose()

# Roughly what type of action do you want the robot to take?
# This only influences the initial guess and some tuning parameters to make the program converge better
# ['jump', 'spinning_jump', 'diagonal_jump', 'barrel_roll', 'backflip']
mp = MotionProfile('diagonal_jump', step_list, mass, g_accel, r, p_feet_bar, p_feet0, f_max, p_body0, dp_body0, Omega0,
                   DOmega0, R0, p_bodyf, Rf)

# GRF limits
f_bounds = np.array([[-inf, inf],
                     [-inf, inf],
                     [0, f_max]])

# COM bounding constraint. Ideally you would set this to some section of a
# tube each timestep within you want the trajectory to lie
p_body_bounds = np.array([[-inf, inf],
                          [-inf, inf],
                          [0, inf]])

# Velocity bounds to make the problem more solvable
dp_body_bounds = np.array([[-inf, inf],
                           [-inf, inf],
                           [-inf, inf]])

# Angular velocity bounds to make the problem more solvable
Omega_bounds = np.array([[-inf, inf],
                         [-inf, inf],
                         [-inf, inf]])

# Time derivative angular velocity bounds to make the problem more solvable
DOmega_bounds = np.array([[-inf, inf],
                          [-inf, inf],
                          [-inf, inf]])

# Decision Variables

# Set up constraint structure
fields = ['T', 'p_body', 'dp_body', 'Omega', 'DOmega', 'F_0', 'F_1', 'F_2', 'F_3']
sizes = [(mp.cons.num_cons, 1), (3, 1), (3, 1), (3, 1), (3, 1), (3, 1), (3, 1), (3, 1), (3, 1)]
split_flags = [False, False, False, False, False, False, False, False, False]
constraints = CTConstraints(fields, sizes, split_flags)

# COM of the body (3x1)
p_body = []

# Time derivative of the COM of the body (3x1)
dp_body = []

# Angular velocity of SRB w.r.t body frame (3x1)
Omega = []

# Time derivative of angular velocity of SRB w.r.t body frame (3x1)
DOmega = []

# Rotation matrix of the body frame (3x3)
# Note: This is not a decision variable; It is dependent on the history of Omega
R = []

# GRF on each foot (3x1)
F_0 = []
F_1 = []
F_2 = []
F_3 = []

# Optimal contact timing for the ith contact phase (n_px1)
T = MX.sym('T', mp.cons.num_cons, 1)

# Total time must be within our bounds
constraints.add_general_constraints(sum1(T), [mp.tMin], [mp.tMax])

# All contact timings must be positive
constraints.add_design_constraints(T, mp.t_imin,
                                   mp.t_imax,
                                   mp.t_guess, 'T')
R_k = R0
for k in range(mp.cons.num_steps):
    i = mp.cons.get_current_phase(k)
    dt = T[i] / mp.cons.step_list[i]

    p_body = vertcat(p_body, MX.sym('p_body_k{}'.format(k), 3, 1))
    dp_body = vertcat(dp_body, MX.sym('dp_body_k{}'.format(k), 3, 1))
    Omega_k = MX.sym('Omega_k{}'.format(k), 3, 1)
    Omega = vertcat(Omega, Omega_k)
    DOmega = vertcat(DOmega, MX.sym('DOmega_k{}'.format(k), 3, 1))
    if k == 0:
        R = horzcat(R, R_k)
    else:
        R_k = mtimes(R_k, approximate_exp_a(skew(Omega_k * dt), e_terms))
        R = horzcat(R, R_k)

    if mp.cons.contact_list[mp.cons.get_current_phase(k)][0]:
        F_0 = vertcat(F_0, MX.sym('F_0_k{}'.format(k), 3, 1))
    if mp.cons.contact_list[mp.cons.get_current_phase(k)][1]:
        F_1 = vertcat(F_1, MX.sym('F_1_k{}'.format(k), 3, 1))
    if mp.cons.contact_list[mp.cons.get_current_phase(k)][2]:
        F_2 = vertcat(F_2, MX.sym('F_2_k{}'.format(k), 3, 1))
    if mp.cons.contact_list[mp.cons.get_current_phase(k)][3]:
        F_3 = vertcat(F_3, MX.sym('F_3_k{}'.format(k), 3, 1))

J = 0
for k in range(mp.cons.num_steps):
    # Gather Decision Variables
    i = mp.cons.get_current_phase(k)
    dt = T[i] / mp.cons.step_list[i]

    # COM of the body (3x1)
    p_body_k = p_body[3 * k: 3 * (k + 1)]

    # Time derivative of the COM of the body (3x1)
    dp_body_k = dp_body[3 * k: 3 * (k + 1)]

    # Angular velocity of SRB w.r.t body frame (3x1)
    Omega_k = Omega[3 * k: 3 * (k + 1)]

    # Time derivative of angular velocity of SRB w.r.t body frame (3x1)
    DOmega_k = DOmega[3 * k: 3 * (k + 1)]

    # Rotation matrix of the body frame (3x3)
    R_k = R[:, 3 * k: 3 * (k + 1)]
    R_ref_k = mp.R_ref[k]

    if k != 0:
        constraints.add_design_constraints(dp_body_k, dp_body_bounds[:, 0], dp_body_bounds[:, 1], dp_body0,
                                           'dp_body')
        constraints.add_design_constraints(Omega_k, Omega_bounds[:, 0], Omega_bounds[:, 1], Omega0, 'Omega')
        constraints.add_design_constraints(DOmega_k, DOmega_bounds[:, 0], DOmega_bounds[:, 1],
                                           np.zeros((3, 1)), 'DOmega')

        if k != mp.cons.num_steps - 1:
            # Add body bounding box constraints
            if i == mp.cons.num_cons - 1:
                p_interp = (k - mp.cons.cum_steps[i - 1]) * (p_bodyf - p_body0) / mp.cons.step_list[i]
                constraints.add_design_constraints(p_body_k, p_body_bounds[:, 0], p_body_bounds[:, 1],
                                                   p_interp, 'p_body')
            else:
                constraints.add_design_constraints(p_body_k, p_body_bounds[:, 0], p_body_bounds[:, 1],
                                                   p_body0, 'p_body')

    # Add friction cone, GRF, and foot position constraints to each leg
    grf = np.zeros((3, 1))
    tau = np.zeros((3, 1))
    clegs = mp.cons.contact_list[i][0] + mp.cons.contact_list[i][1] + \
            mp.cons.contact_list[i][2] + mp.cons.contact_list[i][3]
    if mp.cons.contact_list[i][0]:
        # GRF on each foot (3x1)
        F_0_k = F_0[3 * k: 3 * (k + 1)]
        grf = grf + F_0_k
        tau = tau + cross(F_0_k, (p_body_k - p_feet0[:, 0]))
        constraints.add_general_constraints(fabs(F_0_k[0] / F_0_k[2]), [0], [mu])
        constraints.add_general_constraints(fabs(F_0_k[1] / F_0_k[2]), [0], [mu])
        constraints.add_general_constraints(norm_2(mtimes(R_k, (p_feet0[:, 0] - p_body_k)) - p_feet_bar[:, 0]),
                                            [0], [r])
        # constraints.add_design_constraints(F_0_k, f_bounds[:, 0], f_bounds[:, 1], mp.f_ref[:, 0, k], 'F_0')
        constraints.add_design_constraints(F_0_k, f_bounds[:, 0], f_bounds[:, 1], [0, 0, g_accel[2] * mass / clegs],
                                           'F_0')
    if mp.cons.contact_list[i][1]:
        F_1_k = F_1[3 * k: 3 * (k + 1)]
        grf = grf + F_1_k
        tau = tau + cross(F_1_k, (p_body_k - p_feet0[:, 1]))
        constraints.add_general_constraints(fabs(F_1_k[0] / F_1_k[2]), [0], [mu])
        constraints.add_general_constraints(fabs(F_1_k[1] / F_1_k[2]), [0], [mu])
        constraints.add_general_constraints(norm_2(mtimes(R_k, (p_feet0[:, 1] - p_body_k)) - p_feet_bar[:, 1]),
                                            [0], [r])
        # constraints.add_design_constraints(F_1_k, f_bounds[:, 0], f_bounds[:, 1], mp.f_ref[:, 1, k], 'F_1')
        constraints.add_design_constraints(F_1_k, f_bounds[:, 0], f_bounds[:, 1], [0, 0, g_accel[2] * mass / clegs],
                                           'F_1')
    if mp.cons.contact_list[i][2]:
        F_2_k = F_2[3 * k: 3 * (k + 1)]
        grf = grf + F_2_k
        tau = tau + cross(F_2_k, (p_body_k - p_feet0[:, 2]))
        constraints.add_general_constraints(fabs(F_2_k[0] / F_2_k[2]), [0], [mu])
        constraints.add_general_constraints(fabs(F_2_k[1] / F_2_k[2]), [0], [mu])
        constraints.add_general_constraints(norm_2(mtimes(R_k, (p_feet0[:, 2] - p_body_k)) - p_feet_bar[:, 2]),
                                            [0], [r])
        # constraints.add_design_constraints(F_2_k, f_bounds[:, 0], f_bounds[:, 1], mp.f_ref[:, 2, k], 'F_2')
        constraints.add_design_constraints(F_2_k, f_bounds[:, 0], f_bounds[:, 1], [0, 0, g_accel[2] * mass / clegs],
                                           'F_2')
    if mp.cons.contact_list[i][3]:
        F_3_k = F_3[3 * k: 3 * (k + 1)]
        grf = grf + F_3_k
        tau = tau + cross(F_3_k, (p_body_k - p_feet0[:, 3]))
        constraints.add_general_constraints(fabs(F_3_k[0] / F_3_k[2]), [0], [mu])
        constraints.add_general_constraints(fabs(F_3_k[1] / F_3_k[2]), [0], [mu])
        constraints.add_general_constraints(norm_2(mtimes(R_k, (p_feet0[:, 3] - p_body_k)) - p_feet_bar[:, 3]),
                                            [0], [r])
        # constraints.add_design_constraints(F_3_k, f_bounds[:, 0], f_bounds[:, 1], mp.f_ref[:, 3, k], 'F_3')
        constraints.add_design_constraints(F_3_k, f_bounds[:, 0], f_bounds[:, 1], [0, 0, g_accel[2] * mass / clegs],
                                           'F_3')

    # Discrete dynamics
    if k < mp.cons.num_steps - 1:
        p_body_k1 = p_body[3 * (k + 1): 3 * (k + 2)]
        dp_body_k1 = dp_body[3 * (k + 1): 3 * (k + 2)]
        Omega_k1 = Omega[3 * (k + 1): 3 * (k + 2)]
        DOmega_k1 = DOmega[3 * (k + 1): 3 * (k + 2)]

        ddp_body = ((grf / mass) - g_accel.reshape((3, 1)))
        p_body_next = p_body_k + (dp_body_k * dt) + ((1 / 2) * ddp_body * power(dt, 2))
        dp_body_next = dp_body_k + ddp_body * dt
        Omega_next = Omega_k + DOmega_k * dt
        DOmega_next = mtimes(np.linalg.inv(inertia),
                             (mtimes(transpose(R_k), tau) - cross(Omega_k, mtimes(inertia, Omega_k))))

        constraints.add_general_constraints(p_body_k1 - p_body_next, np.zeros((3, 1)), np.zeros((3, 1)))
        constraints.add_general_constraints(Omega_k1 - Omega_next, np.zeros((3, 1)), np.zeros((3, 1)))
        constraints.add_general_constraints(dp_body_k1 - dp_body_next, np.zeros((3, 1)), np.zeros((3, 1)))
        constraints.add_general_constraints(DOmega_k1 - DOmega_next, np.zeros((3, 1)), np.zeros((3, 1)))

    # Initial States
    if k == 0:
        constraints.add_design_constraints(p_body_k, p_body0, p_body0, p_body0, 'p_body')
        constraints.add_design_constraints(dp_body_k, dp_body0, dp_body0, dp_body0, 'dp_body')
        constraints.add_design_constraints(Omega_k, Omega0, Omega0, Omega0, 'Omega')
        constraints.add_design_constraints(DOmega_k, DOmega0, DOmega0, DOmega0, 'DOmega')

    if k == mp.cons.num_steps - 1:
        constraints.add_general_constraints(reshape(R_k, (9, 1)), np.reshape(Rf, (9, 1)), np.reshape(Rf, (9, 1)))
        constraints.add_design_constraints(p_body_k, p_bodyf, p_bodyf, p_bodyf, 'p_body')
        for leg in range(4):
            constraints.add_general_constraints(norm_2(mtimes(R_k, (p_feetf[:, leg] - p_body_k)) - p_feet_bar[:, leg]),
                                                [0], [r])

    # Objective Function
    e_R_k = inv_skew(approximate_log_a(mtimes(transpose(R_ref_k), R_k), l_terms))
    # e_R_k = (1/2) * inv_skew(mtimes(transpose(R_ref_k), R_k) - mtimes(transpose(R_k), R_ref_k))
    J = J + (mp.eOmega * mtimes(transpose(Omega_k), Omega_k))
    J = J + (mp.eF * mtimes(transpose(grf), grf))
    J = J + (mp.eR * mtimes(transpose(e_R_k), e_R_k))

x = constraints.w
lbx = constraints.lbw
ubx = constraints.ubw
x0 = constraints.w0

# Initialize an NLP solver
nlp = {'x': x, 'f': J, 'g': constraints.g}

# Solver options
opts = {"expand": True,
        "detect_simple_bounds": True, "warn_initial_bounds": True,
        "ipopt": {"max_iter": 100,
                  "fixed_variable_treatment": "make_constraint",
                  "hessian_approximation": "limited-memory",
                  "mumps_mem_percent": 10000,
                  "print_level": 5}}

# Allocate a solver
solver = nlpsol("solver", "ipopt", nlp, opts)

# Solve the NLP
sol = solver(x0=x0, lbg=constraints.lbg, ubg=constraints.ubg, lbx=lbx, ubx=ubx)
sol_f = sol["f"]
sol_x = sol["x"]
sol_lam_x = sol["lam_x"]
sol_lam_g = sol["lam_g"]

solution = constraints.unpack_all(sol_x, False)
T_opt = solution["T"]["opt_x"]
p_body_opt = solution["p_body"]["opt_x"]
dp_body_opt = solution["dp_body"]["opt_x"]
Omega_opt = solution["Omega"]["opt_x"]
DOmega_opt = solution["DOmega"]["opt_x"]
R_opt = integrate_omega_history(mp.cons, R0, Omega_opt, T_opt, e_terms)
F_0_opt = solution["F_0"]["opt_x"]
F_1_opt = solution["F_1"]["opt_x"]
F_2_opt = solution["F_2"]["opt_x"]
F_3_opt = solution["F_3"]["opt_x"]

T_opt = T_opt.reshape(T_opt.shape[0], -1)
p_body_opt = p_body_opt.reshape(p_body_opt.shape[0], -1)
dp_body_opt = dp_body_opt.reshape(dp_body_opt.shape[0], -1)
Omega_opt = Omega_opt.reshape(Omega_opt.shape[0], -1)
DOmega_opt = DOmega_opt.reshape(DOmega_opt.shape[0], -1)
R_opt = R_opt.reshape(mp.cons.num_steps, 9)
R_guess = np.array(mp.R_ref).reshape(mp.cons.num_steps, 9)
F_0_opt = F_0_opt.reshape(F_0_opt.shape[0], -1)
F_1_opt = F_1_opt.reshape(F_1_opt.shape[0], -1)
F_2_opt = F_2_opt.reshape(F_2_opt.shape[0], -1)
F_3_opt = F_3_opt.reshape(F_3_opt.shape[0], -1)

# T_guess = mp.t_guess
# p_body_guess = mp.p_guess
# dp_body_guess = mp.dp_guess
# Omega_guess = mp.Omega_guess
# DOmega_guess = mp.DOmega_guess
# F0_guess = mp.f_ref[:, 0, :]
# F1_guess = mp.f_ref[:, 1, :]
# F2_guess = mp.f_ref[:, 2, :]
# F3_guess = mp.f_ref[:, 3, :]
# F0_guess.reshape(F0_guess.shape[0], -1)
# F1_guess.reshape(F1_guess.shape[0], -1)
# F2_guess.reshape(F2_guess.shape[0], -1)
# F3_guess.reshape(F3_guess.shape[0], -1)

# Print solution
# print("-----")
print("objective at solution =", sol_f)
# print("-----")
print("primal solution =", sol_x)
# print("-----")
# print("dual solution (x) =", sol_lam_x)
# print("-----")
# print("dual solution (g) =", sol_lam_g)
# print("-----")

np.savetxt('solo_12/opt/sol_f.csv', sol_f, delimiter=',')
np.savetxt('solo_12/opt/sol_x.csv', sol_x, delimiter=',')
np.savetxt('solo_12/opt/sol_lam_x.csv', sol_lam_x, delimiter=',')
np.savetxt('solo_12/opt/sol_lam_g.csv', sol_lam_g, delimiter=',')

np.savetxt('solo_12/opt/T_opt.csv', np.transpose(np.array(T_opt)), delimiter=',')
np.savetxt('solo_12/opt/p_body_opt.csv', p_body_opt, delimiter=',')
np.savetxt('solo_12/opt/dp_body_opt.csv', dp_body_opt, delimiter=',')
np.savetxt('solo_12/opt/Omega_opt.csv', Omega_opt, delimiter=',')
np.savetxt('solo_12/opt/DOmega_opt.csv', DOmega_opt, delimiter=',')
np.savetxt('solo_12/opt/R_opt.csv', R_opt, delimiter=',')
np.savetxt('solo_12/initial_guess/R_guess.csv', R_guess, delimiter=',')
np.savetxt('solo_12/opt/F0_opt.csv', F_0_opt, delimiter=',')
np.savetxt('solo_12/opt/F1_opt.csv', F_1_opt, delimiter=',')
np.savetxt('solo_12/opt/F2_opt.csv', F_2_opt, delimiter=',')
np.savetxt('solo_12/opt/F3_opt.csv', F_3_opt, delimiter=',')

# np.savetxt('solo_12/initial_guess/T_guess.csv', T_guess, delimiter=',')
# np.savetxt('solo_12/initial_guess/p_body_guess.csv', p_body_guess, delimiter=',')
# np.savetxt('solo_12/initial_guess/dp_body_guess.csv', dp_body_guess, delimiter=',')
# np.savetxt('solo_12/initial_guess/Omega_guess.csv', Omega_guess, delimiter=',')
# np.savetxt('solo_12/initial_guess/DOmega_guess.csv', DOmega_guess, delimiter=',')
# np.savetxt('solo_12/initial_guess/F0_guess.csv', F0_guess, delimiter=',')
# np.savetxt('solo_12/initial_guess/F1_guess.csv', F1_guess, delimiter=',')
# np.savetxt('solo_12/initial_guess/F2_guess.csv', F2_guess, delimiter=',')
# np.savetxt('solo_12/initial_guess/F3_guess.csv', F3_guess, delimiter=',')

np.savetxt('solo_12/metadata/step_list.csv', np.array(step_list), delimiter=',')
np.savetxt('solo_12/metadata/contact_list.csv', np.array(mp.cons.contact_list), delimiter=',')
np.savetxt('solo_12/metadata/p_feet0.csv', p_feet0, delimiter=',')
np.savetxt('solo_12/metadata/p_feetf.csv', p_feetf, delimiter=',')
np.savetxt('solo_12/metadata/p_feet_bar.csv', np.array(p_feet_bar), delimiter=',')
np.savetxt('solo_12/metadata/r.csv', [r], delimiter=',')
