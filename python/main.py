from casadi import *
import numpy as np
from scipy.spatial.transform import Rotation as rp
from scipy.spatial.transform import Slerp


class LogMap(Callback):
    def __init__(self, name, d, opts={}):
        Callback.__init__(self)
        self.d = d
        self.construct(name, opts)

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        return Sparsity.dense(3, 3)

    def get_sparsity_out(self, i):
        return Sparsity.dense(1, 1)

    def eval(self, arg):
        x_mat = arg[0]
        r_err = mtimes(transpose(self.d), x_mat)
        q = self.matrix_to_quat(r_err)
        e_r = self.quat_to_axis_angle(q)
        scalarized = mtimes(transpose(e_r), e_r)
        # jacobian of this quadratic form is 2transpose(x)dx
        return [scalarized]

    def matrix_to_quat(self, r):
        tr = trace(r)
        if trace(r) > 0:
            t = sqrt(1 + tr)
            q_r = (0.5 * t)
            q_v = (0.5 / t) * inv_skew(r - transpose(r))
            q_x = q_v[0]
            q_y = q_v[1]
            q_z = q_v[2]
        elif r[0, 0] >= r[1, 1] and r[0, 0] >= r[2, 2]:
            t = sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
            q_r = (0.5 / t) * (r[2, 1] - r[1, 2])
            q_x = (0.5 * t)
            q_y = (0.5 / t) * (r[1, 0] + r[0, 1])
            q_z = (0.5 / t) * (r[2, 0] + r[0, 2])
        elif r[1, 1] > r[0, 0] and r[1, 1] >= r[2, 2]:
            t = sqrt(1 + r[1, 1] - r[2, 2] - r[0, 0])
            q_r = (0.5 / t) * (r[0, 2] - r[2, 0])
            q_x = (0.5 / t) * (r[0, 1] + r[1, 0])
            q_y = (0.5 * t)
            q_z = (0.5 / t) * (r[2, 1] + r[1, 2])
        elif r[2, 2] > r[0, 0] and r[2, 2] > r[1, 1]:
            t = sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1])
            q_r = (0.5 / t) * (r[1, 0] - r[0, 1])
            q_x = (0.5 / t) * (r[0, 2] + r[2, 0])
            q_y = (0.5 / t) * (r[1, 2] + r[2, 1])
            q_z = (0.5 * t)
        else:
            q_r = 0
            q_x = 0
            q_y = 0
            q_z = 0
            print("This should not be possible.")
        return vertcat(q_r, q_x, q_y, q_z)

    def quat_to_axis_angle(self, q):
        q = sign(q) * q
        q_r, q_x, q_y, q_z = vertsplit(q)
        q_v = vertcat(q_x, q_y, q_z)
        if norm_2(q_v) < 1e-3:
            return ((2 / q_r) - (2 / 3) * (power(norm_2(q_v), 2) / power(q_r, 3))) * q_v
        else:
            return 4 * atan2(norm_2(q_v), (q_r + sqrt(power(q_r, 2) + power(norm_2(q_v), 2)))) * (q_v / norm_2(q_v))


class DesignField:
    def __init__(self, size, split_flag):
        self.size = size
        self.split_flag = split_flag
        self.indices = []
        self.w = []
        self.lbw = []
        self.ubw = []
        self.w0 = []


class CTConstraints:
    def __init__(self, field_name_list, size_list, split_list):
        self.fields = {field_name: DesignField(size, split_flag) for (field_name, size, split_flag) in
                       zip(field_name_list, size_list, split_list)}
        self.current_index = 0
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
            f_field.w = vertcat(f_field.w, w_k)
            f_field.lbw = vertcat(f_field.lbw, lbw_k)
            f_field.ubw = vertcat(f_field.ubw, ubw_k)
            f_field.w0 = vertcat(f_field.w0, w0_k)
            f_field.indices = vertcat(f_field.indices, range(a, b))
            self.current_index = b
        else:
            print("Wrong size in design constraints dummy")

    def unpack_indices(self, w_opt, field_name):
        f_field = self.fields[field_name]
        n = len(f_field.indices)
        s1, s2 = f_field.size
        if f_field.split_flag:
            opt_design_vars = np.zeros((s1, s2, floor(n / s2)))
            for i in range(floor(n / s2)):
                for j in range(s2):
                    tmp_var = f_field.indices[i:i + s2 - 1]
                    opt_design_vars[:, j, i] = reshape(w_opt[vertcat(tmp_var[j]), 1], s1, 1)
            return opt_design_vars
        else:
            return np.reshape(w_opt[f_field.indices], (s1, s2, n))


def approximate_exp_a(a, deg):
    exp_a = DM(np.zeros((3, 3)))
    for i in range(deg):
        exp_a = exp_a + (mpower(a, i) / np.math.factorial(i))
    return exp_a


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


def get_current_phase(k, nch):
    i = 0
    for j in range(len(nch)):
        i = i + (k > nch[j])
    return i


# n x 2 matrix of lower[1] and upper[2] bounds
def rand_in_bounds(bounds):
    n = len(bounds)
    r = np.zeros((n, 1))
    for i in range(n):
        r[i, 0] = (bounds[i, 1] - bounds[i, 0]) * np.random.rand() + bounds[i, 0]
    return r


# Constant Parameters

# Omega cost weight
eOmega = 0.0875

# Force cost weight
eF = 0.0009125

# Rotation error cost weight
eR = 0.1

# Minimum total time
tMin = 0.5

# Maximum total time
tMax = 1.5

# Number of contact phases
n_p = 2

# Predefined number of steps for the ith contact phase
N = np.ones((n_p, 1), dtype=int) * 3

# Helper for checking what contact we are in
Nch = np.cumsum(N)

# Total number of steps
Nc = np.sum(N)

# Friction coefficient
mu = 0.7

# Very small number
eps = 1e-9

# GRF limits
f_bounds = np.array([[-75, 75],
                     [-75, 75],
                     [-100, 100],
                     [-75, 75],
                     [-75, 75],
                     [-100, 100],
                     [-75, 75],
                     [-75, 75],
                     [-100, 100],
                     [-75, 75],
                     [-75, 75],
                     [-100, 100]])

# Acceleration due to gravity
g_accel = np.array([[0.0], [0.0], [-9.81]])

# COM bounding constraint. Ideally you would set this to some section of a
# tube each timestep within you want the trajectory to lie
p_body_bounds = np.array([[-1.0, 1.0],
                          [-1.0, 1.0],
                          [-eps, 0.625]])

# Velocity bounds to make the problem more solvable
dp_body_bounds = np.array([[-3, 3],
                           [-3, 3],
                           [-20.0, 20.0]])

# Simple bounding box for all feet
p_feet_bounds = np.array([[-1.0, 1.0],
                          [-1.0, 1.0],
                          [-eps, 0.5],
                          [-1.0, 1.0],
                          [-1.0, 1.0],
                          [-eps, 0.5],
                          [-1.0, 1.0],
                          [-1.0, 1.0],
                          [-eps, 0.5],
                          [-1.0, 1.0],
                          [-1.0, 1.0],
                          [-eps, 0.5]])

# Angular velocity bounds to make the problem more solvable
Omega_bounds = np.array([[-1, 1],
                         [-1, 1],
                         [-1, 1]])

# Time derivative angular velocity bounds to make the problem more solvable
DOmega_bounds = np.array([[-4, 4],
                          [-4, 4],
                          [-4, 4]])

# The sth foot position is constrained in a sphere of radius r to satisfy
# joint limits. This parameter is the center of the sphere w.r.t the COM
pbar = [0.23, 0.19, -0.3]
p_feet_bar = np.array([leg_mask(pbar, 1), leg_mask(pbar, 2), leg_mask(pbar, 3), leg_mask(pbar, 4)]).transpose()

# Sphere of radius r that the foot position is constrained to
r = np.ones((3, 1)) * 0.25

# Mass of the SRB
mass = 2.50000279

# Inertia of SRB
inertia = np.array([[3.09249e-2, -9.00101e-7, 1.865287e-5],
                    [-8.00101e-7, 5.106100e-2, 1.245813e-4],
                    [1.865287e-5, 1.245813e-4, 6.939757e-2]])
inv_inertia = np.linalg.inv(inertia)

# Decision Variables

# Set up constraint structure
fields = ['p_body', 'dp_body', 'p_feet', 'Omega', 'DOmega', 'R', 'F', 'T']
sizes = [(3, 1), (3, 1), (3, 4), (3, 1), (3, 1), (3, 3), (3, 4), (n_p, 1)]
split_flags = [False, False, True, False, False, False, True, False]
constraints = CTConstraints(fields, sizes, split_flags)

# COM of the body (3x1)
p_body = []

# Time derivative of the COM of the body (3x1)
dp_body = []

# Foot position in the world frame of each foot (3x4)
p_feet = []

# Angular velocity of SRB w.r.t body frame (3x1)
Omega = []

# Time derivative of angular velocity of SRB w.r.t body frame (3x1)
DOmega = []

# Rotation matrix of the body frame (3x3)
R = []

# GRF on each foot (3x4)
F = []

# Logarithm function callback
log_callback_fun_helper = []

# Rotation matrix scalarized error function (3x3 -> 1x1)
log_callback_fun = []

for k in range(Nc):
    p_body = vertcat(p_body, MX.sym('p_body_k{}'.format(k), 3, 1))
    dp_body = vertcat(dp_body, MX.sym('dp_body_k{}'.format(k), 3, 1))
    p_feet = horzcat(p_feet, MX.sym('p_feet_k{}'.format(k), 3, 4))
    Omega = vertcat(Omega, MX.sym('Omega_k{}'.format(k), 3, 1))
    DOmega = vertcat(DOmega, MX.sym('DOmega_k{}'.format(k), 3, 1))
    R = horzcat(R, MX.sym('R_k{}'.format(k), 3, 3))
    F = horzcat(F, MX.sym('F_k{}'.format(k), 3, 4))

# Optimal contact timing for the ith contact phase (n_px1)
T = MX.sym('T', n_p, 1)

# Total time must be within our bounds
constraints.add_general_constraints(sum1(T), [tMin], [tMax])

# All contact timings must be positive
constraints.add_design_constraints(T, np.ones((n_p, 1)) * (tMin / (n_p + 1)), np.ones((n_p, 1)) * tMax,
                                   np.ones((n_p, 1)) * ((tMax - tMin) / n_p), 'T')

# Initial States
p_body0 = [0, 0, 0.35]
dp_body0 = np.zeros((3, 1))
R0 = rp.from_euler('xyz', [-0.2, 0.1, 5], True)
tmp = [0.1345, 0.1873, 0.0313]
p_feet0 = np.array([p_body0 + np.matmul(R0.as_matrix(), leg_mask(tmp, 1)),
                    p_body0 + np.matmul(R0.as_matrix(), leg_mask(tmp, 2)),
                    p_body0 + np.matmul(R0.as_matrix(), leg_mask(tmp, 3)),
                    p_body0 + np.matmul(R0.as_matrix(), leg_mask(tmp, 4))]).transpose()
Omega0 = np.zeros((3, 1))
DOmega0 = np.zeros((3, 1))

# Final States
p_bodyf = [0.388, -0.194, 0.25]
Rf = rp.from_euler('xyz', [2, -6, -30], True)
tmp = [0.5470, 0.0113, 0.0339]
p_feetf = np.array([p_bodyf + np.matmul(Rf.as_matrix(), leg_mask(tmp, 1)),
                    p_bodyf + np.matmul(Rf.as_matrix(), leg_mask(tmp, 2)),
                    p_bodyf + np.matmul(Rf.as_matrix(), leg_mask(tmp, 3)),
                    p_bodyf + np.matmul(Rf.as_matrix(), leg_mask(tmp, 4))]).transpose()

# Reference Trajectory
slerp = Slerp([0, Nc], rp.concatenate([R0, Rf]))
R_ref = slerp(np.linspace(0, Nc))
J = 0
for k in range(Nc):
    # Gather Decision Variables
    i = get_current_phase(k, Nch)
    dt = T[i] / N[i]

    # COM of the body (3x1)
    p_body_k = p_body[3 * k: 3 * (k + 1)]

    # Time derivative of the COM of the body (3x1)
    dp_body_k = dp_body[3 * k: 3 * (k + 1)]

    # Foot position in the world frame of each foot (3x4)
    p_feet_k = p_feet[:, 4 * k: 4 * (k + 1)]

    # Angular velocity of SRB w.r.t body frame (3x1)
    Omega_k = Omega[3 * k: 3 * (k + 1)]

    # Time derivative of angular velocity of SRB w.r.t body frame (3x1)
    DOmega_k = DOmega[3 * k: 3 * (k + 1)]

    # Rotation matrix of the body frame (3x3)
    R_k = reshape(R[:, 3 * k: 3 * (k + 1)], (3, 3))

    R_ref_k = R_ref[k].as_matrix()

    # GRF on each foot (3x4)
    F_k = F[:, 4 * k: 4 * (k + 1)]

    if k != 0:
        # Add dummy constraints
        constraints.add_design_constraints(dp_body_k, dp_body_bounds[:, 0], dp_body_bounds[:, 1], dp_body0, 'dp_body')
        constraints.add_design_constraints(Omega_k, Omega_bounds[:, 0], Omega_bounds[:, 1], Omega0, 'Omega')
        constraints.add_design_constraints(DOmega_k, DOmega_bounds[:, 0], DOmega_bounds[:, 1], DOmega0, 'DOmega')
        if k != Nc - 1:
            # Add body bounding box constraints
            constraints.add_design_constraints(p_body_k, p_body_bounds[:, 0], p_body_bounds[:, 1], p_body0, 'p_body')
            if i == 0:
                constraints.add_design_constraints(reshape(p_feet_k, (12, 1)), np.reshape(p_feet0, (12, 1)),
                                                   np.reshape(p_feet0, (12, 1)), np.reshape(p_feet0, (12, 1)), 'p_feet')
            else:
                constraints.add_design_constraints(reshape(p_feet_k, (12, 1)), p_feet_bounds[:, 0], p_feet_bounds[:, 1],
                                                   rand_in_bounds(p_feet_bounds), 'p_feet')


            # for leg in range(4):
                # if i == 0:
                #     constraints.add_design_constraints(p_feet_k[:, leg], p_feet0[:, 0], p_feet0[:, 0],
                #                                        p_feet0[:, 0], 'p_feet')
                # else:
                #     constraints.add_design_constraints(p_feet_k[:, leg], p_feet_bounds[:, 0], p_feet_bounds[:, 1],
                #                                        rand_in_bounds(p_feet_bounds), 'p_feet')
            # Add dummy rotation constraints
            constraints.add_design_constraints(reshape(R_k, (9, 1)), np.ones((9, 1)) * (-1.05), np.ones((9, 1)) * 1.05,
                                               reshape(rp.random().as_matrix(), (9, 1)), 'R')

        # Add friction cone, GRF, and foot position constraints to each leg
        grf = np.zeros((3, 1))
        tau = np.zeros((3, 1))
        for leg in range(4):
            tau = tau + cross(F_k[:, leg], (p_body_k - p_feet_k[:, leg]))
            if i == 1:
                grf = grf + F_k[:, leg]
                constraints.add_general_constraints(fabs(F_k[0, leg] / F_k[2, leg]), [0], [mu])
                constraints.add_general_constraints(fabs(F_k[1, leg] / F_k[2, leg]), [0], [mu])
            constraints.add_general_constraints(fabs(mtimes(R_k, (p_feet_k[:, leg] - p_body_k)) - p_feet_bar[:, leg]),
                                                np.zeros((3, 1)), r)
            # constraints.add_design_constraints(F_k[:, leg], f_bounds[:, 0], f_bounds[:, 1], [0, 0, mass / 4], 'F')
        constraints.add_design_constraints(reshape(F_k, (12, 1)), f_bounds[:, 0], f_bounds[:, 1], [0, 0, mass / 4, 0, 0, mass / 4, 0, 0, mass / 4, 0, 0, mass / 4], 'F')

        # Discrete dynamics
        if k < Nc - 1:
            p_body_k1 = p_body[3 * (k + 1): 3 * (k + 2)]
            dp_body_k1 = dp_body[3 * (k + 1): 3 * (k + 2)]
            Omega_k1 = Omega[3 * (k + 1): 3 * (k + 2)]
            DOmega_k1 = DOmega[3 * (k + 1): 3 * (k + 2)]
            R_k1 = reshape(R[:, 3 * (k + 1): 3 * (k + 2)], (3, 3))

            p_body_next = p_body_k + dp_body_k * dt
            dp_body_next = dp_body_k + ((grf / mass) + g_accel) * dt
            Omega_next = Omega_k + DOmega_k * dt
            DOmega_next = DOmega_k + mtimes(inv_inertia, (mtimes(transpose(R_k), tau) -
                                                          cross(Omega_k, mtimes(inertia, Omega_k)))) * dt
            R_next = mtimes(R_k, approximate_exp_a(skew(Omega_k * dt), 20))

            constraints.add_general_constraints(p_body_k1 - p_body_next, np.zeros((3, 1)), np.zeros((3, 1)))
            constraints.add_general_constraints(dp_body_k1 - dp_body_next, np.zeros((3, 1)), np.zeros((3, 1)))
            constraints.add_general_constraints(Omega_k1 - Omega_next, np.zeros((3, 1)), np.zeros((3, 1)))
            constraints.add_general_constraints(DOmega_k1 - DOmega_next, np.zeros((3, 1)), np.zeros((3, 1)))
            constraints.add_general_constraints(reshape(R_k1 - R_next, 9, 1), np.zeros((9, 1)), np.zeros((9, 1)))

        # Initial States
        if k == 0:
            constraints.add_design_constraints(p_body_k, p_body0, p_body0, p_body0, 'p_body')
            constraints.add_design_constraints(dp_body_k, dp_body0, dp_body0, dp_body0, 'dp_body')
            constraints.add_design_constraints(Omega_k, Omega0, Omega0, Omega0, 'Omega')
            constraints.add_design_constraints(DOmega_k, DOmega0, DOmega0, DOmega0, 'DOmega')

            constraints.add_design_constraints(reshape(p_feet_k, (12, 1)), np.reshape(p_feet0, (12, 1)), np.reshape(p_feet0, (12, 1)),
                                               np.reshape(p_feet0, (12, 1)), 'p_feet')
            # for leg in range(4):
                # constraints.add_design_constraints(p_feet_k[:, leg], p_feet0[:, leg], p_feet0[:, leg],
                #                                    p_feet0[:, leg], 'p_feet')
            constraints.add_design_constraints(reshape(R_k, (9, 1)), np.reshape(R0.as_matrix(), (9, 1)),
                                               np.reshape(R0.as_matrix(), (9, 1)),
                                               np.reshape(R0.as_matrix(), (9, 1)), 'R')
        if k == Nc - 1:
            constraints.add_design_constraints(p_body_k, p_bodyf, p_bodyf, p_bodyf, 'p_body')
            constraints.add_design_constraints(reshape(p_feet_k, (12, 1)), np.reshape(p_feetf, (12, 1)), np.reshape(p_feetf, (12, 1)), np.reshape(p_feetf, (12, 1)), 'p_feet')
            # for leg in range(4):
                # constraints.add_design_constraints(p_feet_k[:, leg], p_feetf[:, leg], p_feetf[:, leg],
                #                                    p_feetf[:, leg], 'p_feet')
            constraints.add_design_constraints(reshape(R_k, (9, 1)), np.reshape(Rf.as_matrix(), (9, 1)),
                                               np.reshape(Rf.as_matrix(), (9, 1)),
                                               np.reshape(Rf.as_matrix(), (9, 1)), 'R')

        # Objective Function

        log_callback_k = LogMap('log_callback_k{}'.format(k), R_ref_k, {"enable_fd": True})
        fun_k = Function('fun_k{}'.format(k), [R_k], [log_callback_k(R_k)])

        log_callback_fun_helper = [log_callback_fun_helper, log_callback_k]
        log_callback_fun = [log_callback_fun, fun_k]

        J = J + (eOmega * mtimes(transpose(Omega_k), Omega_k)) + (eF * mtimes(transpose(grf), grf)) + (eR * fun_k(R_k))

x = []
lbx = []
ubx = []
x0 = []

for field in fields:
    x = vertcat(x, constraints.fields[field].w)
    lbx = vertcat(lbx, constraints.fields[field].lbw)
    ubx = vertcat(ubx, constraints.fields[field].ubw)
    x0 = vertcat(x0, constraints.fields[field].w0)

# Initialize an NLP solver
nlp = {'x': x, 'f': J, 'g': constraints.g}

# Solver options
opts = {}

# Allocate a solver
solver = nlpsol("solver", "ipopt", nlp, opts)

# Solve the NLP
sol = solver(x0=x0, lbg=constraints.lbg, ubg=constraints.ubg, lbx=lbx, ubx=ubx)

# Print solution
print("-----")
solx = sol["x"]
print("primal solution = ")
print(solx)
print("-----")

