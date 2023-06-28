from casadi import *
import numpy as np
from scipy.spatial.transform import Rotation as rp
from scipy.spatial.transform import Slerp


class DesignField:
    def __init__(self, size, split_flag):
        self.size = size
        self.split_flag = split_flag
        self.indices = []


class CTConstraints:
    def __init__(self, field_name_list, size_list, split_list):
        self.fields = {field_name: DesignField(size, split_flag) for (field_name, size, split_flag) in
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

    def unpack_indices(self, w_opt, field_name):
        f_field = self.fields[field_name]
        n = len(f_field.indices)
        s1, s2 = f_field.size
        if f_field.split_flag:
            opt_design_vars = np.zeros((s1, s2, int(floor(n / s2))))
            for i in range(int(floor(n / s2))):
                for j in range(s2):
                    tmp_var = f_field.indices[i:i + s2 - 1]
                    opt_design_vars[:, j, i] = np.reshape(w_opt[tmp_var], s1)
            return opt_design_vars
        else:
            return np.reshape(w_opt[f_field.indices], (int(floor(n / (s1 * s2))), s2, s1))


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


def integrate_omega_history(cons, R0, Omega_opt, T_opt):
    R_opt = np.zeros((cons.num_steps, 3, 3))
    R_k = R0
    for k in range(cons.num_steps):
        i = cons.get_current_phase(k)
        dt = T_opt[0, 0, i]/cons.step_list[i]
        if k != 0:
            R_k = np.matmul(R_k, approximate_exp_a(skew(Omega_opt[k, :, :] * dt), 4))
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

# Omega cost weight
eOmega = 1e-2

# Force cost weight
eF = 1e-6

# Rotation error cost weight
eR = 1e-3

# Minimum total time
tMin = 0.5

# Maximum total time
tMax = 1.5

# Steps per contact phase
step_list = [30, 30, 30]

# Contact pattern
contact_list = [[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]]

# Specify the contact metadata
cons = Contacts(step_list, contact_list)

# Initial States
p_body0 = np.array([0, 0, 0.2])
dp_body0 = np.zeros((3, 1))
R0 = rp.from_euler('xyz', [0, 0, 0], True).as_matrix()
tmp1 = [0.194, 0.1479, -0.2]
p_feet0 = np.array([p_body0 + np.matmul(R0, leg_mask(tmp1, 1)),
                    p_body0 + np.matmul(R0, leg_mask(tmp1, 2)),
                    p_body0 + np.matmul(R0, leg_mask(tmp1, 3)),
                    p_body0 + np.matmul(R0, leg_mask(tmp1, 4))]).transpose()
Omega0 = np.zeros((3, 1))
DOmega0 = np.zeros((3, 1))

# Final States
p_bodyf = np.array([0.4, 0.3, 0.25])
Rf = rp.from_euler('xyz', [0, 0, 45], True).as_matrix()
tmp2 = [0.194, 0.1479, -0.25]
p_feetf = np.array([p_bodyf + np.matmul(Rf, leg_mask(tmp2, 1)),
                    p_bodyf + np.matmul(Rf, leg_mask(tmp2, 2)),
                    p_bodyf + np.matmul(Rf, leg_mask(tmp2, 3)),
                    p_bodyf + np.matmul(Rf, leg_mask(tmp2, 4))]).transpose()

# The sth foot position is constrained in a sphere of radius r to satisfy
# joint limits. This parameter is the center of the sphere w.r.t the COM
pbar = [0.194, 0.1479, -0.16]
p_feet_bar = np.array([leg_mask(pbar, 1), leg_mask(pbar, 2), leg_mask(pbar, 3), leg_mask(pbar, 4)]).transpose()

# Sphere of radius r that the foot position is constrained to
r = 0.2375

# Friction coefficient
mu = 0.7

# GRF limits
f_bounds = np.array([[-25, 25],
                     [-25, 25],
                     [-0.01, 35]])

# Acceleration due to gravity
g_accel = np.array([[0], [0], [9.81]])

# COM bounding constraint. Ideally you would set this to some section of a
# tube each timestep within you want the trajectory to lie
p_body_bounds = np.array([[-1, 1],
                          [-1, 1],
                          [0, 1]])

# Velocity bounds to make the problem more solvable
dp_body_bounds = np.array([[-2.5, 2.5],
                           [-2.5, 2.5],
                           [-2.5, 2.5]])

# Angular velocity bounds to make the problem more solvable
Omega_bounds = np.array([[-pi, pi],
                         [-pi, pi],
                         [-pi, pi]])

# Time derivative angular velocity bounds to make the problem more solvable
DOmega_bounds = np.array([[-10, 10],
                          [-10, 10],
                          [-10, 10]])

# Mass of the SRB
mass = 2.50000279

# Inertia of SRB
inertia = np.array([[3.09249e-2, 0, 0],
                    [0, 5.106100e-2, 0],
                    [0, 0, 6.939757e-2]])
inv_inertia = np.linalg.inv(inertia)

# Decision Variables

# Set up constraint structure
fields = ['T', 'p_body', 'dp_body', 'Omega', 'DOmega', 'F_0', 'F_1', 'F_2', 'F_3']
sizes = [(cons.num_cons, 1), (3, 1), (3, 1), (3, 1), (3, 1), (3, 1), (3, 1), (3, 1), (3, 1)]
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
T = MX.sym('T', cons.num_cons, 1)

# Total time must be within our bounds
constraints.add_general_constraints(sum1(T), [tMin], [tMax])

# All contact timings must be positive
constraints.add_design_constraints(T, np.ones((cons.num_cons, 1)) * (tMin / (cons.num_cons + 1)),
                                   np.ones((cons.num_cons, 1)) * tMax,
                                   np.ones((cons.num_cons, 1)) * ((tMax - tMin) / cons.num_cons), 'T')
R_k = R0
for k in range(cons.num_steps):
    i = cons.get_current_phase(k)
    dt = T[i] / cons.step_list[i]

    p_body = vertcat(p_body, MX.sym('p_body_k{}'.format(k), 3, 1))
    dp_body = vertcat(dp_body, MX.sym('dp_body_k{}'.format(k), 3, 1))
    Omega_k = MX.sym('Omega_k{}'.format(k), 3, 1)
    Omega = vertcat(Omega, Omega_k)
    DOmega = vertcat(DOmega, MX.sym('DOmega_k{}'.format(k), 3, 1))
    if k == 0:
        R = horzcat(R, R_k)
    else:
        R_k = mtimes(R_k, approximate_exp_a(skew(Omega_k * dt), 4))
        R = horzcat(R, R_k)

    if cons.contact_list[cons.get_current_phase(k)][0]:
        F_0 = vertcat(F_0, MX.sym('F_0_k{}'.format(k), 3, 1))
    if cons.contact_list[cons.get_current_phase(k)][1]:
        F_1 = vertcat(F_1, MX.sym('F_1_k{}'.format(k), 3, 1))
    if cons.contact_list[cons.get_current_phase(k)][2]:
        F_2 = vertcat(F_2, MX.sym('F_2_k{}'.format(k), 3, 1))
    if cons.contact_list[cons.get_current_phase(k)][3]:
        F_3 = vertcat(F_3, MX.sym('F_3_k{}'.format(k), 3, 1))

# Reference Trajectory
slerp = Slerp([0, cons.num_steps], rp.concatenate([rp.from_matrix(R0), rp.from_matrix(Rf)]))
R_ref = slerp(np.linspace(0, cons.num_steps, cons.num_steps))

J = 0
for k in range(cons.num_steps):
    # Gather Decision Variables
    i = cons.get_current_phase(k)
    dt = T[i] / cons.step_list[i]

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
    R_ref_k = R_ref[k].as_matrix()

    if k != 0:
        # Add dummy constraints
        # if i == cons.num_cons-1:
        #     constraints.add_design_constraints(dp_body_k, dp_body_bounds[:, 0], dp_body_bounds[:, 1],
        #                                        (p_bodyf - p_body0)/cons.step_list[i], 'dp_body')
        # else:
        constraints.add_design_constraints(dp_body_k, dp_body_bounds[:, 0], dp_body_bounds[:, 1], dp_body0,
                                           'dp_body')
        constraints.add_design_constraints(Omega_k, Omega_bounds[:, 0], Omega_bounds[:, 1], Omega0, 'Omega')
        constraints.add_design_constraints(DOmega_k, DOmega_bounds[:, 0], DOmega_bounds[:, 1], DOmega0, 'DOmega')

        if k != cons.num_steps - 1:
            # Add body bounding box constraints
            if i == cons.num_cons - 1:
                p_interp = (k - cons.cum_steps[i - 1]) * (p_bodyf - p_body0) / cons.step_list[i]
                constraints.add_design_constraints(p_body_k, p_body_bounds[:, 0], p_body_bounds[:, 1],
                                                   p_interp, 'p_body')
            else:
                constraints.add_design_constraints(p_body_k, p_body_bounds[:, 0], p_body_bounds[:, 1],
                                                   p_body0, 'p_body')

    # Add friction cone, GRF, and foot position constraints to each leg
    grf = np.zeros((3, 1))
    tau = np.zeros((3, 1))
    clegs = cons.contact_list[i][0] + cons.contact_list[i][1] + \
            cons.contact_list[i][2] + cons.contact_list[i][3]
    if cons.contact_list[i][0]:
        # GRF on each foot (3x1)
        F_0_k = F_0[3 * k: 3 * (k + 1)]
        grf = grf + F_0_k
        tau = tau + cross(F_0_k, (p_body_k - p_feet0[:, 0]))
        constraints.add_general_constraints(fabs(F_0_k[0] / F_0_k[2]), [0], [mu])
        constraints.add_general_constraints(fabs(F_0_k[1] / F_0_k[2]), [0], [mu])
        constraints.add_general_constraints(norm_2(mtimes(R_k, (p_feet0[:, 0] - p_body_k)) - p_feet_bar[:, 0]),
                                            [0], [r])
        # constraints.add_design_constraints(F_0_k, f_bounds[:, 0], f_bounds[:, 1], rand_in_bounds(f_bounds), 'F_0')
        constraints.add_design_constraints(F_0_k, f_bounds[:, 0], f_bounds[:, 1], [0, 0, g_accel[2] * mass / clegs],
                                           'F_0')
    if cons.contact_list[i][1]:
        F_1_k = F_1[3 * k: 3 * (k + 1)]
        grf = grf + F_1_k
        tau = tau + cross(F_1_k, (p_body_k - p_feet0[:, 1]))
        constraints.add_general_constraints(fabs(F_1_k[0] / F_1_k[2]), [0], [mu])
        constraints.add_general_constraints(fabs(F_1_k[1] / F_1_k[2]), [0], [mu])
        constraints.add_general_constraints(norm_2(mtimes(R_k, (p_feet0[:, 1] - p_body_k)) - p_feet_bar[:, 1]),
                                            [0], [r])
        # constraints.add_design_constraints(F_1_k, f_bounds[:, 0], f_bounds[:, 1], rand_in_bounds(f_bounds), 'F_1')
        constraints.add_design_constraints(F_1_k, f_bounds[:, 0], f_bounds[:, 1], [0, 0, g_accel[2] * mass / clegs],
                                           'F_1')
    if cons.contact_list[i][2]:
        F_2_k = F_2[3 * k: 3 * (k + 1)]
        grf = grf + F_2_k
        tau = tau + cross(F_2_k, (p_body_k - p_feet0[:, 2]))
        constraints.add_general_constraints(fabs(F_2_k[0] / F_2_k[2]), [0], [mu])
        constraints.add_general_constraints(fabs(F_2_k[1] / F_2_k[2]), [0], [mu])
        constraints.add_general_constraints(norm_2(mtimes(R_k, (p_feet0[:, 2] - p_body_k)) - p_feet_bar[:, 2]),
                                            [0], [r])
        # constraints.add_design_constraints(F_2_k, f_bounds[:, 0], f_bounds[:, 1], rand_in_bounds(f_bounds), 'F_2')
        constraints.add_design_constraints(F_2_k, f_bounds[:, 0], f_bounds[:, 1], [0, 0, g_accel[2] * mass / clegs],
                                           'F_2')
    if cons.contact_list[i][3]:
        F_3_k = F_3[3 * k: 3 * (k + 1)]
        grf = grf + F_3_k
        tau = tau + cross(F_3_k, (p_body_k - p_feet0[:, 3]))
        constraints.add_general_constraints(fabs(F_3_k[0] / F_3_k[2]), [0], [mu])
        constraints.add_general_constraints(fabs(F_3_k[1] / F_3_k[2]), [0], [mu])
        constraints.add_general_constraints(norm_2(mtimes(R_k, (p_feet0[:, 3] - p_body_k)) - p_feet_bar[:, 3]),
                                            [0], [r])
        # constraints.add_design_constraints(F_3_k, f_bounds[:, 0], f_bounds[:, 1], rand_in_bounds(f_bounds), 'F_3')
        constraints.add_design_constraints(F_3_k, f_bounds[:, 0], f_bounds[:, 1], [0, 0, g_accel[2] * mass / clegs],
                                           'F_3')

    # Discrete dynamics
    if k < cons.num_steps - 1:
        p_body_k1 = p_body[3 * (k + 1): 3 * (k + 2)]
        dp_body_k1 = dp_body[3 * (k + 1): 3 * (k + 2)]
        Omega_k1 = Omega[3 * (k + 1): 3 * (k + 2)]
        DOmega_k1 = DOmega[3 * (k + 1): 3 * (k + 2)]

        ddp_body = ((grf / mass) - g_accel)
        p_body_next = p_body_k + (dp_body_k * dt) + ((1 / 2) * ddp_body * power(dt, 2))
        dp_body_next = dp_body_k + ddp_body * dt
        Omega_next = Omega_k + DOmega_k * dt
        DOmega_next = mtimes(inv_inertia, (mtimes(transpose(R_k), tau) - cross(Omega_k, mtimes(inertia, Omega_k))))

        constraints.add_general_constraints(p_body_k1 - p_body_next, np.zeros((3, 1)), np.zeros((3, 1)))
        constraints.add_general_constraints(dp_body_k1 - dp_body_next, np.zeros((3, 1)), np.zeros((3, 1)))
        constraints.add_general_constraints(Omega_k1 - Omega_next, np.zeros((3, 1)), np.zeros((3, 1)))
        constraints.add_general_constraints(DOmega_k1 - DOmega_next, np.zeros((3, 1)), np.zeros((3, 1)))

    # Initial States
    if k == 0:
        constraints.add_design_constraints(p_body_k, p_body0, p_body0, p_body0, 'p_body')
        constraints.add_design_constraints(dp_body_k, dp_body0, dp_body0, dp_body0, 'dp_body')
        constraints.add_design_constraints(Omega_k, Omega0, Omega0, Omega0, 'Omega')
        constraints.add_design_constraints(DOmega_k, DOmega0, DOmega0, DOmega0, 'DOmega')

    if k == cons.num_steps - 1:
        constraints.add_general_constraints(reshape(R_k, (9, 1)), np.reshape(Rf, (9, 1)), np.reshape(Rf, (9, 1)))
        constraints.add_design_constraints(p_body_k, p_bodyf, p_bodyf, p_bodyf, 'p_body')
        for leg in range(4):
            constraints.add_general_constraints(norm_2(mtimes(R_k, (p_feetf[:, leg] - p_body_k)) - p_feet_bar[:, leg]),
                                                [0], [r])

    # Objective Function
    e_R_k = inv_skew(approximate_log_a(mtimes(transpose(R_ref_k), R_k), 4))
    # J = J + (eOmega * mtimes(transpose(Omega_k), Omega_k)) + \
    #         (eF * mtimes(transpose(grf), grf)) + \
    #         (eR * mtimes(transpose(e_R_k), e_R_k))
    J = J + (eOmega * mtimes(transpose(Omega_k), Omega_k))
    J = J + (eF * mtimes(transpose(grf), grf))
    J = J + (eR * mtimes(transpose(e_R_k), e_R_k))

x = constraints.w
lbx = constraints.lbw
ubx = constraints.ubw
x0 = constraints.w0

# Initialize an NLP solver
nlp = {'x': x, 'f': J, 'g': constraints.g}

# Solver options
opts = {}
# opts["verbose"] = True
opts["ipopt"] = {"max_iter": 1000,
                 "fixed_variable_treatment": "make_constraint",
                 # # "nlp_scaling_method": "gradient-based",
                 # "nlp_scaling_max_gradient": 1,
                 # "nlp_scaling_min_value": 1e-16,
                 # # "bound_mult_init_method": "constant",
                 # "mu_strategy": "adaptive",
                 # # "mu_oracle": "quality-function",
                 # # "fixed_mu_oracle": "average_compl",
                 # "adaptive_mu_globalization": "kkt-error",
                 # "corrector_type": "affine",
                 # "max_soc": 0,
                 # "accept_every_trial_step": "yes",
                 # "linear_system_scaling": "none",
                 # # "neg_curv_test_tol": 0,
                 # "neg_curv_test_reg": "yes",
                 # "max_refinement_steps": 0,
                 # "min_refinement_steps": 0,
                 # # "linear_solver": "ma86",
                 # # "ma86_order": "auto",
                 # # "ma86_scaling": "mc64",
                 # # "ma86_small": 1e-10,
                 # # "ma86_static": 1,
                 # # "recalc_y": "yes",
                 "hessian_approximation": "limited-memory",
                 "mumps_mem_percent": 10000,
                 "print_level": 5}

# warm_start_init_point yes
# warm_start_bound_push 1e-9
# warm_start_bound_frac 1e-9
# warm_start_slack_bound_frac 1e-9
# warm_start_slack_bound_push 1e-9
# warm_start_mult_bound_push 1e-9

# Allocate a solver
solver = nlpsol("solver", "ipopt", nlp, opts)

# Solve the NLP
sol = solver(x0=x0, lbg=constraints.lbg, ubg=constraints.ubg, lbx=lbx, ubx=ubx)
sol_f = sol["f"]
sol_x = sol["x"]
sol_lam_x = sol["lam_x"]
sol_lam_g = sol["lam_g"]
T_opt = constraints.unpack_indices(sol_x, "T")
p_body_opt = constraints.unpack_indices(sol_x, "p_body")
dp_body_opt = constraints.unpack_indices(sol_x, "dp_body")
Omega_opt = constraints.unpack_indices(sol_x, "Omega")
DOmega_opt = constraints.unpack_indices(sol_x, "DOmega")
R_opt = integrate_omega_history(cons, R0, Omega_opt, T_opt)
# R_opt = constraints.unpack_indices(sol_x, "R")
F_0_opt = constraints.unpack_indices(sol_x, "F_0")
F_1_opt = constraints.unpack_indices(sol_x, "F_1")
F_2_opt = constraints.unpack_indices(sol_x, "F_2")
F_3_opt = constraints.unpack_indices(sol_x, "F_3")

T_opt = T_opt.reshape(T_opt.shape[0], -1)
p_body_opt = p_body_opt.reshape(p_body_opt.shape[0], -1)
dp_body_opt = dp_body_opt.reshape(dp_body_opt.shape[0], -1)
Omega_opt = Omega_opt.reshape(Omega_opt.shape[0], -1)
DOmega_opt = DOmega_opt.reshape(DOmega_opt.shape[0], -1)
R_opt = R_opt.reshape(cons.num_steps, 9)
F_0_opt = F_0_opt.reshape(F_0_opt.shape[0], -1)
F_1_opt = F_1_opt.reshape(F_1_opt.shape[0], -1)
F_2_opt = F_2_opt.reshape(F_2_opt.shape[0], -1)
F_3_opt = F_3_opt.reshape(F_3_opt.shape[0], -1)

# Print solution
print("-----")
print("objective at solution =", sol_f)
print("-----")
print("primal solution =", sol_x)
print("-----")
print("dual solution (x) =", sol_lam_x)
print("-----")
print("dual solution (g) =", sol_lam_g)
print("-----")
np.savetxt('opt/sol_f.csv', sol_f, delimiter=',')
np.savetxt('opt/sol_x', sol_x, delimiter=',')
np.savetxt('opt/sol_lam_x.csv', sol_lam_x, delimiter=',')
np.savetxt('opt/sol_lam_g.csv', sol_lam_g, delimiter=',')

np.savetxt('opt/T_opt.csv', np.transpose(np.array(T_opt)), delimiter=',')
np.savetxt('opt/p_body_opt.csv', p_body_opt, delimiter=',')
np.savetxt('opt/dp_body_opt.csv', dp_body_opt, delimiter=',')
np.savetxt('opt/Omega_opt.csv', Omega_opt, delimiter=',')
np.savetxt('opt/DOmega_opt.csv', DOmega_opt, delimiter=',')
np.savetxt('opt/R_opt.csv', R_opt, delimiter=',')
np.savetxt('opt/F_0_opt.csv', F_0_opt, delimiter=',')
np.savetxt('opt/F_1_opt.csv', F_1_opt, delimiter=',')
np.savetxt('opt/F_2_opt.csv', F_2_opt, delimiter=',')
np.savetxt('opt/F_3_opt.csv', F_3_opt, delimiter=',')

np.savetxt('metadata/step_list.csv', np.array(step_list), delimiter=',')
np.savetxt('metadata/contact_list.csv', np.array(contact_list), delimiter=',')
np.savetxt('metadata/p_feet0.csv', p_feet0, delimiter=',')
np.savetxt('metadata/p_feetf.csv', p_feetf, delimiter=',')
np.savetxt('metadata/p_feet_bar.csv', np.array(p_feet_bar), delimiter=',')
np.savetxt('metadata/r.csv', [r], delimiter=',')
