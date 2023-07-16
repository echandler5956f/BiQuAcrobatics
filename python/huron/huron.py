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
        dt = T_opt[0, 0, i] / cons.step_list[i]
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

# Period
period = 1

# Steps per contact phase
step_list = [30, 30, 30]

# Contact pattern for each step
contact_list = [[1, 1], [1, 0], [0, 1]]

# Specify the contact metadata
cons = Contacts(step_list, contact_list)

# Time periods
T = np.divide(np.ones((cons.num_cons, 1)) * period, cons.cum_steps)

# Initial States
p_body0 = np.array([0, 0, 0.55516])
dp_body0 = np.zeros((3, 1))
R0 = rp.from_euler('xyz', [0, 0, 0], True).as_matrix()
tmp1 = [0.0125, 0.0755, -0.55516]
p_feet0 = np.array([p_body0 + np.matmul(R0, leg_mask(tmp1, 1)),
                    p_body0 + np.matmul(R0, leg_mask(tmp1, 2))]).transpose()
Theta0 = np.zeros((3, 1))
Omega0 = np.zeros((3, 1))

# Final States
p_bodyf = np.array([0.4, 0, 0.55516])
Rf = rp.from_euler('xyz', [0, 0, 0], True).as_matrix()
tmp2 = [0.0125, 0.0775, -0.55516]
p_feetf = np.array([p_bodyf + np.matmul(Rf, leg_mask(tmp2, 1)),
                    p_bodyf + np.matmul(Rf, leg_mask(tmp2, 2))]).transpose()

# Friction coefficient
mu = 0.7

# GRF limits
f_bounds = np.array([[-300, 300],
                     [-300, 300],
                     [0, 600]])

# Acceleration due to gravity
g_accel = np.array([[0], [0], [9.81]])

# COM bounding constraint. Ideally you would set this to some section of a
# tube each timestep within you want the trajectory to lie
p_body_bounds = np.array([[-5, 5],
                          [-5, 5],
                          [0, 5]])

# Velocity bounds to make the problem more solvable
dp_body_bounds = np.array([[-2.5, 2.5],
                           [-2.5, 2.5],
                           [-2.5, 2.5]])

# Angular velocity bounds to make the problem more solvable
Theta_bounds = np.array([[-2*pi, 2*pi],
                         [-2*pi, 2*pi],
                         [-2*pi, 2*pi]])

# Time derivative angular velocity bounds to make the problem more solvable
DOmega_bounds = np.array([[-10, 10],
                          [-10, 10],
                          [-10, 10]])

# Mass of the SRB
mass = 37

# Inertia of SRB
inertia = np.array([[3.09249, 0, 0],
                    [0, 5.106100, 0],
                    [0, 0, 6.939757]])
inv_inertia = np.linalg.inv(inertia)

# Decision Variables

# Set up constraint structure
fields = ['p_body', 'dp_body', 'Theta', 'Omega', 'Fl', 'Fr']
sizes = [(3, 1), (3, 1), (3, 1), (3, 1), (3, 1), (3, 1)]
split_flags = [False, False, False, False, False, False]
constraints = CTConstraints(fields, sizes, split_flags)

# COM of the body (3x1)
p_body = []

# Time derivative of the COM of the body (3x1)
dp_body = []

# Euler angles of SRB w.r.t world frame (3x1)
Theta = []

# Angular velocity of SRB in world frame (3x1)
Omega = []

# GRF on each foot (3x1)
Fl = []
Fr = []

for k in range(cons.num_steps):
    i = cons.get_current_phase(k)
    dt = T[i] / cons.step_list[i]

    p_body = vertcat(p_body, MX.sym('p_body_k{}'.format(k), 3, 1))
    dp_body = vertcat(dp_body, MX.sym('dp_body_k{}'.format(k), 3, 1))
    Theta = vertcat(Theta, MX.sym('Theta_k{}'.format(k), 3, 1))
    Omega = vertcat(Omega, MX.sym('Omega_k{}'.format(k), 3, 1))
    if cons.contact_list[cons.get_current_phase(k)][0]:
        Fl = vertcat(Fl, MX.sym('Fl_k{}'.format(k), 3, 1))
    if cons.contact_list[cons.get_current_phase(k)][1]:
        Fr = vertcat(Fr, MX.sym('Fr_k{}'.format(k), 3, 1))

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
    Theta_k = Omega[3 * k: 3 * (k + 1)]

    # Time derivative of angular velocity of SRB w.r.t body frame (3x1)
    Omega_k = Omega[3 * k: 3 * (k + 1)]

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
        # constraints.add_design_constraints(F_0_k, f_bounds[:, 0], f_bounds[:, 1], rand_in_bounds(f_bounds), 'F_0')
        constraints.add_design_constraints(F_0_k, f_bounds[:, 0], f_bounds[:, 1], [0, 0, g_accel[2] * mass / clegs],
                                           'F_0')
    if cons.contact_list[i][1]:
        F_1_k = F_1[3 * k: 3 * (k + 1)]
        grf = grf + F_1_k
        tau = tau + cross(F_1_k, (p_body_k - p_feet0[:, 1]))
        constraints.add_general_constraints(fabs(F_1_k[0] / F_1_k[2]), [0], [mu])
        constraints.add_general_constraints(fabs(F_1_k[1] / F_1_k[2]), [0], [mu])
        # constraints.add_design_constraints(F_1_k, f_bounds[:, 0], f_bounds[:, 1], rand_in_bounds(f_bounds), 'F_1')
        constraints.add_design_constraints(F_1_k, f_bounds[:, 0], f_bounds[:, 1], [0, 0, g_accel[2] * mass / clegs],
                                           'F_1')

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
solution = constraints.unpack_all(sol_x, True)
p_body_opt = solution["p_body"]["opt_x"]
dp_body_opt = solution["dp_body"]["opt_x"]
Theta_opt = solution["Theta"]["opt_x"]
Omega_opt = solution["Omega"]["opt_x"]
Fl_opt = solution["Fr"]["opt_x"]
Fr_opt = solution["Fl"]["opt_x"]

p_body_opt = p_body_opt.reshape(p_body_opt.shape[0], -1)
dp_body_opt = dp_body_opt.reshape(dp_body_opt.shape[0], -1)
Theta_opt = Theta_opt.reshape(Theta_opt.shape[0], -1)
Omega_opt = Omega_opt.reshape(Omega_opt.shape[0], -1)
Fl_opt = Fr_opt.reshape(Fl_opt.shape[0], -1)
Fr_opt = Fl_opt.reshape(Fr_opt.shape[0], -1)

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
np.savetxt('huron/opt/sol_f.csv', sol_f, delimiter=',')
np.savetxt('huron/opt/sol_x', sol_x, delimiter=',')
np.savetxt('huron/opt/sol_lam_x.csv', sol_lam_x, delimiter=',')
np.savetxt('huron/opt/sol_lam_g.csv', sol_lam_g, delimiter=',')

np.savetxt('huron/opt/p_body_opt.csv', p_body_opt, delimiter=',')
np.savetxt('huron/opt/dp_body_opt.csv', dp_body_opt, delimiter=',')
np.savetxt('huron/opt/Theta_opt.csv', Theta_opt, delimiter=',')
np.savetxt('huron/opt/Omega_opt.csv', Omega_opt, delimiter=',')
np.savetxt('huron/opt/Fl_opt.csv', Fl_opt, delimiter=',')
np.savetxt('huron/opt/Fr_opt.csv', Fr_opt, delimiter=',')

np.savetxt('huron/metadata/step_list.csv', np.array(step_list), delimiter=',')
np.savetxt('huron/metadata/contact_list.csv', np.array(contact_list), delimiter=',')
np.savetxt('huron/metadata/p_feet0.csv', p_feet0, delimiter=',')
np.savetxt('huron/metadata/p_feetf.csv', p_feetf, delimiter=',')
