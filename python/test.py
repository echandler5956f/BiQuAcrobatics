from casadi import *
import numpy as np
from scipy.spatial.transform import Rotation as R


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
        # print('arg: ', arg)
        x_mat = arg[0]
        # print('x_mat: ', x_mat)
        r_err = mtimes(transpose(self.d), x_mat)
        # print('r_err: ', r_err)
        q = self.matrix_to_quat(r_err)
        # print('q: ', q)
        e_r = self.quat_to_axis_angle(q)
        # print('e_r: ', e_r)
        scalarized = mtimes(transpose(e_r), e_r)
        # jacobian of this quadratic form is 2xdx
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

def approximate_exp_a(a, deg):
    exp_a = DM(np.zeros((3, 3)))
    for i in range(deg):
        exp_a = exp_a + (mpower(a, i)/np.math.factorial(i))
    return exp_a


# x = MX.sym('x', 3, 3)
# f = LogMap('f', (R.from_euler('xyz', [0.0, 0.0, 0.0])).as_matrix())
# g = Function('g', [x], [f(x)])
#
# rot = (R.from_euler('xyz', [np.pi/4, 0.0, 0.0])).as_matrix()
# print(rot)
#
# result = g(rot)
# print(result.full())

# # Derivates OPTION 1: finite-differences
# f = LogMap('f', (R.from_euler('xyz', [0.0, 0.0, 0.0])).as_matrix(), {"enable_fd": True})
# J = Function('J', [x], [jacobian(f(x), x)])
# print(J((R.from_euler('xyz', [np.pi/2, 0.0, 0.0])).as_matrix()))

N = 2
dt = 1
u_weight = 1
r_weight = 500000
f = 0
x = []
u = []
g = []
lbg = []
ubg = []
lbx = []
ubx = []

r_ref = (R.from_euler('xyz', [-pi/4, 0.0, 0.0])).as_matrix()
h = LogMap('h', r_ref, {"enable_fd": True})

for k in range(N):
    x_k = MX.sym('x{}'.format(k), 3, 3)
    u_k = MX.sym('u{}'.format(k), 3, 1)
    x = vertcat(x, reshape(x_k, 9, 1))
    u = vertcat(u, reshape(u_k, 3, 1))
    f_tmp = Function('f_tmp{}'.format(k), [x_k], [h(x_k)])
    f = f + r_weight*f_tmp(x_k)
    if k < N - 1:
        f = f + u_weight*mtimes(transpose(u_k), u_k)
        x_next = mtimes(x_k, approximate_exp_a(skew(u_k*dt), 20))
        x_k1 = reshape(x[k*9:(k+1)*9], 3, 3)
        # print(reshape(x_k1-x_next, 9, 1))
        g = vertcat(g, reshape(x_k1-x_next, 9, 1))
        lbg = vertcat(lbg, DM(np.zeros((9, 1))))
        ubg = vertcat(ubg, DM(np.zeros((9, 1))))
    if k == 0:
        lbx = vertcat(lbx, vertcat(1, 0, 0, 0, 1, 0, 0, 0, 1))
        ubx = vertcat(ubx, vertcat(1, 0, 0, 0, 1, 0, 0, 0, 1))
    elif k == N - 1:
        lbx = vertcat(lbx, np.reshape(r_ref, 9))
        ubx = vertcat(ubx, np.reshape(r_ref, 9))
    else:
        lbx = vertcat(lbx, -1*np.ones(9, 1))
        ubx = vertcat(ubx, np.ones(9, 1))

# Initialize an NLP solver
nlp = {'x': vertcat(x, u), 'f': f, 'g': g}

# Solver options
opts = {}

# Allocate a solver
solver = nlpsol("solver", "ipopt", nlp, opts)

x0 = vertcat(repmat(vertcat(1, 0, 0, 0, 1, 0, 0, 0, 1), N), repmat(np.zeros((3, 1)), N))
lbx = vertcat(lbx, -pi*np.ones((3*N, 1)))
ubx = vertcat(ubx, pi*np.ones((3*N, 1)))

# Solve the NLP
sol = solver(x0=x0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

# Print solution
print("-----")
solx = sol["x"]
print("primal solution = ")
print(transpose(reshape(solx[(N-1)*9:N*9], (3, 3))))
print("-----")
print("reference = ")
print(DM(r_ref))
print("-----")
