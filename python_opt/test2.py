#
#     MIT No Attribution
#
#     Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl, KU Leuven.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy of this
#     software and associated documentation files (the "Software"), to deal in the Software
#     without restriction, including without limitation the rights to use, copy, modify,
#     merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#     permit persons to whom the Software is furnished to do so.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#     PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#     OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

chs_x0 = ca.SX.sym('chs_x0')
chs_x1 = ca.SX.sym('chs_x1')
chs_dx0 = ca.SX.sym('chs_dx0')
chs_dx1 = ca.SX.sym('chs_dx1')
chs_t = ca.SX.sym('chs_t')
chs_DT = ca.SX.sym('chs_DT')
chs_a0 = chs_x0
chs_a1 = chs_dx0
chs_a2 = - (3.0 * (chs_x0 - chs_x1) + chs_DT * (2.0 * chs_dx0 + chs_dx1)) / ca.power(chs_DT, 2)
chs_a3 = (2.0 * (chs_x0 - chs_x1) + chs_DT * (chs_dx0 + chs_dx1)) / ca.power(chs_DT, 3)
chs_x = chs_a0 + (chs_a1 * chs_t) + (chs_DT * ca.power(chs_t, 2)) + (chs_a3 * ca.power(chs_t, 3))
x_evaluate_at_t = ca.Function('x_evaluate_at_t', [chs_x0, chs_x1, chs_dx0, chs_dx1, chs_t, chs_DT], [chs_x],
                              ['x0', 'x1', 'dx0', 'dx1', 't', 'DT'], ['x'])

chs_dx = chs_a1 + (2.0 * chs_a2 * chs_t) + (3.0 * chs_a3 * ca.power(chs_t, 2))
dx_evaluate_at_t = ca.Function('dx_evaluate_at_t', [chs_x0, chs_x1, chs_dx0, chs_dx1, chs_t, chs_DT], [chs_x],
                               ['x0', 'x1', 'dx0', 'dx1', 't', 'DT'], ['x'])

# Time horizon
T = 10.

# Declare model variables
x1 = ca.SX.sym('x1')
x2 = ca.SX.sym('x2')
x = ca.vertcat(x1, x2)
u = ca.SX.sym('u')

# Model equations
xdot = ca.vertcat((1 - x2 ** 2) * x1 - x2 + u, x1)

# Objective term
L = x1 ** 2 + x2 ** 2 + u ** 2

# Continuous time dynamics
f = ca.Function('f', [x, u], [xdot, L], ['x', 'u'], ['xdot', 'L'])

# Control discretization
N = 200  # number of control intervals
h = T / N

# Start with an empty NLP
w = []
w0 = []
lbw = []
ubw = []
J = 0
g = []
lbg = []
ubg = []

# For plotting x and u given w
x_plot = []
u_plot = []

Xs = []
Us = []

for k in range(N):
    X_k = ca.MX.sym('X_' + str(k), 2)
    Xs = ca.horzcat(Xs, X_k)

    if k == 0:
        w.append(X_k)
        lbw.append([0, 1])
        ubw.append([0, 1])
        w0.append([0, 1])
    else:
        w.append(X_k)
        lbw.append([-0.25, -np.inf])
        ubw.append([np.inf, np.inf])
        w0.append([0, 0])
    x_plot.append(X_k)

    U_k = ca.MX.sym('U_' + str(k))
    Us = ca.horzcat(Us, U_k)
    w.append(U_k)
    lbw.append([-1])
    ubw.append([1])
    w0.append([0])
    u_plot.append(U_k)

# Formulate the NLP
for k in range(N - 1):

    x_k = Xs[:, k]
    u_k = Us[:, k]
    x_k_next = Xs[:, k+1]
    u_k_next = Us[:, k+1]
    x_kdot, q_k = f(x_k, u_k)
    x_kdot_next, q_knext = f(x_k_next, u_k_next)

    x_k12 = ca.MX.sym('X_' + str(k) + '_12', 2)
    w.append(x_k12)
    lbw.append([-0.25, -np.inf])
    ubw.append([np.inf, np.inf])
    w0.append([0, 0])

    g.append(x_k12 - (((x_k + x_k_next) / 2) + (h / 8) * (x_kdot - x_kdot_next)))
    lbg.append([0, 0])
    ubg.append([0, 0])

    u_k12 = ca.MX.sym('U_' + str(k) + '_12')
    w.append(u_k12)
    lbw.append([-1])
    ubw.append([1])
    w0.append([0])

    g.append(u_k12 - (u_k + (u_k + u_k_next) / 2))
    lbg.append([0])
    ubg.append([0])

    x_k12dot, q_k12 = f(x_k12, u_k12)
    g.append((x_k_next - x_k) - ((x_kdot + 4 * x_k12dot + x_kdot_next) * (h / 6)))
    lbg.append([0, 0])
    ubg.append([0, 0])

    # J = J + (h / 6) * ((ca.power(x_k[0], 2) + 4 * ca.power(x_k12[0], 2) + ca.power(x_k_next[0], 2)) +
    #                    (ca.power(x_k[1], 2) + 4 * ca.power(x_k12[1], 2) + ca.power(x_k_next[1], 2)))
    # J = J + (ca.power(u_k, 2) + ca.power(u_k_next, 2)) / 2
    J = J + q_k

# Concatenate vectors
w = ca.vertcat(*w)
g = ca.vertcat(*g)
x_plot = ca.horzcat(*x_plot)
u_plot = ca.horzcat(*u_plot)
w0 = np.concatenate(w0)
lbw = np.concatenate(lbw)
ubw = np.concatenate(ubw)
lbg = np.concatenate(lbg)
ubg = np.concatenate(ubg)

# print(w)
# print(w.size())
# print(x_plot.size())
# print(u_plot.size())

# Create an NLP solver
prob = {'f': J, 'x': w, 'g': g}
solver = ca.nlpsol('solver', 'ipopt', prob)

# Function to get x and u trajectories from w
trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

trajectoriestest = ca.Function('trajectories2', [w], [x_plot], ['w'], ['x'])

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

# print(x_plot)
# print(trajectoriestest(sol['x']))
# print('blank')
# print(trajectories(sol['x']))
# print(sol['x'])

# print(sol['x'].size())
x_opt, u_opt = trajectories(sol['x'])
# print(x_opt.size())
# print(u_opt.size())
x_opt = x_opt.full()  # to numpy array
u_opt = u_opt.full()  # to numpy array

# Plot the result
tgrid = np.linspace(0, T, N)
plt.figure(1)
plt.clf()
plt.plot(tgrid, x_opt[0], 'o')
plt.plot(tgrid, x_opt[1], 'o')
plt.step(tgrid, u_opt[0], 'o')
plt.xlabel('t')
plt.legend(['x1', 'x2', 'u'])
plt.grid()
plt.show()
