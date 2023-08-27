# #
# #     MIT No Attribution
# #
# #     Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl, KU Leuven.
# #
# #     Permission is hereby granted, free of charge, to any person obtaining a copy of this
# #     software and associated documentation files (the "Software"), to deal in the Software
# #     without restriction, including without limitation the rights to use, copy, modify,
# #     merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# #     permit persons to whom the Software is furnished to do so.
# #
# #     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# #     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# #     PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# #     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# #     OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# #     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# #
# import casadi as ca
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Degree of interpolating polynomial
# d = 3
#
# # Get collocation points
# tau_root = np.append(0, ca.collocation_points(d, 'legendre'))
#
# print(tau_root)
#
# # Coefficients of the collocation equation
# C = np.zeros((d+1,d+1))
#
# # Coefficients of the continuity equation
# D = np.zeros(d+1)
#
# # Coefficients of the quadrature function
# B = np.zeros(d+1)
#
# # Construct polynomial basis
# for j in range(d+1):
#     # Construct Lagrange polynomials to get the polynomial basis at the collocation point
#     p = np.poly1d([1])
#     for r in range(d+1):
#         if r != j:
#             p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])
#
#     # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
#     D[j] = p(1.0)
#
#     # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
#     pder = np.polyder(p)
#     for r in range(d+1):
#         C[j,r] = pder(tau_root[r])
#
#     # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
#     pint = np.polyint(p)
#     B[j] = pint(1.0)
#
# print(B)
# print('')
# print(C)
# print('')
# print(D)
#
# # Time horizon
# T = 10.
#
# # Declare model variables
# x1 = ca.SX.sym('x1')
# x2 = ca.SX.sym('x2')
# x = ca.vertcat(x1, x2)
# u = ca.SX.sym('u')
#
# # Model equations
# xdot = ca.vertcat((1-x2**2)*x1 - x2 + u, x1)
#
# # Objective term
# L = x1**2 + x2**2 + u**2
#
# # Continuous time dynamics
# f = ca.Function('f', [x, u], [xdot, L], ['x', 'u'], ['xdot', 'L'])
#
# # Control discretization
# N = 10 # number of control intervals
# h = T/N
#
# # Start with an empty NLP
# w=[]
# w0 = []
# lbw = []
# ubw = []
# J = 0
# g=[]
# lbg = []
# ubg = []
#
# # For plotting x and u given w
# x_plot = []
# u_plot = []
#
# # "Lift" initial conditions
# Xk = ca.MX.sym('X0', 2)
# w.append(Xk)
# lbw.append([0, 1])
# ubw.append([0, 1])
# w0.append([0, 1])
# x_plot.append(Xk)
#
# # Formulate the NLP
# for k in range(N):
#     # New NLP variable for the control
#     Uk = ca.MX.sym('U_' + str(k))
#     w.append(Uk)
#     lbw.append([-1])
#     ubw.append([1])
#     w0.append([0])
#     u_plot.append(Uk)
#
#     # State at collocation points
#     Xc = []
#     for j in range(d):
#         Xkj = ca.MX.sym('X_'+str(k)+'_'+str(j), 2)
#         Xc.append(Xkj)
#         w.append(Xkj)
#         lbw.append([-0.25, -np.inf])
#         ubw.append([np.inf,  np.inf])
#         w0.append([0, 0])
#
#     # Loop over collocation points
#     Xk_end = D[0]*Xk
#     for j in range(1,d+1):
#        # Expression for the state derivative at the collocation point
#        xp = C[0,j]*Xk
#        for r in range(d): xp = xp + C[r+1,j]*Xc[r]
#
#        # Append collocation equations
#        fj, qj = f(Xc[j-1],Uk)
#        g.append(h*fj - xp)
#        lbg.append([0, 0])
#        ubg.append([0, 0])
#
#        # Add contribution to the end state
#        Xk_end = Xk_end + D[j]*Xc[j-1]
#
#     # Add contribution to quadrature function
#     fk, qk = f(Xk, Uk)
#     J = J + qk
#
#     # New NLP variable for state at end of interval
#     Xk = ca.MX.sym('X_' + str(k+1), 2)
#     w.append(Xk)
#     lbw.append([-0.25, -np.inf])
#     ubw.append([np.inf,  np.inf])
#     w0.append([0, 0])
#     x_plot.append(Xk)
#
#     # Add equality constraint
#     g.append(Xk_end-Xk)
#     lbg.append([0, 0])
#     ubg.append([0, 0])
#
# # Concatenate vectors
# w = ca.vertcat(*w)
# g = ca.vertcat(*g)
# x_plot = ca.horzcat(*x_plot)
# u_plot = ca.horzcat(*u_plot)
# w0 = np.concatenate(w0)
# lbw = np.concatenate(lbw)
# ubw = np.concatenate(ubw)
# lbg = np.concatenate(lbg)
# ubg = np.concatenate(ubg)
#
# # print(w)
# # print(w.size())
# # print(x_plot.size())
# # print(u_plot.size())
#
# # Create an NLP solver
# prob = {'f': J, 'x': w, 'g': g}
# solver = ca.nlpsol('solver', 'ipopt', prob)
#
# # Function to get x and u trajectories from w
# trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])
#
# trajectoriestest = ca.Function('trajectories2', [w], [x_plot], ['w'], ['x'])
#
# # Solve the NLP
# sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
#
# # print(x_plot)
# # print(trajectoriestest(sol['x']))
# # print('blank')
# # print(trajectories(sol['x']))
# # print(sol['x'])
#
# # print(sol['x'].size())
# x_opt, u_opt = trajectories(sol['x'])
# # print(x_opt.size())
# # print(u_opt.size())
# x_opt = x_opt.full() # to numpy array
# u_opt = u_opt.full() # to numpy array
#
# # Plot the result
# tgrid = np.linspace(0, T, N+1)
# plt.figure(1)
# plt.clf()
# plt.plot(tgrid, x_opt[0], 'o')
# plt.plot(tgrid, x_opt[1], 'o')
# plt.step(tgrid, np.append(np.nan, u_opt[0]), 'o')
# plt.xlabel('t')
# plt.legend(['x1','x2','u'])
# plt.grid()
# plt.show()






















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

# Degree of interpolating polynomial
d = 3

# Get collocation points
tau_root = np.append(0, ca.collocation_points(d, 'radau'))

# Coefficients of the collocation equation
C = np.zeros((d+1,d+1))

# Coefficients of the continuity equation
D = np.zeros(d+1)

# Coefficients of the quadrature function
B = np.zeros(d+1)

# Construct polynomial basis
for j in range(d+1):
    # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    p = np.poly1d([1])
    for r in range(d+1):
        if r != j:
            p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

    # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    D[j] = p(1.0)

    # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    pder = np.polyder(p)
    for r in range(d+1):
        C[j,r] = pder(tau_root[r])

    # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
    pint = np.polyint(p)
    B[j] = pint(1.0)

# Time horizon
T = 10.

# Declare model variables
x1 = ca.SX.sym('x1')
x2 = ca.SX.sym('x2')
x = ca.vertcat(x1, x2)
u = ca.SX.sym('u')

# Model equations
xdot = ca.vertcat((1-x2**2)*x1 - x2 + u, x1)

# Objective term
L = x1**2 + x2**2 + u**2

# Continuous time dynamics
f = ca.Function('f', [x, u], [xdot, L], ['x', 'u'], ['xdot', 'L'])

# Control discretization
N = 20 # number of control intervals
h = T/N

# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []

# For plotting x and u given w
x_plot = []
u_plot = []

# "Lift" initial conditions
Xk = ca.MX.sym('X0', 2)
w.append(Xk)
lbw.append([0, 1])
ubw.append([0, 1])
w0.append([0, 1])
x_plot.append(Xk)

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = ca.MX.sym('U_' + str(k))
    w.append(Uk)
    lbw.append([-0.75])
    ubw.append([1])
    w0.append([0])
    u_plot.append(Uk)

    # State at collocation points
    Xc = []
    for j in range(d):
        Xkj = ca.MX.sym('X_'+str(k)+'_'+str(j), 2)
        Xc.append(Xkj)
        w.append(Xkj)
        lbw.append([-np.inf, -np.inf])
        ubw.append([np.inf,  np.inf])
        w0.append([0, 0])

    # Loop over collocation points
    Xk_end = D[0]*Xk
    for j in range(1,d+1):
       # Expression for the state derivative at the collocation point
       xp = C[0,j]*Xk
       for r in range(d): xp = xp + C[r+1,j]*Xc[r]

       # Append collocation equations
       fj, qj = f(Xc[j-1],Uk)
       g.append(h*fj - xp)
       lbg.append([0, 0])
       ubg.append([0, 0])

       # Add contribution to the end state
       Xk_end = Xk_end + D[j]*Xc[j-1];

       # Add contribution to quadrature function
       J = J + B[j]*qj*h

    # New NLP variable for state at end of interval
    Xk = ca.MX.sym('X_' + str(k+1), 2)
    w.append(Xk)
    lbw.append([-np.inf, -np.inf])
    ubw.append([np.inf,  np.inf])
    w0.append([0, 0])
    x_plot.append(Xk)

    # Add equality constraint
    g.append(Xk_end-Xk)
    lbg.append([0, 0])
    ubg.append([0, 0])

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

# Create an NLP solver
prob = {'f': J, 'x': w, 'g': g}
solver = ca.nlpsol('solver', 'ipopt', prob);

# Function to get x and u trajectories from w
trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
x_opt, u_opt = trajectories(sol['x'])
x_opt = x_opt.full() # to numpy array
u_opt = u_opt.full() # to numpy array

# Plot the result
tgrid = np.linspace(0, T, N+1)
plt.figure(1)
plt.clf()
plt.plot(tgrid, x_opt[0], '--')
plt.plot(tgrid, x_opt[1], '-')
plt.step(tgrid, np.append(np.nan, u_opt[0]), '-.')
plt.xlabel('t')
plt.legend(['x1','x2','u'])
plt.grid()
plt.show()


# Number of nonzeros in equality constraint Jacobian...:      812
# Number of nonzeros in inequality constraint Jacobian.:        0
# Number of nonzeros in Lagrangian Hessian.............:      200
#
# Total number of variables............................:      180
#                      variables with only lower bounds:        0
#                 variables with lower and upper bounds:       20
#                      variables with only upper bounds:        0
# Total number of equality constraints.................:      160
# Total number of inequality constraints...............:        0
#         inequality constraints with only lower bounds:        0
#    inequality constraints with lower and upper bounds:        0
#         inequality constraints with only upper bounds:        0


# Number of Iterations....: 9
#
#                                    (scaled)                 (unscaled)
# Objective...............:   2.9329598203471381e+00    2.9329598203471381e+00
# Dual infeasibility......:   2.8148727793109174e-10    2.8148727793109174e-10
# Constraint violation....:   5.7509719209036803e-11    5.7509719209036803e-11
# Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
# Complementarity.........:   7.4385647634491964e-09    7.4385647634491964e-09
# Overall NLP error.......:   7.4385647634491964e-09    7.4385647634491964e-09
#
#
# Number of objective function evaluations             = 10
# Number of objective gradient evaluations             = 10
# Number of equality constraint evaluations            = 10
# Number of inequality constraint evaluations          = 0
# Number of equality constraint Jacobian evaluations   = 10
# Number of inequality constraint Jacobian evaluations = 0
# Number of Lagrangian Hessian evaluations             = 9
# Total seconds in IPOPT                               = 0.016
#
# EXIT: Optimal Solution Found.
#       solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
#        nlp_f  |        0 (       0) 167.00us ( 16.70us)        10
#        nlp_g  |        0 (       0) 409.00us ( 40.90us)        10
#   nlp_grad_f  |        0 (       0) 495.00us ( 41.25us)        12
#   nlp_hess_l  |   1.00ms (111.11us) 659.00us ( 73.22us)         9
#    nlp_jac_g  |   2.00ms (181.82us)   1.50ms (136.09us)        11
#        total  |  17.00ms ( 17.00ms)  16.45ms ( 16.45ms)         1