clear all; clc; close all

%% Initialization

import casadi.*

%% Constant Parameters

% Omega cost weight
eOmega = 1;

% Force cost weight
eF = 0.25;

% Rotation error cost weight
eR = 0.75;

% Minimum total time
tMin = 0.25;

% Maximum total time
tMax = 2.0;

% Number of contact phases
n_p = 2;

% Predefined number of steps for the ith contact phase
N = ones(n_p,1).*3;

% Helper for checking what contact we are in
Nch = cumsum(N);

% Total number of steps
Nc = sum(N);

% Friction coefficient
mu = 0.7;

% GRF limits
f_bounds = [-50, 50;
            -50, 50;
            -100, 100];

% Acceleration due to gravity
g_accel = [0;0;-9.81];

% COM bounding constraint. Ideally you would set this to some section of a 
% tube each timestep within you want the trajectory to lie
p_body_bounds = [0, 0.5;
                 0, 0.5;
                 0.125, 1.0];

% Velocity bounds to make the problem more solvable
dp_body_bounds = [-5.0, 5.0;
                  -5.0, 5.0;
                  -5.0, 5.0];

% Angular velocity bounds to make the problem more solvable
Omega_bounds = [-2*pi, 2*pi;
                -2*pi, 2*pi;
                -2*pi, 2*pi];

% The sth foot position is constrained in a sphere of radius r to satisfy 
% joint limits. This parameter is the center of the sphere w.r.t the COM
pbar = [0.23;0.19;-0.3];
p_feet_bar = [legMask(pbar,1),legMask(pbar,2),legMask(pbar,3),legMask(pbar,4)];

% Sphere of radius r that the foot position is constrained to
r = ones(3,1).*0.25;

% Mass of the SRB
mass = 2.50000279;

% Inertia of SRB
inertia = [3.09249e-2, -9.00101e-7, 1.865287e-5;
         -8.00101e-7, 5.106100e-2, 1.245813e-4;
         1.865287e-5, 1.245813e-4, 6.939757e-2];
invinertia = pinv(inertia);

% Kinematics
syms q [3, 1]
bodyHalfLength = 0.194;
bodyHalfWidth = 0.0875;
bodyHalfHeight = 0.025;
abadLinkLength = 0.01295;
hipLinkLength = 0.160;
kneeLinkY_offset = 0.04745;
kneeLinkLength = 0.1675;
bounds = [deg2rad(-50), deg2rad(20);
          deg2rad(-60), deg2rad(20);
          deg2rad(-25), deg2rad(25)];
PM = utils.tdh(pi/2, 0, bodyHalfWidth, -pi/2) * ...
     utils.tdh(0, -bodyHalfLength, 0, 0);
omegaList = [[0;0;1], [1;0;0], [1;0;0]];
pList = [[0;0;0], [abadLinkLength; 0; 0], ...
         [kneeLinkY_offset; hipLinkLength; 0]];
R_home = [0,1,0;
          0,0,-1;
          -1,0,0];
t_home = [abadLinkLength + kneeLinkY_offset; ...
          hipLinkLength + kneeLinkLength;0];
M = [R_home, t_home;
     0, 0, 0, 1];
kin = kinematics.KinematicsPOE(PM,M,omegaList,pList,q,bounds,6,3);

%% Decision Variables

% COM of the body (3x1)
p_body = {};

% Time derivative of the COM of the body (3x1)
dp_body = {};

% Foot position in the world frame of each foot (3x4)
p_feet = {};

% Angular velocity of SRB w.r.t body frame (3x1)
Omega = {};

% Rotation matrix of the body frame (3x3)
R = {};

% GRF on each foot (3x4)
F = {};

for k = 1 : Nc
    p_body = {p_body{:}, SX.sym(['p_body_k' num2str(k)],3,1)};
    dp_body = {dp_body{:}, SX.sym(['dp_body_k' num2str(k)],3,1)};
    p_feet = {p_feet{:}, SX.sym(['p_feet_k' num2str(k)],3,4)};
    Omega = {Omega{:}, SX.sym(['Omega_k' num2str(k)],3,1)}; 
    R = {R{:}, SX.sym(['R_k' num2str(k)],3,3)};
    F = {F{:}, SX.sym(['F_k' num2str(k)],3,4)};
end

% Optimal contact timing for the ith contact phase (n_px1)
T = SX.sym('T',n_p,1);

%% Initial States

p_body0 = [0;0;0.3];
dp_body0 = zeros(3,1);
tmp = kin.fk([deg2rad(-5);deg2rad(5);deg2rad(5)]);
p_feet0 = [legMask(tmp,1),legMask(tmp,2),legMask(tmp,3),legMask(tmp,4)];
Omega0 = zeros(3,1);
R0 = eye(3);

%% Final States

p_bodyf = [bodyHalfLength*2;bodyHalfWidth*2;0.25];
tmp = kin.fk([deg2rad(-3);deg2rad(2);deg2rad(7)]);
p_feetf = [legMask(tmp,1),legMask(tmp,2),legMask(tmp,3),legMask(tmp,4)];
theta = 15;
Rf = [cos(deg2rad(theta)), -sin(deg2rad(theta)), 0;
      sin(deg2rad(theta)), cos(deg2rad(theta)), 0;
      0,  0,  1];

%% Reference Trajectory

R_ref = zeros(3,3, Nc);
R0SO3 = SO3(R0);
RfSO3 = SO3(Rf);
for i = 1 : Nc
    R0SO3.interp(RfSO3, i/Nc);
    R_ref(:,:,i) = R0SO3.interp(RfSO3, i/Nc);
end

%% Contact Timing Optimization

% Start with an empty NLP
global w lbw ubw w0 g lbg ubg
w = {};
w0 = {};
lbw = {};
ubw = {};
J = 0;
g = {};
lbg = {};
ubg = {};

% Total time must be within our bounds
addGeneralConstraints(T.sum(), tMin, tMax);

% All contact timings must be positive
addDesignConstraintsAndInit(T, zeros(n_p,1), ones(n_p,1).*(tMax), ...
    ones(n_p,1).*((tMax-tMin)/Nc));

for k = 1 : Nc
    %% Gather Decision Variables

    i = getCurrentPhase(k, Nch);
    dt = T(i)/N(i);

    % COM of the body (3x1)
    p_body_k = p_body{1,k};
    
    % Time derivative of the COM of the body (3x1)
    dp_body_k = dp_body{1,k};
    
    % Foot position in the world frame of each foot (3x4)
    p_feet_k = p_feet{1,k};
    
    % Angular velocity of SRB w.r.t body frame (3x1)
    Omega_k = Omega{1,k};
    
    % Rotation matrix of the body frame (3x3)
    R_k = R{1,k};
    
    % GRF on each foot (3x4)
    F_k = F{1,k};

    %% Add Constraints

    % Add dummy velocity constraints
    addDesignConstraintsAndInit(dp_body_k, dp_body_bounds(:,1), ...
            dp_body_bounds(:,2), randInBounds(dp_body_bounds));
    
    if (k ~= 1) && (k ~= Nc)
        % Add body bounding box constraints
        addDesignConstraintsAndInit(p_body_k, p_body_bounds(:,1), ...
            p_body_bounds(:,2), p_body0);

        for leg = 1 : 4
            addDesignConstraintsAndInit( p_feet_k(:,leg), ...
                ones(3,1).*(-1), ones(3,1).*(1), p_feet_bar(:,leg));
        end

        % Add dummy rotation constraints
        addDesignConstraintsAndInit(R_k.reshape(9,1), ...
            ones(9,1).*(-5), ones(9,1).*(5), ...
            reshape(R_ref(:,:,k),9,1));
    end
    addDesignConstraintsAndInit(Omega_k, Omega_bounds(:,1), ...
            Omega_bounds(:,2), randInBounds(Omega_bounds));

    % Add friction cone, GRF, and foot position constraints to each leg
    GRF = zeros(3,1);
        for leg = 1 : 4
            if i == 1
                GRF = GRF + F_k(:,leg);
                addGeneralConstraints(abs(F_k(1,leg)/F_k(3,leg)), 0, mu);
                addGeneralConstraints(abs(F_k(2,leg)/F_k(3,leg)), 0, mu);
            end
            addGeneralConstraints(abs(R_k*(p_feet_k(:,leg) - p_body_k) - ...
                p_feet_bar(:,leg)), zeros(3,1), r);
            addDesignConstraintsAndInit(F_k(:,leg), f_bounds(:,1), ...
                f_bounds(:,2), [0;0;mass/4]);
        end

    % Discrete dynamics
    if k < Nc
        p_body_k1 = p_body{1,k+1};
        dp_body_k1 = dp_body{1,k+1};
        Omega_k1 = Omega{1,k+1};
        R_k1 = R{1,k+1};
        p_body_next = p_body_k + dp_body_k.*dt;
        dp_body_next = dp_body_k + ((GRF+g_accel)./mass).*dt;
        accum1 = zeros(3,1);
        for leg = 1 : 4
            accum1 = accum1 + cross((p_feet_k(:,leg)-p_body_k), ...
                F_k(:,leg));
        end
        OmegaSO3 = Omega_k.skew();
        Omega_next = Omega_k + invinertia*((transpose(R_k)*accum1) - ...
            (OmegaSO3*inertia*Omega_k)).*dt;
        R_next = R_k*approximateExpA(OmegaSO3.*dt,4);
        addGeneralConstraints(p_body_k1-p_body_next, zeros(3,1), ...
            zeros(3,1));
        addGeneralConstraints(dp_body_k1-dp_body_next, zeros(3,1), ...
            zeros(3,1));
        addGeneralConstraints(Omega_k1-Omega_next, zeros(3,1), zeros(3,1));
        addGeneralConstraints(reshape(R_k1-R_next,9,1), zeros(9,1), ...
            zeros(9,1));
    end

    % Initial States
    if k == 1
        addDesignConstraintsAndInit(p_body_k, p_body0, p_body0, p_body0);
        for leg = 1 : 4
            addDesignConstraintsAndInit(p_feet_k(:,leg), ...
                p_feet0(:,leg), p_feet0(:,leg), p_feet0(:,leg));
        end
        addDesignConstraintsAndInit(R_k.reshape(9,1), ...
            reshape(R0,9,1), reshape(R0,9,1), reshape(R0,9,1));
    end
    
    % Final States
    if k == Nc
        addDesignConstraintsAndInit(p_body_k, p_bodyf, p_bodyf, p_bodyf);
        for leg = 1 : 4
            addDesignConstraintsAndInit(p_feet_k(:,leg), ...
                p_feetf(:,leg), p_feetf(:,leg), p_feetf(:,leg));
        end
        addDesignConstraintsAndInit(R_k.reshape(9,1), reshape(Rf,9,1), ...
            reshape(Rf,9,1), reshape(Rf,9,1));
    end

    %% Objective Function
    
    % Calculate rotation matrix error term
    e_R_k = vex(abs(R_ref(:,:,k) - R_k));
    % e_R_k = logMap(transpose(R_ref(:,:,k)*R_k));
    J = J + (eOmega.*transpose(Omega_k)*Omega_k) + ...
        (eF.*transpose(GRF)*GRF) + (eR.*transpose(e_R_k)*e_R_k);
end

%% Solve Problem

% Create an NLP solver
% disp(J);
% disp(vertcat(w{:}));
% disp(vertcat(g{:}));
% A = vertcat(w{:});
% B = vertcat(g{:});
options = struct('expand', true, 'ipopt', struct('max_iter', 100000, ...
    'fixed_variable_treatment', 'make_constraint', 'print_level', 6));
problem = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}));
solver = nlpsol('solver', 'ipopt', problem, options);

% Solve the NLP
% disp(vertcat(w0{:}));
% disp(vertcat(lbw{:}));
% disp(vertcat(ubw{:}));
% disp(vertcat(lbg{:}));
% disp(vertcat(ubg{:}));
% C = vertcat(w0{:});
% D = vertcat(lbw{:});
% E = vertcat(ubw{:});
% F = vertcat(lbg{:});
% G = vertcat(ubg{:});
sol = solver('x0', vertcat(w0{:}), 'lbx', vertcat(lbw{:}), 'ubx', ...
    vertcat(ubw{:}), 'lbg', vertcat(lbg{:}), 'ubg', vertcat(ubg{:}));
w_opt = full(sol.x);

function addGeneralConstraints(g_k, lbg_k, ubg_k)
    global g lbg ubg
    if size(g_k,2) > 1 || size(lbg_k,2) > 1 || size(ubg_k,2) > 1
        disp("Invalid size of constraints. The constraints are one " + ...
            "column vector.");
    else
        if all(size(g_k) == size(lbg_k)) && all(size(g_k) == ...
                size(ubg_k)) && all(size(lbg_k) == size(ubg_k))
            g = {g{:}, g_k};
            lbg = {lbg{:}, lbg_k};
            ubg = {ubg{:}, ubg_k};
        else
            disp("Invalid size of constraints. The number of upper " + ...
                "and lower bound constraints should match the size of g.");
        end
    end
end

function addDesignConstraintsAndInit(w_k, lbw_k, ubw_k, w0_k)
    global w lbw ubw w0
    if size(w_k,2) > 1 || size(lbw_k,2) > 1 || size(ubw_k,2) > 1 || ...
            size(w0_k,2) > 1
        disp("Invalid size of constraints. The constraints are one " + ...
            "column vector.");
    else
        if all(size(w_k) == size(lbw_k)) && all(size(w_k) == ...
                size(ubw_k)) && all(size(lbw_k) == size(ubw_k)) && ...
                all(size(w_k) == size(w0_k))
            w = {w{:}, w_k};
            lbw = {lbw{:}, lbw_k};
            ubw = {ubw{:}, ubw_k};
            w0 = {w0{:}, w0_k};
        else
            disp("Invalid size of constraints. The number of upper " + ...
                "and lower bound constraints should match the size of w.");
        end
    end
end

function expA = approximateExpA(A, deg)
    expA = A;
    for i = 0 : deg
        expA = expA + (A.^i)/(factorial(i));
    end
end

% n x 2 matrix of lower[1] and upper[2] bounds
function r = randInBounds(bounds)
    n = length(bounds);
    r = zeros(n,1);
    for i=1:n
        r(i,1) = (bounds(2)-bounds(1))*rand(1) + bounds(1);
    end
end

% this 'analytical' solution has numerical issues near theta=0, which sucks
% because we want close to 0 error
function w = logMap(R)
    theta = acos((R.trace()-1)/2);
    skw = (R-transpose(R))/2/sin(theta);
    w = skw.inv_skew();
    skew(w*theta);
end

function i = getCurrentPhase(k, Nch)
    i = 1;
    for j = 1 : length(Nch)-1
        i = i + (k > Nch(j));
    end
end

function newPos = legMask(pos, leg)
    if leg == 1
        newPos = pos;
    elseif leg == 2
        newPos = [pos(1);-pos(2);pos(3)];
    elseif leg == 3
        newPos = [-pos(1);pos(2);pos(3)];
    else
        newPos = [-pos(1);-pos(2);pos(3)];
    end
end