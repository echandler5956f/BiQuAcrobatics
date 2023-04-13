clear all; clc; close all

%% Initialization

import casadi.*

%% Constant Parameters

% Omega cost weight
eOmega = 1.0;

% Force cost weight
eF = 0.05;

% Rotation error cost weight
eR = 1.0;

% Minimum total time
tMin = 0.5;

% Maximum total time
tMax = 1.5;

% Number of contact phases
n_p = 2;

% Predefined number of steps for the ith contact phase
N = ones(n_p,1).*30;

% Helper for checking what contact we are in
Nch = cumsum(N);

% Total number of steps
Nc = sum(N);

% Friction coefficient
mu = 0.7;

% GRF limits
fz_bounds = [-25, 0;
             -25, 0;
             -25, 0];

% Acceleration due to gravity
g_accel = -9.81;

% COM bounding constraint. Ideally you would set this to some section of a 
% tube each timestep within you want the trajectory to lie
p_body_bounds = [-0.1, 0.1;
                 -0.1, 0.1;
                 0, 0.5];

% The sth foot position is constrained in a sphere of radius r to satisfy 
% joint limits. This parameter is the center of the sphere w.r.t the COM
p_feet_bar = [0.23;0.19;0.15];

% Sphere of radius r that the foot position is constrained to
r = ones(3,1).*0.16;

% Mass of the SRB
mass = 2.50000279;

% Inertia of SRB
inertia = [3.09249e-2, -9.00101e-7, 1.865287e-5;
         -8.00101e-7, 5.106100e-2, 1.245813e-4;
         1.865287e-5, 1.245813e-4, 6.939757e-2];

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

% Time derivative of angular velocity of SRB w.r.t body frame (3x1)
DOmega = {};

% Rotation matrix of the body frame (3x3)
R = {};

% GRF on each foot (3x4)
F = {};

for k = 1 : Nc
    p_body = {p_body{:}, MX.sym(['p_body_k' num2str(k)],3,1)};
    dp_body = {dp_body{:}, MX.sym(['dp_body_k' num2str(k)],3,1)};
    p_feet = {p_feet{:}, MX.sym(['p_feet_k' num2str(k)],3,4)};
    Omega = {Omega{:}, MX.sym(['Omega_k' num2str(k)],3,1)}; 
    DOmega = {DOmega{:}, MX.sym(['DOmega_k' num2str(k)],3,1)};
    R = {R{:}, MX.sym(['R_k' num2str(k)],3,3)};
    F = {F{:}, MX.sym(['F_k' num2str(k)],3,4)};
end

% Optimal contact timing for the ith contact phase (n_px1)
T = MX.sym('T',n_p,1);

%% Initial States

p_body0 = zeros(3,1);
dp_body0 = zeros(3,1);
tmp = kin.fk([deg2rad(-5);deg2rad(5);deg2rad(5)]);
p_feet0 = [legMask(tmp,1),legMask(tmp,2),legMask(tmp,3),legMask(tmp,4)];
Omega0 = zeros(3,1);
DOmega0 = zeros(3,1);
R0 = eye(3);

%% Final States

p_bodyf = [0;0.03;0];
p_feetg = p_feet0;
Rf = eye(3);

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
global g lbg ubg
w = {};
w0 = [];
lbw = [];
ubw = [];
J = 0;
g = {};
lbg = [];
ubg = [];

% Total time must be within our bounds. We only need to say this once
addConstraints(sum(T), tMin, tMax);

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
    
    % Time derivative of angular velocity of SRB w.r.t body frame (3x1)
    DOmega_k = DOmega{1,k};
    
    % Rotation matrix of the body frame (3x3)
    R_k = R{1,k};
    
    % GRF on each foot (3x4)
    F_k = F{1,k};

    %% Add Constraints

    % Add friction cone, GRF, and foot position constraints to each leg
    for leg = 1 : 4
        addConstraints(abs(F_k(1,leg)/F_k(end,leg)), -inf, mu);
        addConstraints(abs(F_k(2,leg)/F_k(end,leg)), -inf, mu);
        addConstraints(F_k(3,leg), fz_bounds(leg,1), fz_bounds(leg,2));
        addConstraints(abs(R_k*(p_feet_k(:,leg) - p_body_k) - ...
            p_feet_bar), zeros(3,1), r);
    end

    % Add body bounding box constraints
    addConstraints(p_body_k, p_body_bounds(:,1), p_body_bounds(:,2));

    % Discrete dynamics
    if i < Nc
        p_body_k1 = p_body{k+1};
        dp_body_k1 = dp_body{k+1};
        p_body_next = p_body_k + dp_body_k.*dt;
        dp_body_next = dp_body_k + ((sum(F_k,2)+g_accel)./mass).*dt;
        addConstraints(p_body_k1, p_body_next, p_body_next);
        addConstraints(dp_body_k1, dp_body_next, dp_body_next);
    end
    
    % Final States
    if i == Nc
        addConstraints(R_k.reshape(1,9), reshape(Rf,1,9), reshape(Rf,1,9))
    end
end

% Create an NLP solver
problem = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}));
solver = nlpsol('solver', 'ipopt', problem);

% Solve the NLP
sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw, ...
             'lbg', lbg, 'ubg', ubg);
w_opt = full(sol.x);

function addConstraints(g_k, lbg_k, ubg_k)
    global g lbg ubg
    if size(g_k,2) > 1 || size(lbg_k,2) > 1 || size(ubg_k,2) > 1
        disp("Invalid size of constraints. The constraints are one " + ...
            "column vector.");
    else
        if (size(g_k) == size(lbg_k)) && (size(g_k) == size(ubg_k)) ...
                && (size(lbg_k) == size(ubg_k))
            g = {g{:}, g_k};
            lbg = {lbg{:}, lbg_k};
            ubg = {ubg{:}, ubg_k};
        else
            disp("Invalid size of constraints. The number of upper " + ...
                "and lower bound constraints should match the size of g.");
        end
    end
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