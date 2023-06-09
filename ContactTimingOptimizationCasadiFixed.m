clear all; clc; close all

%% Initialization

import casadi.*

%% Constant Parameters

% Omega cost weight
eOmega = 0.0875;

% Force cost weight
eF = 0.0009125;

% Rotation error cost weight
eR = 0.1;

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
f_bounds = [-75, 75;
            -75, 75;
            -100, 100];

% Acceleration due to gravity
g_accel = [0;0;-9.81];

% COM bounding constraint. Ideally you would set this to some section of a 
% tube each timestep within you want the trajectory to lie
p_body_bounds = [-1.0, 1.0;
                 -1.0, 1.0;
                 -eps, 0.625];

% Velocity bounds to make the problem more solvable
dp_body_bounds = [-3, 3;
                  -3, 3;
                  -20.0, 20.0];

% Simple bounding box for all feet
p_feet_bounds = [-1.0, 1.0;
                 -1.0, 1.0;
                 -eps, 0.5];

% Angular velocity bounds to make the problem more solvable
Omega_bounds = [-1, 1;
                -1, 1;
                -1, 1];

% Time derivative angular velocity bounds to make the problem more solvable
DOmega_bounds = [-4, 4;
                 -4, 4;
                 -4, 4];

% The sth foot position is constrained in a sphere of radius r to satisfy 
% joint limits. This parameter is the center of the sphere w.r.t the COM
pbar = [0.23;0.19;-0.3];
p_feet_bar = [legMask(pbar,1),legMask(pbar,2),legMask(pbar,3),...
              legMask(pbar,4)];

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

% Time derivative of angular velocity of SRB w.r.t body frame (3x1)
DOmega = {};

% Rotation matrix of the body frame (3x3)
% Note: Rotation matrix is not actually a decision variable. if it was, 
% the problem would be overparameterized! It is dependent on the history 
% of Omega 
R = {};

% GRF on each foot (3x4)
F = {};

for k = 1 : Nc
    p_body = {p_body{:}, SX.sym(['p_body_k' num2str(k)],3,1)};
    dp_body = {dp_body{:}, SX.sym(['dp_body_k' num2str(k)],3,1)};
    Omega = {Omega{:}, SX.sym(['Omega_k' num2str(k)],3,1)}; 
    DOmega = {DOmega{:}, SX.sym(['DOmega_k' num2str(k)],3,1)};
    if k < N(1)
        p_feet = {p_feet{:}, SX.sym(['p_feet_k' num2str(k)],3,4)};
        F = {F{:}, SX.sym(['F_k' num2str(k)],3,4)};
    end
end

% Optimal contact timing for the ith contact phase (n_px1)
T = SX.sym('T',n_p,1);

% Function to get the matrix log transform (SO(3) -> so(3))
index_arg = SX.sym('index_arg');
R_arg = SX.sym('R_arg',3,3);
q_arg = SX.sym('q_arg',4,1);

f_000 = [sqrt(1+trace(R_arg))/2,; ...
        ((1/2)/(sqrt(1+trace(R_arg)))* ...
        vex(R_arg-transpose(R_arg)))];
f_001 = [((1/2)/sqrt(1+R_arg(1,1)-R_arg(2,2)-R_arg(3,3)))*...
            (R_arg(3,2)-R_arg(2,3)); ...
         ((1/2)*sqrt(1+R_arg(1,1)-R_arg(2,2)-R_arg(3,3))); ...
         ((1/2)/sqrt(1+R_arg(1,1)-R_arg(2,2)-R_arg(3,3)))*...
            (R_arg(2,1)+R_arg(1,2)); ...
         ((1/2)/sqrt(1+R_arg(1,1)-R_arg(2,2)-R_arg(3,3)))*...
            (R_arg(3,1)+R_arg(1,3))];
f_010 = [((1/2)/sqrt(1+R_arg(1,1)-R_arg(2,2)-R_arg(3,3)))*...
            (R_arg(1,3)-R_arg(3,1)); ...
         ((1/2)/sqrt(1+R_arg(1,1)-R_arg(2,2)-R_arg(3,3)))*...
            (R_arg(1,2)+R_arg(2,1)); ...
         ((1/2)*sqrt(1+R_arg(1,1)-R_arg(2,2)-R_arg(3,3))); ...
         ((1/2)/sqrt(1+R_arg(1,1)-R_arg(2,2)-R_arg(3,3)))*...
            (R_arg(3,2)+R_arg(2,3))];
f_011 = [((1/2)/sqrt(1+R_arg(1,1)-R_arg(2,2)-R_arg(3,3)))*...
            (R_arg(2,1)-R_arg(1,2)); ...
         ((1/2)/sqrt(1+R_arg(1,1)-R_arg(2,2)-R_arg(3,3)))*...
            (R_arg(1,3)+R_arg(3,1)); ...
         ((1/2)/sqrt(1+R_arg(1,1)-R_arg(2,2)-R_arg(3,3)))*...
            (R_arg(2,3)+R_arg(3,2)); ...
         ((1/2)*sqrt(1+R_arg(1,1)-R_arg(2,2)-R_arg(3,3)))];
f_master1 = [f_000,f_001,f_010,f_011];
f_master_cell1 = horzsplit(f_master1);
f_c1 = conditional(index_arg,f_master_cell1,0,false);
matrix_to_quat = Function('matrix_to_quat',{index_arg,R_arg},{f_c1});

f_11 = ((2/q_arg(1))-(2/3)*((norm(q_arg(2:4))^2)/ ...
        (q_arg(1)^3))).*q_arg(2:4);
f_10 = 4*atan2(norm(q_arg(2:4)), (q_arg(1)+ ...
       sqrt((q_arg(1)^2)+(norm(q_arg(2:4))^2)))).* ...
       (q_arg(2:4)/norm(q_arg(2:4)));
f_master2 = [f_11,f_10];
f_master_cell2 = horzsplit(f_master2);
f_c2 = conditional(index_arg,f_master_cell2,0,false);
quat_to_axis_angle = Function('quat_to_axis_angle', ...
    {index_arg,q_arg},{f_c2});

%% Initial States

p_body0 = [0;0;0.35];
dp_body0 = zeros(3,1);
R0 = eul2rotm([deg2rad(-0.2),deg2rad(0.1),deg2rad(5)], 'XYZ');
tmp = kin.fk([deg2rad(-5);deg2rad(5);deg2rad(5)]);
p_feet0 = [p_body0 + R0*legMask(tmp,1),p_body0 + R0*legMask(tmp,2), ...
    p_body0 + R0*legMask(tmp,3), p_body0 + R0*legMask(tmp,4)];
Omega0 = zeros(3,1);
DOmega0 = zeros(3,1);

%% Final States

p_bodyf = [bodyHalfLength*2;0;0.25];
Rf = eul2rotm([deg2rad(2),deg2rad(-6),deg2rad(0)], 'XYZ');
tmp = kin.fk([deg2rad(-20);deg2rad(30);deg2rad(5)]);
p_feetf = [p_bodyf + Rf*legMask(tmp,1),p_bodyf + Rf*legMask(tmp,2), ...
    p_bodyf + Rf*legMask(tmp,3), p_bodyf + Rf*legMask(tmp,4)];

%% Reference Trajectory

R_ref = zeros(3,3,Nc);
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

global p_body_idx dp_body_idx p_feet_idx Omega_idx DOmega_idx F_idx ...
T_idx currentIndex
p_body_idx = {};
dp_body_idx = {};
p_feet_idx = {};
Omega_idx = {};
DOmega_idx = {};
F_idx = {};
T_idx = {};
currentIndex = 1;

% Total time must be within our bounds
addGeneralConstraints(T.sum(), tMin, tMax);

% All contact timings must be positive
addDesignConstraintsAndInit(T, ones(n_p,1).*(tMin/(n_p+1)), ...
    ones(n_p,1).*(tMax), ones(n_p,1).*((tMax-tMin)/n_p), 'T');

R_k = R0;

for k = 1 : Nc
    %% Gather Decision Variables

    i = getCurrentPhase(k, Nch);
    dt = T(i)/N(i);

    % COM of the body (3x1)
    p_body_k = p_body{1,k};
    
    % Time derivative of the COM of the body (3x1)
    dp_body_k = dp_body{1,k};
    
    % Angular velocity of SRB w.r.t body frame (3x1)
    Omega_k = Omega{1,k};

    % Time derivative of angular velocity of SRB w.r.t body frame (3x1)
    DOmega_k = DOmega{1,k};
    
    % Rotation matrix of the body frame (3x3)
    % Not a decision variable!
    R_k = R_k*approximateExpA(skew(Omega_k/dt), 4);

    % Foot position in the world frame of each foot (3x4)
    p_feet_k = p_feet{1,k};
    
    % GRF on each foot (3x4)
    F_k = F{1,k};

    %% Add Constraints

    if k ~= 1
        % Add dummy constraints
        addDesignConstraintsAndInit(dp_body_k, dp_body_bounds(:,1), ...
                dp_body_bounds(:,2), dp_body0, 'dp_body');
        addDesignConstraintsAndInit(Omega_k, Omega_bounds(:,1), ...
                Omega_bounds(:,2), Omega0, 'Omega');
        addDesignConstraintsAndInit(DOmega_k, DOmega_bounds(:,1), ...
                DOmega_bounds(:,2), DOmega0, 'DOmega');
        if k ~= Nc
            % Add body bounding box constraints
            addDesignConstraintsAndInit(p_body_k, p_body_bounds(:,1), ...
                p_body_bounds(:,2), p_body0, 'p_body');
    
            for leg = 1 : 4
                if i == 1
                    addDesignConstraintsAndInit(p_feet_k(:,leg), ...
                        p_feet0(:,leg), p_feet0(:,leg), ...
                        p_feet0(:,leg), 'p_feet');
                end
            end
   
        end
    end

    % Add friction cone, GRF, and foot position constraints to each leg
    grf = zeros(3,1);
    tau = zeros(3,1);
        for leg = 1 : 4
            if i == 1
                tau = tau + cross(F_k(:,leg),(p_body_k-p_feet_k(:,leg)));
                grf = grf + F_k(:,leg);
                addGeneralConstraints(abs(F_k(1,leg)/F_k(3,leg)), 0, mu);
                addGeneralConstraints(abs(F_k(2,leg)/F_k(3,leg)), 0, mu);
                addGeneralConstraints(abs(R_k*(p_feet_k(:,leg) - ...
                    p_body_k) - p_feet_bar(:,leg)), zeros(3,1), r);
                addDesignConstraintsAndInit(F_k(:,leg), f_bounds(:,1), ...
                    f_bounds(:,2), [0;0;mass/4], 'F');
            end
        end

    % Discrete dynamics
    if k < Nc
        p_body_k1 = p_body{1,k+1};
        dp_body_k1 = dp_body{1,k+1};
        Omega_k1 = Omega{1,k+1};
        DOmega_k1 = DOmega{1,k+1};

        p_body_next = p_body_k + dp_body_k.*dt;
        dp_body_next = dp_body_k + ((grf./mass) + g_accel).*dt;
        Omega_next = Omega_k + DOmega_k.*dt;
        DOmega_next = DOmega_k + invinertia*((transpose(R_k)*tau) - ...
            cross(Omega_k,(inertia*Omega_k))).*dt;

        addGeneralConstraints(p_body_k1-p_body_next, zeros(3,1), ...
            zeros(3,1));
        addGeneralConstraints(dp_body_k1-dp_body_next, zeros(3,1), ...
            zeros(3,1));
        addGeneralConstraints(Omega_k1-Omega_next, zeros(3,1), zeros(3,1));
        addGeneralConstraints(DOmega_k1-DOmega_next, zeros(3,1), ...
            zeros(3,1));
    end

    % Initial States
    if k == 1
        addDesignConstraintsAndInit(p_body_k, p_body0, p_body0, ...
            p_body0, 'p_body');
        addDesignConstraintsAndInit(dp_body_k, dp_body0, dp_body0, ...
            dp_body0, 'dp_body');
        addDesignConstraintsAndInit(Omega_k, Omega0, Omega0, Omega0, ...
            'Omega');
        addDesignConstraintsAndInit(DOmega_k, DOmega0, DOmega0, ...
            DOmega0, 'DOmega');
        for leg = 1 : 4
            addDesignConstraintsAndInit(p_feet_k(:,leg), ...
                p_feet0(:,leg), p_feet0(:,leg), p_feet0(:,leg), 'p_feet');
        end
    end
    
    % Final States
    if k == Nc
        addDesignConstraintsAndInit(p_body_k, p_bodyf, p_bodyf, ...
            p_bodyf, 'p_body');
        addGeneralConstraints(R_k.reshape(9,1), reshape(Rf,9,1), ...
            reshape(Rf,9,1));
    end

    %% Objective Function
    
    % Calculate rotation matrix error term
    R_err_k = transpose(R_ref(:,:,k))*R_k;
    s1 = trace(R_err_k) > 0;
    s2 = (R_err_k(1,1)>=R_err_k(2,2))*(R_err_k(1,1)>=R_err_k(3,3));
    s3 = (R_err_k(2,2)>R_err_k(1,1))*(R_err_k(2,2)>=R_err_k(3,3));
    s4 = (R_err_k(3,3)>R_err_k(1,1))*(R_err_k(3,3)>R_err_k(2,2));
    index_s = -1 + s1 + ...
              2*(~s1)*s2 + ...
              3*(~s1)*(~s2)*s3 + ...
              4*(~s1)*(~s2)*(~s3)*s4;
    q_k = matrix_to_quat(index_s, R_err_k);
    q_k = sign(q_k(1)).*q_k;
    e_R_k = quat_to_axis_angle(0, q_k);

    J = J + (eOmega.*transpose(Omega_k)*Omega_k) + ...
        (eF.*transpose(grf)*grf) + (eR.*transpose(e_R_k)*e_R_k);
end

%% Solve Problem

% Create an NLP solver
options = struct('expand', true, 'ipopt', struct('max_iter', 100000, ...
    'fixed_variable_treatment', 'make_constraint', ...
    'mumps_mem_percent', 10000, 'print_level', 5));
problem = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}));
solver = nlpsol('solver', 'ipopt', problem, options);

% Solve the NLP
sol = solver('x0', vertcat(w0{:}), 'lbx', vertcat(lbw{:}), 'ubx', ...
    vertcat(ubw{:}), 'lbg', vertcat(lbg{:}), 'ubg', vertcat(ubg{:}));
w_opt = full(sol.x);

%% Unpack Solution

p_body_opt = unpackIndices(w_opt, p_body_idx, 3, 1, false);
dp_body_opt = unpackIndices(w_opt, dp_body_idx, 3, 1, false);
p_feet_opt = unpackIndices(w_opt, p_feet_idx, 3, 4, true);
Omega_opt = unpackIndices(w_opt, Omega_idx, 3, 1, false);
DOmega_opt = unpackIndices(w_opt, DOmega_idx, 3, 1, false);
R_opt = R;
F_opt = unpackIndices(w_opt, F_idx, 3, 4, true);
T_opt = unpackIndices(w_opt, T_idx, 2, 1, false);

%% Visualization
close all;
robot = importrobot("solo_12\urdf\solo_12.urdf","DataFormat","column");
qc = [transpose(rotm2eul(R_opt(:,:,1), 'XYZ')); p_body_opt(:,:,1)]; % Pose
qj = [0;0;0;       % Leg 1
      0;0;0;       % Leg 2
      0;0;0;       % Leg 3
      0;0;0];      % Leg 4
qcj = [qc;qj];
initVisualizer(robot, qcj);
slowDown = 2;
r1 = rateControl((N(1)/T_opt(1))/slowDown);
r2 = rateControl((N(2)/T_opt(2))/slowDown);
plts = [];
while true
    for k = 1 : Nc
        i = getCurrentPhase(k, Nch);
        qc = [transpose(rotm2eul(R_opt(:,:,k), 'ZYX')); p_body_opt(:,:,k)];
        qcj = [qc; qj];
        plts = drawQuadruped(robot,qcj,p_feet_opt(:,:,k),p_feet_bar,r(1), ...
            R_opt(:,:,k),F_opt(:,:,k),plts);
        if i == 1
            waitfor(r1);
        else
            waitfor(r2);
        end
    end
end

function initVisualizer(robot, qj)
    figure;
    ax = show(robot, qj, "PreservePlot", false,"Frames","on");
    hold on;
    ax.XLabel.String = "X (m)";
    ax.YLabel.String = "Y (m)";
    ax.ZLabel.String = "Z (m)";
    ax.XLim = [-0.5 3];
    ax.YLim = [-2 2];
    ax.ZLim = [-0.1 1];
    view(ax, 135, 25);
    scale = 20;
    % hack to draw a checkerboard with a surface
    [mX, mY] = meshgrid(-0.025*scale:0.025*scale:0.15*scale, ...
        -0.125*scale:0.025*scale:0.125*scale);
    J = checkerboard(1, 5, 4) > 0.5;
    J(:,:,2) = J(:,:);
    J(:,:,3) = J(:,:,1);
    J = cast(J, "double");
    checkSurf = surf(ax, mX, mY, 0 * mX, "FaceColor","flat");
    checkSurf.CDataMode = "manual";
    checkSurf.CData = J;
end

function plts = drawQuadruped(robot, q, p_feet, p_feet_bar, r, R, F, ...
    old_plts)
    p_body = [-q(4);-q(5);q(6)];

    q_t = [q(1)+pi;q(2:3);p_body;q(7:end)];
    show(robot, q_t, "PreservePlot", false, "FastUpdate", true, ...
        "Frames","on");
    plts = [];
    for leg = 1 : 4
        color = 'k';
        switch (leg)
            case 1
                color = 'r';
            case 2
                color = 'g';
            case 3
                color = 'b';
            case 4
                color = 'y';
            otherwise
                disp("ERROR")
        end
        plts = [plts; quiver3(-p_feet(1,leg),-p_feet(2,leg), ...
            p_feet(3,leg), -F(1,leg),-F(2,leg),F(3,leg), ...
            "MarkerEdgeColor",color)];
        tmp = p_feet_bar(:,leg);
        tr = (-[p_body(1);p_body(2);-p_body(3)]) - R*[tmp(1);tmp(2);
            -tmp(3)];
        [x,y,z] = sphere;
        x = x*r + tr(1);
        y = y*r + tr(2);
        z = z*r + tr(3);
        h = surfl(x,y,z);
        set(h, 'FaceAlpha', 0.25)
        plts = [plts; h];
        if ~isempty(old_plts)
            delete(old_plts(leg));
            delete(old_plts(leg+4));
        end
    end
    drawnow limitrate;
end

function optDesignVars = unpackIndices(w_opt, designIndices, s1, s2, split)
    n = length(designIndices);
    if split == false
        optDesignVars = zeros(s1,s2,n);
        for i = 1 : n
            optDesignVars(:,:,i) = ...
            reshape(w_opt(vertcat(designIndices{i}),1),s1,s2);
        end
    else
        optDesignVars = zeros(s1,s2,floor(n/s2));
        for i = 1 : floor(n/s2)
            for j = 1 : s2
                tmp = designIndices(i:i+s2-1);
                optDesignVars(:,j,i) = reshape(w_opt(vertcat( ...
                    tmp{j}),1),s1,1);
            end
        end
    end
end

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

function addDesignConstraintsAndInit(w_k, lbw_k, ubw_k, w0_k, field)
    global w lbw ubw w0 p_body_idx dp_body_idx p_feet_idx Omega_idx ...
    DOmega_idx R_idx F_idx T_idx currentIndex
    if size(w_k,2) > 1 || size(lbw_k,2) > 1 || size(ubw_k,2) > 1 || ...
            size(w0_k,2) > 1
        disp("Invalid size of constraints. The constraints are one " + ...
            "column vector.");
    else
        if all(size(w_k) == size(lbw_k)) && all(size(w_k) == ...
                size(ubw_k)) && all(size(lbw_k) == size(ubw_k)) && ...
                all(size(w_k) == size(w0_k))
            n = size(w_k);
            w = {w{:}, w_k};
            lbw = {lbw{:}, lbw_k};
            ubw = {ubw{:}, ubw_k};
            w0 = {w0{:}, w0_k};
            a = currentIndex;
            b = currentIndex + n-1;
            switch (field)
                case 'p_body'
                    p_body_idx = {p_body_idx{:}, a : b};
                case 'dp_body'
                    dp_body_idx = {dp_body_idx{:}, a : b};
                case 'p_feet'
                    p_feet_idx = {p_feet_idx{:}, a : b};
                case 'Omega'
                    Omega_idx = {Omega_idx{:}, a : b};
                case 'DOmega'
                    DOmega_idx = {DOmega_idx{:}, a : b};
                case 'R'
                    R_idx = {R_idx{:}, a : b};
                case 'F'
                    F_idx = {F_idx{:}, a : b};
                case 'T'
                    T_idx = {T_idx{:}, a : b};
            end
            currentIndex = b+1;
        else
            disp("Invalid size of constraints. The number of upper " + ...
                "and lower bound constraints should match the size of w.");
        end
    end
end

function expA = approximateExpA(A, deg)
    expA = zeros(3,3);
    for i = 0 : deg
        expA = expA + (A^i)/(factorial(i));
    end
end

% n x 2 matrix of lower[1] and upper[2] bounds
function r = randInBounds(bounds)
    n = length(bounds);
    r = zeros(n,1);
    for i=1:n
        r(i,1) = (bounds(i,2)-bounds(i,1))*rand(1) + bounds(i,1);
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