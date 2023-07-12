%% Initialize

clc; close all

animate = true;

% Kinematics

syms q [3, 1]
bodyHalfLength = 0.194;
bodyHalfWidth = 0.0875;
bodyHalfHeight = 0.025;
abadLinkLength = 0.01295;
hipLinkLength = 0.160;
kneeLinkY_offset = 0.04745;
kneeLinkLength = 0.1675;
mass = 2.50000279;
bounds = [deg2rad(-90), deg2rad(90);
          deg2rad(-125), deg2rad(125);
          deg2rad(-175), deg2rad(175)];
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

% Verify kinematics
% close all;
% p_feet_des = [[0.225;0.175;0], zeros(3)];
% q1 = getJointAngles(kin, [0;0;0.125], eye(3), p_feet_des,  [deg2rad(5);deg2rad(-45);deg2rad(-45);zeros(9,1)]);
% q1 = [-q1(1);q1(2:3)];
% % q1 = [deg2rad(0);deg2rad(-45);deg2rad(110)];
% % q1 = [deg2rad(5);deg2rad(-45);deg2rad(45)];
% % q1 = kin.ik([0;0;0.1], [0;0;0]);
% disp(q1)
% x1 = kin.fk([-q1(1);q1(2);q1(3)]);
% disp(x1);
% robot = importrobot("solo_12\urdf\solo_12_leg.urdf","DataFormat","column");
% initVisualizer(robot, q1);
% plot3(x1(1),x1(2),x1(3),'or');

%% Unpack Solution

% Import the data

step_list = table2array(readtable(pwd + "\python\solo_12\metadata\step_list"));
contact_list = table2array(readtable(pwd + "\python\solo_12\metadata\contact_list"));
p_feet0 = table2array(readtable(pwd + "\python\solo_12\metadata\p_feet0"));
p_feetf = table2array(readtable(pwd + "\python\solo_12\metadata\p_feetf"));
p_feet_bar = table2array(readtable(pwd + "\python\solo_12\metadata\p_feet_bar"));
r = table2array(readtable(pwd + "\python\solo_12\metadata\r"));

n_p = size(step_list, 1);
Nch = cumsum(step_list);
Nc = sum(step_list);

p_body_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\p_body_opt"))), 3, 1, Nc);
dp_body_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\dp_body_opt"))), 3, 1, Nc);
Omega_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\Omega_opt"))), 3, 1, Nc);
DOmega_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\DOmega_opt"))), 3, 1, Nc);
R_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\R_opt"))), 3, 3, Nc);
R_tmp = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\initial_guess\R_guess"))), 3, 3, Nc);
p_feet_opt = zeros(3, 4, Nc);

f_idx = [0;0;0;0];
for i = 1 : n_p
    f_idx = f_idx + transpose(contact_list(i, :));
end
F0_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\F0_opt"))), 3, 1, Nch(f_idx(1)));
F1_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\F1_opt"))), 3, 1, Nch(f_idx(2)));
F2_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\F2_opt"))), 3, 1, Nch(f_idx(3)));
F3_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\F3_opt"))), 3, 1, Nch(f_idx(4)));
T_opt = table2array(readtable(pwd + "\python\solo_12\opt\T_opt"));

% p_body_guess = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\initial_guess\p_body_guess"))), 3, 1, Nc);
% dp_body_guess = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\initial_guess\dp_body_guess"))), 3, 1, Nc);
% Omega_guess = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\initial_guess\Omega_guess"))), 3, 1, Nc);
% DOmega_guess = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\initial_guess\DOmega_guess"))), 3, 1, Nc);
% F0_guess = reshape(table2array(readtable(pwd + "\python\solo_12\initial_guess\F0_guess")), 3, 1, Nc);
% F1_guess = reshape(table2array(readtable(pwd + "\python\solo_12\initial_guess\F1_guess")), 3, 1, Nc);
% F2_guess = reshape(table2array(readtable(pwd + "\python\solo_12\initial_guess\F2_guess")), 3, 1, Nc);
% F3_guess = reshape(table2array(readtable(pwd + "\python\solo_12\initial_guess\F3_guess")), 3, 1, Nc);
% T_guess = table2array(readtable(pwd + "\python\solo_12\initial_guess\T_guess"));

F_opt = zeros(3, 4, Nc);
R_ref = [];
p_ref = [];

p = p_body_opt(:,:,1);
v = zeros(3,1);
for k = 1 : Nc
    i = getCurrentPhase(k, Nch);
    R_opt(:,:,k) = transpose(R_opt(:,:,k));
    R_ref = [R_ref; rotm2quat(transpose(R_tmp(:,:,k)))];
    % p_ref = [p_ref; transpose(p_body_opt(:,:,k))];
    p_ref = [p_ref; [2,0,0]];
    F_k = zeros(3, 4);
    if k < Nch(f_idx(1))
        F_k(:,1) = F0_opt(:,:,k);
        p_feet_opt(:,1,k) = p_feet0(:,1);
    else
        p_feet_opt(:,1,k) = p_feetf(:,1);
    end
    if k < Nch(f_idx(2))
        F_k(:,2) = F1_opt(:,:,k);
        p_feet_opt(:,2,k) = p_feet0(:,2);
    else
        p_feet_opt(:,2,k) = p_feetf(:,2);
    end
    if k < Nch(f_idx(3))
        F_k(:,3) = F2_opt(:,:,k);
        p_feet_opt(:,3,k) = p_feet0(:,3);
    else
        p_feet_opt(:,3,k) = p_feetf(:,3);
    end
    if k < Nch(f_idx(4))
        F_k(:,4) = F3_opt(:,:,k);
        p_feet_opt(:,4,k) = p_feet0(:,4);
    else
        p_feet_opt(:,4,k) = p_feetf(:,4);
    end
    F_opt(:,:,k) = F_k;
    if k <= Nch(2)
        v = v + (([sum(F_k(1,:));sum(F_k(2,:));sum(F_k(3,:))]/mass) - ...
            [0;0;9.81]) * (T_opt(i)/step_list(i));
        p = p + (v * (T_opt(i)/step_list(i)));
    end
end

% disp(v)
% disp(v/(T_opt(1) + T_opt(2)))
% disp(p + v*T_opt(3) - [0;0;9.81/2]*T_opt(3)^2)

% Test Body Bounds
% close all;
% des_height = 0.15;
% qc_test = [0;0;0;0;0;des_height];
% qj_test = getJointAngles(kin, [0;0;des_height], eye(3), p_feet_opt(:,:,1), zeros(12,1));
% qcj_test = [qc_test;qj_test];
% qcj_test = [qcj_test(1:6);
%     -qcj_test(7);qcj_test(8:9);
%     qcj_test(10:12);
%     -qcj_test(13:15);
%     qcj_test(16);-qcj_test(17:18)];
% robot = importrobot("solo_12\urdf\solo_12.urdf","DataFormat","column");
% 
% initVisualizer(robot, qcj_test);
% plot3(0,0,des_height, 'or');

%% Visualization
close all;

slowDown = 1;
rates = {};
t = zeros(Nc, 1);
it = 1;
for i = 1 : n_p
    dt = T_opt(i) / step_list(i);
    rates = {rates{:}, rateControl(slowDown/dt)};
    for k = 1 : step_list(i)
        if it ~= 1
            t(it, 1) = t(it-1, 1) + dt;
        end
        it = it + 1;
    end
end

if animate == true
    robot = importrobot("solo_12\urdf\solo_12.urdf","DataFormat","column");
    qc = [transpose(rotm2eul(R_opt(:,:,1), 'ZYX')); p_body_opt(:,:,1)]; % Pose
    
    qj = getJointAngles(kin, p_body_opt(:,:,1), R_opt(:,:,1), p_feet_opt(:,:,1), zeros(12,1));
    qcj = [qc;qj];
    initVisualizer(robot, qcj);
    
    % plotTransforms(p_ref, R_ref)
    plts = [];
    while true
        for k = 1 : Nc
            i = getCurrentPhase(k, Nch);
            qc = [transpose(rotm2eul(R_opt(:,:,k), 'ZYX')); p_body_opt(:,:,k)];
            qj = getJointAngles(kin, p_body_opt(:,:,k), R_opt(:,:,k), p_feet_opt(:,:,k), zeros(12,1));
            qcj = [qc; qj];
            plts = drawQuadruped(robot,qcj,p_feet_opt(:,:,k),p_feet_bar,r, ...
                R_opt(:,:,k),F_opt(:,:,k),p_body_opt(:,:,1),plts);
            waitfor(rates{i});
        end
    end
else
    F0_new = reshape(F0_opt, 3, length(F0_opt));
    F1_new = reshape(F1_opt, 3, length(F1_opt));
    F2_new = reshape(F2_opt, 3, length(F2_opt));
    F3_new = reshape(F3_opt, 3, length(F3_opt));
    
    tiledlayout(4, 1)
    nexttile
    plot(t(1:length(F0_new), 1), F0_new)
    nexttile
    plot(t(1:length(F1_new), 1), F1_new)
    legend('x', 'y', 'z')
    nexttile
    plot(t(1:length(F2_new), 1), F2_new)
    legend('x', 'y', 'z')
    nexttile
    plot(t(1:length(F3_new), 1), F3_new)
    legend('x', 'y', 'z')
end

function qj = getJointAngles(kin, p_body_k, R_k, p_feet_k, y0)
    T_wb = [transpose(R_k),-p_body_k; 
            0, 0, 0, 1];
    T_bf1 = T_wb*[p_feet_k(:,1);1];
    T_bf2 = T_wb*[p_feet_k(:,2);1];
    T_bf3 = T_wb*[p_feet_k(:,3);1];
    T_bf4 = T_wb*[p_feet_k(:,4);1];

    qj = [kin.ik(legMask(T_bf1(1:3,1),1), y0(1:3));
          kin.ik(legMask(T_bf2(1:3,1),2), y0(4:6));
          kin.ik(legMask(T_bf3(1:3,1),3), y0(7:9));
          kin.ik(legMask(T_bf4(1:3,1),4), y0(10:12))];
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
    p_body0, old_plts)
    p_body = q(4:6);
    q = [q(1:6);
        -q(7);q(8:9);
        q(10:12);
        -q(13:15);
        q(16);-q(17:18)];
    show(robot, q, "PreservePlot", false, "FastUpdate", true, ...
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
        plts = [plts; quiver3(p_feet(1,leg),p_feet(2,leg), ...
            p_feet(3,leg), F(1,leg),F(2,leg),F(3,leg), ...
            "Color",color,"LineWidth",2,"AutoScaleFactor",1, ...
            "ShowArrowHead","on")];
        tr = R*(p_body + p_feet_bar(:,leg));
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
