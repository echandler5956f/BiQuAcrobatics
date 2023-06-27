%% Initialize

clc; close all

% Kinematics

syms q [3, 1]
bodyHalfLength = 0.194;
bodyHalfWidth = 0.0875;
bodyHalfHeight = 0.025;
abadLinkLength = 0.01295;
hipLinkLength = 0.160;
kneeLinkY_offset = 0.04745;
kneeLinkLength = 0.1675;
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

step_list = table2array(readtable(pwd + "\python\metadata\step_list"));
contact_list = table2array(readtable(pwd + "\python\metadata\contact_list"));
p_feet0 = table2array(readtable(pwd + "\python\metadata\p_feet0"));
p_feetf = table2array(readtable(pwd + "\python\metadata\p_feetf"));
p_feet_bar = table2array(readtable(pwd + "\python\metadata\p_feet_bar"));
r = table2array(readtable(pwd + "\python\metadata\r"));

n_p = size(step_list, 1);
Nch = cumsum(step_list);
Nc = sum(step_list);

p_body_opt = reshape(transpose(table2array(readtable(pwd + "\python\opt\p_body_opt"))), 3, 1, Nc);
dp_body_opt = reshape(transpose(table2array(readtable(pwd + "\python\opt\dp_body_opt"))), 3, 1, Nc);
Omega_opt = reshape(transpose(table2array(readtable(pwd + "\python\opt\Omega_opt"))), 3, 1, Nc);
DOmega_opt = reshape(transpose(table2array(readtable(pwd + "\python\opt\DOmega_opt"))), 3, 1, Nc);
R_opt = reshape(transpose(table2array(readtable(pwd + "\python\opt\R_opt"))), 3, 3, Nc);
p_feet_opt = zeros(3, 4, Nc);

f_idx = [0;0;0;0];
for i = 1 : n_p
    f_idx = f_idx + transpose(contact_list(i, :));
end
F_0_opt = reshape(table2array(readtable(pwd + "\python\opt\F_0_opt")), 3, 1, Nch(f_idx(1)));
F_1_opt = reshape(table2array(readtable(pwd + "\python\opt\F_1_opt")), 3, 1, Nch(f_idx(2)));
F_2_opt = reshape(table2array(readtable(pwd + "\python\opt\F_2_opt")), 3, 1, Nch(f_idx(3)));
F_3_opt = reshape(table2array(readtable(pwd + "\python\opt\F_3_opt")), 3, 1, Nch(f_idx(4)));
T_opt = table2array(readtable(pwd + "\python\opt\T_opt"));

F_opt = zeros(3, 4, Nc);
for k = 1 : Nc
    i = getCurrentPhase(k, Nch);
    R_opt(:,:,k) = transpose(R_opt(:,:,k));
    F_k = zeros(3, 4);
    if k < Nch(f_idx(1))
        F_k(:,1) = F_0_opt(:,:,k);
        p_feet_opt(:,1,k) = p_feet0(:,1);
    else
        p_feet_opt(:,1,k) = p_feetf(:,1);
    end
    if k < Nch(f_idx(2))
        F_k(:,2) = F_1_opt(:,:,k);
        p_feet_opt(:,2,k) = p_feet0(:,2);
    else
        p_feet_opt(:,2,k) = p_feetf(:,2);
    end
    if k < Nch(f_idx(3))
        F_k(:,3) = F_2_opt(:,:,k);
        p_feet_opt(:,3,k) = p_feet0(:,3);
    else
        p_feet_opt(:,3,k) = p_feetf(:,3);
    end
    if k < Nch(f_idx(4))
        F_k(:,4) = F_3_opt(:,:,k);
        p_feet_opt(:,4,k) = p_feet0(:,4);
    else
        p_feet_opt(:,4,k) = p_feetf(:,4);
    end
    F_opt(:,:,k) = F_k;
end

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
robot = importrobot("solo_12\urdf\solo_12.urdf","DataFormat","column");
qc = [transpose(rotm2eul(R_opt(:,:,1), 'ZYX')); p_body_opt(:,:,1)]; % Pose

qj = getJointAngles(kin, p_body_opt(:,:,1), R_opt(:,:,1), p_feet_opt(:,:,1), zeros(12,1));
qcj = [qc;qj];
initVisualizer(robot, qcj);
slowDown = 1;
rates = {};
for i = 1 : n_p
    rates = {rates{:}, rateControl((step_list(i)/T_opt(i))/slowDown)};
end

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
