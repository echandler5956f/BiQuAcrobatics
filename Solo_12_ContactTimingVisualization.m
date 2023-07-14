%% Initialize

clc; close all

animate = true;
visualizeReference = false;

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
bounds = [deg2rad(-180), deg2rad(180);
          deg2rad(-180), deg2rad(180);
          deg2rad(-180), deg2rad(180)];
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

p_body_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\p_body_opt"))), 3, Nc);
dp_body_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\dp_body_opt"))), 3, Nc);
Omega_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\Omega_opt"))), 3, Nc);
DOmega_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\DOmega_opt"))), 3, Nc);
R_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\R_opt"))), 3, 3, Nc);
R_ref = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\initial_guess\R_guess"))), 3, 3, Nc);
p_feet_opt = zeros(3, 4, Nc);

f_idx = [0;0;0;0];
for i = 1 : n_p
    f_idx = f_idx + transpose(contact_list(i, :));
end
F0_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\F0_opt"))), 3, Nc);
F1_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\F1_opt"))), 3, Nc);
F2_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\F2_opt"))), 3, Nc);
F3_opt = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\opt\F3_opt"))), 3, Nc);
T_opt = table2array(readtable(pwd + "\python\solo_12\opt\T_opt"));

p_body_guess = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\initial_guess\p_body_guess"))), 3, Nc);
dp_body_guess = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\initial_guess\dp_body_guess"))), 3, Nc);
Omega_guess = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\initial_guess\Omega_guess"))), 3, Nc);
DOmega_guess = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\initial_guess\DOmega_guess"))), 3, Nc);
F0_guess = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\initial_guess\F0_guess"))), 3, Nc);
F1_guess = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\initial_guess\F1_guess"))), 3, Nc);
F2_guess = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\initial_guess\F2_guess"))), 3, Nc);
F3_guess = reshape(transpose(table2array(readtable(pwd + "\python\solo_12\initial_guess\F3_guess"))), 3, Nc);
T_guess = table2array(readtable(pwd + "\python\solo_12\initial_guess\T_guess"));

F_opt = zeros(3, 4, Nc);
F_ref = zeros(3, 4, Nc);
R_qref = [];
p_qref = [];
eul_opt_xyz = zeros(3, Nc);
eul_ref_xyz = zeros(3, Nc);

p = p_body_opt(:,:,1);
v = zeros(3,1);
for k = 1 : Nc
    i = getCurrentPhase(k, Nch);
    R_opt(:,:,k) = transpose(R_opt(:,:,k));
    R_ref(:,:,k) = transpose(R_ref(:,:,k));
    R_qref = [R_qref; rotm2quat(transpose(R_ref(:,:,k)))];
    p_qref = [p_qref; transpose(p_body_opt(:,k))];
    % p_qref = [p_qref; [2,0,0]];
    eul_opt_xyz(:, k) = transpose(rad2deg(rotm2eul(R_opt(:,:,k), 'ZXY')));
    eul_ref_xyz(:, k) = transpose(rad2deg(rotm2eul(R_ref(:,:,k), 'ZXY')));
    eul_opt_xyz(:, k) = [eul_opt_xyz(2, k); eul_opt_xyz(3, k); eul_opt_xyz(1, k)];
    eul_ref_xyz(:, k) = [eul_ref_xyz(2, k); eul_ref_xyz(3, k); eul_ref_xyz(1, k)];
    F_k = zeros(3, 4);
    F_k_ref = zeros(3, 4);
    if k < Nch(f_idx(1))
        F_k(:,1) = F0_opt(:,k);
        F_k_ref(:,1) = F0_guess(:,k);
        p_feet_opt(:,1,k) = p_feet0(:,1);
    else
        F0_opt(:,k) = NaN(3,1);
        p_feet_opt(:,1,k) = p_feetf(:,1);
    end
    if k < Nch(f_idx(2))
        F_k(:,2) = F1_opt(:,k);
        F_k_ref(:,2) = F1_guess(:,k);
        p_feet_opt(:,2,k) = p_feet0(:,2);
    else
        F1_opt(:,k) = NaN(3,1);
        p_feet_opt(:,2,k) = p_feetf(:,2);
    end
    if k < Nch(f_idx(3))
        F_k(:,3) = F2_opt(:,k);
        F_k_ref(:,3) = F2_guess(:,k);
        p_feet_opt(:,3,k) = p_feet0(:,3);
    else
        F2_opt(:,k) = NaN(3,1);
        p_feet_opt(:,3,k) = p_feetf(:,3);
    end
    if k < Nch(f_idx(4))
        F_k(:,4) = F3_opt(:,k);
        F_k_ref(:,4) = F3_guess(:,k);
        p_feet_opt(:,4,k) = p_feet0(:,4);
    else
        F3_opt(:,k) = NaN(3,1);
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

slowDown = 5;
rates = {};
t = zeros(Nc, 1);
it = 1;
for i = 1 : n_p
    dt = T_opt(i) / step_list(i);
    rates = {rates{:}, rateControl(1/(dt*slowDown))};
    for k = 1 : step_list(i)
        if it ~= 1
            t(it, 1) = t(it-1, 1) + dt;
        end
        it = it + 1;
    end
end

if animate == true
    robot = importrobot("solo_12\urdf\solo_12.urdf","DataFormat","column");
    if ~visualizeReference
        qc = [p_body_opt(:,1); transpose(rotm2eul(R_opt(:,:,1), 'ZXY'))]; % Pose
        
        qj = getJointAngles(kin, p_body_opt(:,1), R_opt(:,:,1), p_feet_opt(:,:,1), zeros(12,1));
        qcj = [qc;qj];
        initVisualizer(robot, qcj);
        
        % plotTransforms(p_qref, R_qref)
        plts = [];
        while true
            for k = 1 : Nc
                i = getCurrentPhase(k, Nch);
                qc = [p_body_opt(:,k); transpose(rotm2eul(R_opt(:,:,k), 'ZXY'))];
                qj = getJointAngles(kin, p_body_opt(:,k), R_opt(:,:,k), p_feet_opt(:,:,k), zeros(12,1));
                qcj = [qc; qj];
                plts = drawQuadruped(robot,qcj,p_feet_opt(:,:,k),p_feet_bar,r, ...
                    R_opt(:,:,k),F_opt(:,:,k),p_body_opt(:,1),plts);
                waitfor(rates{i});
            end
        end
    else
        qc = [p_body_guess(:,1); transpose(rotm2eul(R_ref(:,:,1), 'ZXY'))]; % Pose
        
        qj = getJointAngles(kin, p_body_guess(:,1), R_ref(:,:,1), p_feet_opt(:,:,1), zeros(12,1));
        qcj = [qc;qj];
        initVisualizer(robot, qcj);
        
        % plotTransforms(p_qref, R_qref)
        plts = [];
        while true
            for k = 1 : Nc
                i = getCurrentPhase(k, Nch);
                qc = [p_body_guess(:,k); transpose(rotm2eul(R_ref(:,:,k), 'ZXY'))];
                qj = getJointAngles(kin, p_body_guess(:,k), R_ref(:,:,k), p_feet_opt(:,:,k), zeros(12,1));
                qcj = [qc; qj];
                plts = drawQuadruped(robot,qcj,p_feet_opt(:,:,k),p_feet_bar,r, ...
                    R_ref(:,:,k),F_ref(:,:,k),p_body_guess(:,1),plts);
                waitfor(rates{i});
            end
        end
    end
else
    colors = [0, 0.4470, 0.7410;
              0.6350, 0.0780, 0.1840; 
              0.9290, 0.6940, 0.1250];
    figure;
    tl1 = tiledlayout(4, 1);
    nexttile;
    hold on;
    plot(t, F0_opt');
    plot(t, F0_guess', '--');
    ax = gca;
    ax.ColorOrder = colors;
    legend('$f_{x}$', '$f_{y}$', '$f_{z}$', ...
        '$\hat{f}_{x}$', '$\hat{f}_{y}$', '$\hat{f}_{z}$', ...
        'Interpreter','latex', 'NumColumns', 3, 'Orientation','horizontal');
    nexttile;
    hold on;
    plot(t, F1_opt');
    plot(t, F1_guess', '--');
    ax = gca;
    ax.ColorOrder = colors;
    nexttile;
    hold on;
    plot(t, F2_opt');
    plot(t, F2_guess', '--');
    ax = gca;
    ax.ColorOrder = colors;
    nexttile;
    hold on;
    plot(t, F3_opt');
    plot(t, F3_guess', '--');
    ax = gca;
    ax.ColorOrder = colors;
    title(tl1, '$\vec{f}(t)$', 'Interpreter','latex');

    figure;
    tl2 = tiledlayout(2, 1);
    nexttile;
    hold on;
    plot(t, p_body_opt');
    plot(t, p_body_guess', '--');
    ax = gca;
    ax.ColorOrder = colors;
    legend('$p_{x}$', '$p_{y}$', '$p_{z}$', ...
        '$\hat{p}_{x}$', '$\hat{p}_{y}$', '$\hat{p}_{z}$', ...
        'Interpreter','latex', 'NumColumns', 3, 'Orientation','horizontal');
    nexttile;
    hold on;
    plot(t, dp_body_opt');
    plot(t, dp_body_guess', '--');
    ax = gca;
    ax.ColorOrder = colors;
    legend('$\dot{p}_{x}$', '$\dot{p}_{y}$', '$\dot{p}_{z}$', ...
        '$\dot{\hat{p}}_{x}$', '$\dot{\hat{p}}_{y}$', '$\dot{\hat{p}}_{z}$', ...
        'Interpreter','latex', 'NumColumns', 3, 'Orientation','horizontal');
    title(tl2, '$\vec{p}(t)$ and $\dot{\vec{p}}(t)$', 'Interpreter','latex');

    figure;
    tl3 = tiledlayout(1, 1);
    nexttile;
    hold on;
    plot(t, eul_opt_xyz');
    plot(t, eul_ref_xyz', '--');
    ax = gca;
    ax.ColorOrder = colors;
    legend('$\gamma_{x}$', '$\gamma_{y}$', '$\gamma_{z}$', ...
        '$\hat{\gamma}_{x}$', '$\hat{\gamma}_{y}$', '$\hat{\gamma}_{z}$', ...
        'Interpreter','latex', 'NumColumns', 3, 'Orientation','horizontal');
    title(tl3, '$\vec{\gamma}=$rotm2eul($R$, ZYX)', 'Interpreter','latex');

    figure;
    tl4 = tiledlayout(2, 1);
    nexttile;
    hold on;
    plot(t, Omega_opt');
    plot(t, Omega_guess', '--');
    ax = gca;
    ax.ColorOrder = colors;
    legend('$\Omega_{x}$', '$\Omega_{y}$', '$\Omega_{z}$', ...
        '$\hat{\Omega}_{x}$', '$\hat{\Omega}_{y}$', '$\hat{\Omega}_{z}$', ...
        'Interpreter','latex', 'NumColumns', 3, 'Orientation','horizontal');
    nexttile;
    hold on;
    plot(t, DOmega_opt');
    plot(t, DOmega_guess', '--');
    ax = gca;
    ax.ColorOrder = colors;
    legend('$\dot{\Omega}_{x}$', '$\dot{\Omega}_{y}$', '$\dot{\Omega}_{z}$', ...
        '$\dot{\hat{\Omega}}_{x}$', '$\dot{\hat{\Omega}}_{y}$', '$\dot{\hat{\Omega}}_{z}$', ...
        'Interpreter','latex', 'NumColumns', 3, 'Orientation','horizontal');
    title(tl4, '$\vec{\Omega}(t)$ and $\dot{\vec{\Omega}}(t)$', 'Interpreter','latex');

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
    p_body = q(1:3);
    q = [q(1:6);
        q(7);-q(8:9);
        -q(10:12);
        q(13:15);
        -q(16);q(17:18)];
    show(robot, q, "PreservePlot", false, "FastUpdate", true, ...
        "Frames","off");
    plts = [];
    % for leg = 1 : 4
    %     color = 'k';
    %     switch (leg)
    %         case 1
    %             color = 'r';
    %         case 2
    %             color = 'g';
    %         case 3
    %             color = 'b';
    %         case 4
    %             color = 'y';
    %         otherwise
    %             disp("ERROR")
    %     end
    %     plts = [plts; quiver3(p_feet(1,leg),p_feet(2,leg), ...
    %         p_feet(3,leg), F(1,leg),F(2,leg),F(3,leg), ...
    %         "Color",color,"LineWidth",2,"AutoScaleFactor",1, ...
    %         "ShowArrowHead","on")];
    %     tr = p_body + R*p_feet_bar(:,leg);
    %     [x,y,z] = sphere;
    %     x = x*r + tr(1);
    %     y = y*r + tr(2);
    %     z = z*r + tr(3);
    %     h = surfl(x,y,z);
    %     set(h, 'FaceAlpha', 0.25)
    %     plts = [plts; h];
    %     if ~isempty(old_plts)
    %         delete(old_plts(leg));
    %         delete(old_plts(leg+4));
    %     end
    % end
    % drawnow;
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
