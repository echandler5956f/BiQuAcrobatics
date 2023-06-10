clear all; clc; close all

%% Unpack Solution

% Import the data

step_list = table2array(readtable(pwd + "\python\metadata\step_list"));
contact_list = table2array(readtable(pwd + "\python\metadata\contact_list"));
p_feet0 = table2array(readtable(pwd + "\python\metadata\p_feet0"));
p_feet_bar = table2array(readtable(pwd + "\python\metadata\p_feet_bar"));
r = 0.2;

n_p = size(step_list, 1);
Nch = cumsum(step_list);
Nc = sum(step_list);

p_body_opt = reshape(transpose(table2array(readtable(pwd + "\python\opt\p_body_opt"))), 3, 1, Nc);
dp_body_opt = reshape(transpose(table2array(readtable(pwd + "\python\opt\dp_body_opt"))), 3, 1, Nc);
Omega_opt = reshape(transpose(table2array(readtable(pwd + "\python\opt\Omega_opt"))), 3, 1, Nc);
DOmega_opt = reshape(transpose(table2array(readtable(pwd + "\python\opt\DOmega_opt"))), 3, 1, Nc);
R_opt = reshape(transpose(table2array(readtable(pwd + "\python\opt\R_opt"))), 3, 3, Nc);

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
r1 = rateControl((step_list(1)/T_opt(1))/slowDown);
r2 = rateControl((step_list(2)/T_opt(2))/slowDown);
% r3 = rateControl(100);
% if length(step_list) > 2
%     r3 = rateControl((step_list(3)/T_opt(2))/slowDown);
% end
plts = [];
while true
    for k = 1 : Nc
        i = getCurrentPhase(k, Nch);
        qc = [transpose(rotm2eul(R_opt(:,:,k), 'XYZ')); p_body_opt(:,:,k)];
        qcj = [qc; qj];
        plts = drawQuadruped(robot,qcj,p_feet0,p_feet_bar,r, ...
            R_opt(:,:,k),F_opt(:,:,k),plts, p_body_opt(:, :, 1));
        if i == 1
            waitfor(r1);
        elseif i == 2
            waitfor(r2);
%         else
%             waitfor(r3);
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
    old_plts, p0)
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
        tr = p0 + [-tmp(1);-tmp(2); tmp(3)];
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
