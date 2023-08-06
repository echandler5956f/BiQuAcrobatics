%% Initialize

clc; close all

% animate = true;
% visualizeReference = false;
% 
% % Kinematics
% 
% if true
%     syms q [3, 1]
%     L1 = 0.077;
%     L2 = 0.510;
%     L3 = 0.405;
%     bounds = [deg2rad(-180), deg2rad(180);
%               deg2rad(-180), deg2rad(180);
%               deg2rad(-180), deg2rad(180)];
%     PM = [eye(3), [L1;0;L2+L3];
%           0, 0, 0, 1];
%     DHTable = [0,0,-L1,0;
%                0,0,0,pi/2;
%                q(1)-pi/2,0,0,-pi/2;
%                q(2),0,L2,0;
%                q(3),0,L3,0;
%                0,0,0,pi/2];
% 
%     kin = kinematics.KinematicsDH(PM,DHTable,q,bounds,6,3);
% end
% 
% % Verify kinematics
% close all;
% p_feet_des = [[0;0;0], [0;0;0]];
% p_body = [0;0;L2+L3];
% q1 = getJointAngles(kin, p_body, eye(3), p_feet_des, zeros(6, 1));
% disp(q1)
% x1 = -kin.fk(q1(1:3));
% disp(x1);
% % q = [p_body-[0;0;L2+L3+L0]; 0; 0; 0; zeros(2, 1); q1(1:3); 0; zeros(2, 1); q1(4:6); 0];
% % q = [[0;0;0]; 0; 0; 0; zeros(2, 1); q1(1:3); 0; zeros(2, 1); q1(4:6); 0];

p_feet_bar = table2array(readtable(pwd + "\python\huron\metadata\p_feet_bar"));
contact_list = table2array(readtable(pwd + "\python\huron\metadata\contact_list"));
r = table2array(readtable(pwd + "\python\huron\metadata\r"));
step_list = table2array(readtable(pwd + "\python\huron\metadata\step_list"));

n_p = size(step_list, 1);
Nch = cumsum(step_list);
Nc = sum(step_list);

p_body_opt = reshape(transpose(table2array(readtable(pwd + "\python\huron\opt\p_body_opt"))), 3, Nc);
R_opt = reshape(transpose(table2array(readtable(pwd + "\python\huron\opt\R_opt"))), 3, 3, Nc);

p_fL_opt = reshape(transpose(table2array(readtable(pwd + "\python\huron\opt\p_fL"))), 3, Nc);
p_fR_opt = reshape(transpose(table2array(readtable(pwd + "\python\huron\opt\p_fR"))), 3, Nc);

T_opt = table2array(readtable(pwd + "\python\huron\opt\T_opt"));

robot = importrobot("huron\huron.urdf","DataFormat","column");
ik = inverseKinematics('RigidBodyTree', robot);
weights = [1 1 1 1 1 1];
initialguess = zeros(12,1);
tform = getTransform(robot, zeros(12,1), 'L_FOOT', 'body');
configSoln = ik('L_FOOT', tform, ones(6,1), initialguess);
result = getTransform(robot, configSoln, 'L_FOOT', 'body');
% rerr = vex(logm(transpose(tform(1:3,1:3))*result(1:3,1:3)));
% disp(transpose(rerr)*rerr)
% disterr = tform(1:3,4)-result(1:3,4);
% disp(disterr)
robot2 = importrobot("huron\huron_cheat.urdf","DataFormat","column");
bothlegs = [configSoln(1:6);configSoln(1);-configSoln(2);configSoln(3:5);-configSoln(6)];
% disp(bothlegs)
% disp(result)
q = [0;0;-tform(3,4) + 0.0947957999709941;0;0;0;bothlegs];
% disp(q);
initVisualizer(robot2, q);
% com = centerOfMass(robot2, q);
% disp(com);

plot3(p_body_opt(1,:),p_body_opt(2,:),p_body_opt(3,:),'b--');
% plot3(p_fR_opt(1,:),p_fR_opt(2,:),p_fR_opt(3,:),'ro')
% plot3(p_fL_opt(1,:),p_fL_opt(2,:),p_fL_opt(3,:),'yo')

% p_sphere_r = p_body_opt(:,1) + R_opt(:,:,1) * p_feet_bar(:,1);
% p_sphere_l = p_body_opt(:,1) + R_opt(:,:,1) * p_feet_bar(:,2);
% 
% [x,y,z] = sphere;
% 
% x_l = x*r + p_sphere_l(1);
% y_l = y*r + p_sphere_l(2);
% z_l = z*r + p_sphere_l(3);
% h_l = surfl(x_l,y_l,z_l);
% set(h_l, 'FaceAlpha', 0.25)
% 
% x_r = x*r + p_sphere_r(1);
% y_r = y*r + p_sphere_r(2);
% z_r = z*r + p_sphere_r(3);
% h_r = surfl(x_r,y_r,z_r);
% set(h_r, 'FaceAlpha', 0.25)
tmp1 = [];
tmp2 = [];
flag1 = true;
flag2 = true;
for k = 1 : Nc
    if norm(p_fL_opt(:,k)) > 1e-10
        flag1 = false;
        tmp1 = p_fL_opt(:,k);
    end
    if flag1
        p_fL_opt(:,k) = p_feet_bar(:,2);
    else
        if norm(p_fL_opt(:,k)) <= 1e-10
            p_fL_opt(:,k) = tmp1;
        end
    end

    if norm(p_fR_opt(:,k)) > 1e-10
        flag2 = false;
        tmp2 = p_fR_opt(:,k);
    end
    if flag2
        p_fR_opt(:,k) = p_feet_bar(:,1);
    else
        if norm(p_fR_opt(:,k)) <= 1e-10
            p_fR_opt(:,k) = tmp2;
        end
    end
end

slowDown = 1;
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

qj = zeros(12,1);
qcjmat = [];

for k = 1 : Nc
    i = getCurrentPhase(k, Nch);
    qc = [p_body_opt(1:2,k); p_body_opt(3,k); transpose(rotm2eul(R_opt(:,:,k), 'ZXY'))];
    qj = getJointAngles(ik, p_body_opt(:,k), R_opt(:,:,k), tform(1:3,1:3),p_fR_opt(:,k), p_fL_opt(:,k), qj);
    qcj = [qc; qj];
    qcjmat = [qcjmat,qcj];
end
%% Visualize qcjmat
close all; clc;
klist = [60;75;90;105;120;135;150;165;180;195;210];
initVisualizer(robot2, qcjmat(:,klist(1)));
plot3(p_body_opt(1,klist(1):Nc),p_body_opt(2,klist(1):Nc),p_body_opt(3,klist(1):Nc),'b',LineWidth=2.0);

% while true
%     for k = 45 : Nc
%         show(robot2, qcjmat(:,k), "PreservePlot", true, "FastUpdate", true, "Frames","off");
%         waitfor(rates{getCurrentPhase(k, Nch)});
%     end
% end
for k = 1 : 2 : length(klist)
    show(robot2, qcjmat(:,klist(k)), "PreservePlot", true, "FastUpdate", false, "Frames","off");
end

function qj = getJointAngles(ik, p_body_k, R_k, R_foot,p_fL_k, p_fR_k, y0)
    T_wb = [transpose(R_k),-p_body_k+0.0947957999709941; 
            0, 0, 0, 1];
    T_bf1 = T_wb*[p_fL_k(1:3);1];
    T_bf2 = T_wb*[p_fR_k(1:3);1];
    q_l = ik('L_FOOT',[R_foot,[legMask(T_bf1(1:3,1),1)];[0,0,0,1]],[1;0.001;0.001;1;1;1], y0);
    q_r = ik('L_FOOT',[R_foot,[legMask(T_bf2(1:3,1),2)];[0,0,0,1]],[1;0.001;0.001;1;1;1], y0);
    qj = [q_l(1);q_l(2);q_l(3:6);q_r(1);q_r(2);q_r(3:6)];
end

function initVisualizer(robot, qj)
    figure;
    ax = show(robot, qj, "PreservePlot", false, "FastUpdate", true, "Frames","off");
    hold on;
    ax.XLabel.String = "X (m)";
    ax.YLabel.String = "Y (m)";
    ax.ZLabel.String = "Z (m)";
    ax.XLim = [-0.5 3];
    ax.YLim = [-2 2];
    ax.ZLim = [0 1.25];
    view(ax, 159, 23);
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

function i = getCurrentPhase(k, Nch)
    i = 1;
    for j = 1 : length(Nch)-1
        i = i + (k > Nch(j));
    end
end

function newPos = legMask(pos, leg)
    if leg == 1
        newPos = pos;
    else
        newPos = [pos(1);-pos(2);pos(3)];
    end
end