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

robot = importrobot("huron\huron.urdf","DataFormat","column");
ik = inverseKinematics('RigidBodyTree', robot);
weights = [1 1 1 1 1 1];
initialguess = zeros(12,1);
tmp = getTransform(robot, zeros(12,1), 'L_FOOT', 'body');
tmp(3,4) = tmp(3,4);
rot = tmp(1:3,1:3);
pos = [0;0.0;0.0];
% pos = [0;0;0];
tform1 = [rot, [0;0;0];
         0, 0, 0, 1];
tform2 = [eul2rotm([0,0,0], 'XYZ'), pos + tmp(1:3,4);
          0, 0, 0, 1];
tform = tform2*tform1;
% trplot(tform)
% show(robot, zeros(12,1));
[configSoln,solnInfo] = ik('L_FOOT', tform, weights, initialguess);
% show(robot, configSoln);
result = getTransform(robot, configSoln, 'L_FOOT', 'body');
rerr = vex(logm(transpose(tform(1:3,1:3))*result(1:3,1:3)));
disp(transpose(rerr)*rerr)
disterr = tform(1:3,4)-result(1:3,4);
disp(disterr)
robot2 = importrobot("huron\huron_cheat.urdf","DataFormat","column");
bothlegs = [configSoln(1:6);configSoln(1);-configSoln(2);configSoln(3:5);-configSoln(6)];
disp(bothlegs)
disp(result)
q = [0;0;-tform2(3,4) + 0.0947957999709941;0;0;0;bothlegs];
disp(q);
initVisualizer(robot2, q);
com = centerOfMass(robot2, q);
disp(com);

function qj = getJointAngles(kin, p_body_k, R_k, p_feet_k, y0)
    T_wb = [transpose(R_k),-p_body_k; 
            0, 0, 0, 1];
    T_bf1 = T_wb*[p_feet_k(1:3,1);1];
    T_bf2 = T_wb*[p_feet_k(1:3,2);1];

    qj = [kin.ik([legMask(T_bf1(1:3,1),1)], y0(1:3));
          kin.ik([legMask(T_bf2(1:3,1),2)], y0(4:6))];
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