function model = huron()
clear all;close all;clc
L0 = 0.077*4;
L1 = 0.510;
L2 = 0.405;

% vector elements are ordered as: [mass;
%                                  com_x;com_y;com_z;
%                                  ixx;ixy;ixz;iyy;iyz;izz]

% todo - include the battery and imu spatial inertia in base_footprint

% base_footprint
base = [3.4072;
        0;0;0.026975;0.00319;
        7.149E-11;6.0313E-11;0.010531;6.926E-09;0.012315];

% l_hip_yaw_link
lhy = [1.6968;
       0.00063365;-0.00053469;0.063496;
       0.0022702;-2.5061E-10;6.6464E-06;0.0026746;-5.1468E-09;0.001238];

% l_hip_roll_link
lhr = [1.1884;
       0.07196;0.0061153;-0.0005211;
       0.0009822;-9.9265E-08;4.9766E-09;0.00097045;-9.7282E-08;0.00092365];

% l_hip_pitch_link
lhp = [4.2544;
       0.26752;4.6211E-05;0.053152;
       0.0034527;-3.1056E-08;0.0019797;0.010032;-1.4618E-10;0.01009];

% l_knee_pitch_link
lkp = [3.4875;
       0.13547;0.00068516;0.00486;
       0.0064814;1.2524E-05;-2.7966E-05;0.019762;1.0417E-05;0.017354];

% l_ankle_pitch_link
lap = [2.4242;
       -0.029808;0.001045;3.1135E-08;
       0.0025143;-0.00025651;1.1633E-12;0.0045125;6.7196E-10;0.0049235];

% l_ankle_roll_link
lar = [1.8533;
       0.073422;-5.3261E-05;0.063946;
       0.008682;-1.0255E-05;0.00024351;0.0084057;0.00012332;0.0013019];

% r_hip_yaw_link
rhy = [1.6968;
       0.0006244;0.00052023;0.063505;
       0.0022729;-5.6517E-10;6.6464E-06;0.0026746;5.6889E-09;0.0012354];

% r_hip_roll_link
rhr = [1.1884;
       -0.07196;0.0061152;-0.00052563;
       0.00098222;9.8947E-08;9.2484E-09;0.00097045;9.7414E-08;0.00092365];

% r_hip_pitch_link
rhp = [4.2544;
       0.26753;4.9957E-05;-0.053132;
       0.0034501;-2.7593E-09;-0.0019797;0.010035;2.5351E-10;0.01009];

% r_knee_pitch_link
rkp = [3.4875;
       0.13396;0.00063335;-0.00486;
       0.0064812;1.2949E-05;2.8064E-05;0.019762;-1.0389E-05;0.017354];

% r_ankle_pitch_link
rap = [2.4242;
       -0.029867;0.001045;0.000556;
       0.0025143;-0.00025651;1.5321E-12;0.0045125;9.9589E-10;0.0049235];

% r_ankle_roll_link
rar = [1.8533;
       0.073422;7.2094E-05;0.063933;
       0.0086813;1.0356E-05;0.00024346;0.0084051;-0.00012301;0.0013018];

model.NQ = 18;
model.NB = 13;
model.NB_fb = 6;
model.mass = 20.36039;
model.parent = [0 1 1 2 3 4 5 6 7 8 9 10 11];

model.jtype = {'R','Rz','Rz','Rz','Rz','Rz','Rz','Rz','Rz','Rz','Rz','Rz','Rz'};
model.Xtree = { eye(6), ...
                inv(pluho(tdh(0,0,-L0,0))), ...
                inv(pluho(tdh(0,0,L0,pi))), ...
                inv(pluho(tdh(0,0,0,pi/2))), ...
                inv(pluho(tdh(0,0,0,pi/2))), ...
                inv(pluho(tdh(-pi/2,0,0,-pi/2))), ...
                inv(pluho(tdh(-pi/2,0,0,-pi/2))), ...
                inv(pluho(tdh(0,0,L1,0))), ...
                inv(pluho(tdh(0,0,-L1,0))), ...
                inv(pluho(tdh(0,0,L2,0))), ...
                inv(pluho(tdh(0,0,-L2,0))), ...
                inv(pluho(tdh(0,0,0,pi/2))), ...
                inv(pluho(tdh(0,0,0,pi/2))) };

model.gravity = [0;0;-9.81];

model.I = { mcI(base(1), base(2:4), VtoRI(base)), ...
            mcI(lhy(1), lhy(2:4), VtoRI(lhy)), ...
            mcI(lhr(1), lhr(2:4), VtoRI(lhr)), ...
            mcI(lhp(1), lhp(2:4), VtoRI(lhp)), ...
            mcI(lkp(1), lkp(2:4), VtoRI(lkp)), ...
            mcI(lap(1), lap(2:4), VtoRI(lap)), ...
            mcI(lar(1), lar(2:4), VtoRI(lar)), ... 
            mcI(rhy(1), rhy(2:4), VtoRI(rhy)), ...
            mcI(rhr(1), rhr(2:4), VtoRI(rhr)), ...
            mcI(rhp(1), rhp(2:4), VtoRI(rhp)), ...
            mcI(rkp(1), rkp(2:4), VtoRI(rkp)), ...
            mcI(rap(1), rap(2:4), VtoRI(rap)), ...
            mcI(rar(1), rar(2:4), VtoRI(rar)) };

model.appearance.body{1} = ...
    { 'box', [-0.1 -0.1 -0.1; 0.1 0.1 0.1], ...
    'colour', [0.6 0.3 0.8] };

model.appearance.body{2} = ...
    { 'box', [-0.1 -0.1 -0.1; 0.1 0.1 0.1], ...
    'colour', [0.6 0.3 0.8] };

model.appearance.body{3} = ...
    { 'box', [-0.1 -0.1 -0.1; 0.1 0.1 0.1], ...
    'colour', [0.6 0.3 0.8] };

model.appearance.body{4} = ...
    { 'box', [-0.1 -0.1 -0.1; 0.1 0.1 0.1], ...
    'colour', [0.6 0.3 0.8] };

model.appearance.body{5} = ...
    { 'box', [-0.1 -0.1 -0.1; 0.1 0.1 0.1], ...
    'colour', [0.6 0.3 0.8] };

model.appearance.body{6} = ...
    { 'box', [-0.1 -0.1 -0.1; 0.1 0.1 0.1], ...
    'colour', [0.6 0.3 0.8] };

model.appearance.body{7} = ...
    { 'box', [-0.1 -0.1 -0.1; 0.1 0.1 0.1], ...
    'colour', [0.6 0.3 0.8] };

model.appearance.body{8} = ...
    { 'box', [-0.1 -0.1 -0.1; 0.1 0.1 0.1], ...
    'colour', [0.6 0.3 0.8] };

model.appearance.body{9} = ...
    { 'box', [-0.1 -0.1 -0.1; 0.1 0.1 0.1], ...
    'colour', [0.6 0.3 0.8] };

model.appearance.body{10} = ...
    { 'box', [-0.1 -0.1 -0.1; 0.1 0.1 0.1], ...
    'colour', [0.6 0.3 0.8] };

model.appearance.body{11} = ...
    { 'box', [-0.1 -0.1 -0.1; 0.1 0.1 0.1], ...
    'colour', [0.6 0.3 0.8] };

model.appearance.body{12} = ...
    { 'box', [-0.1 -0.1 -0.1; 0.1 0.1 0.1], ...
    'colour', [0.6 0.3 0.8] };

model.appearance.body{13} = ...
    { 'box', [-0.1 -0.1 -0.1; 0.1 0.1 0.1], ...
    'colour', [0.6 0.3 0.8] };

model.fb_type = 'eul';

model.camera.direction = [0 -3 1];	% a better viewing angle
model.camera.zoom = 0.9;

% replace joint 1 with a chain of 6 joints emulating a floating base
model = floatbase(model);
% time = linspace(1, 10, 1000);
% q = [zeros(6, length(time)); % X, Y, Z cartesian, X, Y, Z euler angles
%     zeros(1, length(time)); % l_hip_yaw
%     zeros(1, length(time)); % r_hip_yaw
%     zeros(1, length(time)); % l_hip_roll
%     zeros(1, length(time)); % r_hip_roll
%     zeros(1, length(time)); % l_hip_pitch
%     zeros(1, length(time)); % r_hip_pitch
%     zeros(1, length(time)); % l_knee_pitch
%     zeros(1, length(time)); % r_knee_pitch
%     zeros(1, length(time)); % l_ankle_pitch
%     zeros(1, length(time)); % r_ankle_pitch
%     zeros(1, length(time)); % l_ankle_roll
%     zeros(1, length(time)) % r_ankle_roll
%     ]; 
% showmotion(model)

% xfb = [0;0;0;1;zeros(9,1)];
% q = zeros(18,1);
% qd = zeros(18,1);
% qdd = zeros(18,1);
% tau = zeros(18,1);

% disp("Mass Matrix:");
% disp(get_mass_matrix(model, q));
% 
% disp("Coriolis and Centripetal Forces:");
% disp(get_generalized_Corilois_force(model, q, qd));
% 
% disp("Gravitational Torque Vector:");
% disp(get_generalized_gravity_force(model, q));
% 
% [xdfb_res, qdd_res] = FDfb(model, xfb, q, qd, tau);
% disp("Forward Dynamics 'xdfb' for Floating Base:");
% disp(xdfb_res);
% 
% disp("Forward Dynamics 'qdd' for Floating Base:");
% disp(qdd_res);
% 
% [xdfb_res, tau_res] = IDfb(model, xfb, q, qd, qdd);
% disp("Inverse Dynamics 'xdfb' for Floating Base:");
% disp(xdfb_res);
% 
% disp("Inverse Dynamics 'tau' for Floating Base:");
% disp(tau_res);
% 
% disp("Energy and Momentum:");
% disp(EnerMo(model, q, qd));
% 
% disp("Center of Mass:");
% disp(get_com_position(model, q));

end