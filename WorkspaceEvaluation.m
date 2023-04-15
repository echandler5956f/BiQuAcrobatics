clear all; clc; close all;

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

PM = utils.tdh(pi/2, 0, bodyHalfWidth, -pi/2) * utils.tdh(0, -bodyHalfLength, 0, 0);
omegaList = [[0;0;1], [1;0;0], [1;0;0]];
pList = [[0;0;0], [abadLinkLength; 0; 0], [kneeLinkY_offset; hipLinkLength; 0]];

R_home = [0,1,0;
          0,0,-1;
          -1,0,0];
t_home = [abadLinkLength + kneeLinkY_offset; hipLinkLength + kneeLinkLength;0];
M = [R_home, t_home;
     0, 0, 0, 1];

kin = kinematics.KinematicsPOE(PM,M,omegaList,pList,q,bounds,6,3);
kin.fk([deg2rad(-5);deg2rad(5);deg2rad(5)]);

%% Plot
close all;
numSamples = 500000;
pos = zeros(3,numSamples);

for i = 1 : numSamples
    qSym = [(bounds(1,1) + (bounds(1,2) - bounds(1,1)) * rand()); ...
            (bounds(2,1) + (bounds(2,2) - bounds(2,1)) * rand()); ...
            (bounds(3,1) + (bounds(3,2) - bounds(3,1)) * rand())];
    pos(:,i) = kin.fk(qSym);
end 

figure
hold on;
scatter3(pos(1,:),pos(2,:),pos(3,:),'MarkerFaceColor','#EDB120') % front left
scatter3(pos(1,:),-pos(2,:),pos(3,:),'MarkerFaceColor','#4DBEEE') % front right
scatter3(-pos(1,:),pos(2,:),pos(3,:),'MarkerFaceColor','#77AC30') % back left
scatter3(-pos(1,:),-pos(2,:),pos(3,:),'MarkerFaceColor','#A2142F') % back right
