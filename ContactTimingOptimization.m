clear all; clc; close all;

ocp = ocl.Problem([], 'vars', @varsfun, 'dae', @daefun, ...
'gridconstraints', @consfun, 'terminalcost', @terminalcost, 'N', 40, 'd', 3);

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

% Initial Conditions
R0 = eye(3);
p_body0 = zeros(3,1);
p = kin.fk([deg2rad(-5);deg2rad(5);deg2rad(5)]);
p_feet0 = [legMask(p,1), legMask(p,2), legMask(p,3), legMask(p,4)];

ocp.setInitialBounds('R', R0);
ocp.setInitialBounds('p_body', p_body0);
ocp.setInitialBounds('p_feet', p_feet0);

ocp.setInitialBounds('Omega', Omega0);
ocp.setInitialBounds('DOmega', DOmega0);
ocp.setInitialBounds('dp_body', dp_body0);

% Final Conditions
Rg = eye(3);
p_bodyg = [0;0.03;0];

ocp.setEndBounds('R', Rg);
ocp.setEndBounds('p_body', p_bodyg);
ocp.setEndBounds('p_feet', p_feetg);

ocp.initialize('theta', [0 1], [pi 0]);
ocp.initialize('F', [0 1], [1 1]);

% Run solver to obtain solution
[sol,times] = ocp.solve();

function varsfun(sh)
    sh.addState('R', [3,3]);
    sh.addState('p_body', [3,1]);
    sh.addState('p_feet', [3,4]);
    sh.addState('Omega', [3,1]);
    sh.addState('DOmega', [3,1]);
    sh.addState('dp_body', [3,1]);
        
    sh.addControl('F', [3,4], 'lb', -12, 'ub', 12);
    sh.addControl('Ti', [3,1])
    
    sh.addParameter('r')
    sh.addParameter('Ni', [3,1])
    sh.addParameter('p_feet_bar', [3,1])
end

function consfun(conh, k, K, x, p)
    conh.add(x.R*(x.p_feet(:,1) - x.p_body - p.p_feet_bar(:,1)), '<=', p.r)
    conh.add(x.R*(x.p_feet(:,2) - x.p_body - p.p_feet_bar(:,2)), '<=', p.r)
    conh.add(x.R*(x.p_feet(:,3) - x.p_body - p.p_feet_bar(:,3)), '<=', p.r)
    conh.add(x.R*(x.p_feet(:,4) - x.p_body - p.p_feet_bar(:,4)), '<=', p.r)
end

function daefun(sh,x,~,u,~)
    sh.setODE('R', x.R*approximateExpA(utils.VecToso3(x.Omega)))    
end

function terminalcost(ocl,x,~)
end

function expA = approximateExpA(A, deg)
    expA = A;
    for i = 1 : deg
        expA = expA + (A.^i)/(factorial(i));
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