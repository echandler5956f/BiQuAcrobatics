clear all; clc; close all;
Ib = [5, 0, 0;
      0, 2, 0;
      0, 0, 3];
R = [0.7071068, -0.7071068,  0;
     0.7071068,  0.7071068,  0;
     0,  0,  1];
syms Omega tau [3 1]
A = Ib;
b = inv(R)*tau - cross(Omega, Ib*Omega);
DOmega1 = A\b;
% disp(DOmega1);
DOmega2 = Ib\(inv(R)*tau - cross(Omega, Ib*Omega));
% disp(DOmega2);
disp(simplify(DOmega1-DOmega2));