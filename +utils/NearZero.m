function judge = NearZero(near)
% *** BASIC HELPER FUNCTIONS ***
% Takes a scalar.
% Checks if the scalar is small enough to be neglected.
% Example Input:
%  
% clear; clc;
% near = -1e-7;
% judge = NearZero(near)
% 
% Output:
% judge =
%     1

if isnumeric(near)
    judge = utils.kinNorm(near) < 1e-6;
else
    judge = false;
end
end
