r = 0.1875;
p_body = [-0.12;0;0.19];
rquat = [-0.0536431, 0, 0.9985602, 0];
R = quat2rotm(rquat);
p_bar = [0.194; 0.1479;0.16];
p_feet_bar = [legMask(p_bar,1),legMask(p_bar,2),legMask(p_bar,3),legMask(p_bar,4)];
p_feet = [[-0.22;0.148;0],[-0.22;-0.148;0],[-0.058;0.14;0.045],[-0.058;-0.154;0.045]];
F_feet = [[5;0;15],[5;0;14],zeros(3,1),zeros(3,1)];
F_grav = [0;0;-9.81*2.5];

% 
% hplots = [];
% 
% for leg = 1 : 4
%     p_sphere = p_body + R * p_feet_bar(:,leg);
%     [x,y,z] = sphere;
% 
%     x = x*r + p_sphere(1);
%     y = y*r + p_sphere(2);
%     z = z*r + p_sphere(3);
%     h = surfl(x,y,z);
%     set(h, 'FaceAlpha', 0.15,'EdgeAlpha', 0, 'FaceLighting','gouraud')
%     hplots = [hplots;h];
% end
% h = plotTransforms(p_body',rquat);
plot3(p_feet(1,:),p_feet(2,:),p_feet(3,:),'.',MarkerSize=100,Color=[0 0.4470 0.7410])
% quiver3([p_feet(1,:),p_body(1)-0.02116582*sin(0.122)],[p_feet(2,:),p_body(2)],[p_feet(3,:)-0.02116582*cos(0.122),p_body(3)+0.05],[F_feet(1,:),F_grav(1)],[F_feet(2,:),F_grav(2)],[F_feet(3,:),F_grav(3)],...
%     'LineWidth',2,'ShowArrowHead','on','Alignment','head',"AutoScaleFactor",1,'MaxHeadSize',0.3,Color=[0.6350 0.0780 0.1840])
% box_bar = [0.175; 0.125];
% v1 = p_body + R * [box_bar(1);box_bar(2);0.05];
% v2 = p_body + R * [box_bar(1);box_bar(2);-0.05];
% v3 = p_body + R * [box_bar(1);-box_bar(2);-0.05];
% v4 = p_body + R * [box_bar(1);-box_bar(2);0.05];
% 
% v5 = p_body + R * [-box_bar(1);box_bar(2);0.05];
% v6 = p_body + R * [-box_bar(1);box_bar(2);-0.05];
% v7 = p_body + R * [-box_bar(1);-box_bar(2);-0.05];
% v8 = p_body + R * [-box_bar(1);-box_bar(2);0.05];
% 
% v = [v1';v2';v3';v4';v5';v6';v7';v8'];
% faces = [1,2,3,4;5,6,7,8;5,1,2,6;7,3,4,8;1,4,8,5;2,3,7,6];
% patch('Vertices',v,'Faces',faces);

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