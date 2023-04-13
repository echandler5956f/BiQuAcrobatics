classdef KinematicsInterface < handle

    properties (SetAccess = protected, GetAccess = protected)
        fk_fun;
        jacob_fun;
        jacobd_fun;
        options;
        ik_lb;
        ik_ub;
    end

    properties
        dof;
        space;
    end

    methods

        function usefulFKSym = getUsefulFKSym(self, wholeFKSym)
            if self.space == 6 && self.dof == 6
                rx = atan2(wholeFKSym(3,2),wholeFKSym(3,3));
                ry = atan2(-wholeFKSym(3,1), sqrt(wholeFKSym(3,2)^2 + wholeFKSym(3,3)^2));
                rz = atan2(wholeFKSym(2,1), wholeFKSym(1,1));
                usefulFKSym = [wholeFKSym(1:end-1,end); rx; ry; rz];
            else
                if self.space == 6 && self.dof == 3
                    usefulFKSym = wholeFKSym(1: end-1,end);
                else
                    if self.space == 3 && self.dof == 2
                        usefulFKSym = wholeFKSym(1: end-2,end);
                    else
                        fprintf("Unsupported task space and dof combination.\n");
                    end
                end
            end
        end

        function res = fk(self, q)
            r = size(q,1);
            c = size(q,2);

            if isnumeric(q)
                tmp = zeros(r,c);
            else
                tmp = sym(zeros(r,c));
            end
            res = tmp;

            if (r >= 1) && (r <= self.dof)
%                 pfk = cell2mat(self.fk_fun(1,r));
                for i = 1 : c
                    res(:,i) = self.fk_fun(q(:,i));
                end                
            else
                fprintf("Each row of q is a joint variable, with multiple columns representing different joint states.\n")
            end
        end

        function res = jacob(self, q)
            r = size(q,1);
            c = size(q,2);
            if isnumeric(q)
                tmp = zeros(self.dof,self.dof,c);
            else
                tmp = sym(zeros(self.dof,self.dof,c));
            end
            res = tmp;
            if (r == self.dof)
                for i = 1 : c
                    res(:,:,i) = self.jacob_fun(q(:,i));
                end
            else
                fprintf("Each row of q is a joint variable, with multiple columns representing different joint states.\n")
            end
        end

        function res = jacobd(self, q)
            r = size(q,1);
            c = size(q,2);
            if isnumeric(q)
                tmp = zeros(self.dof,self.dof,c);
            else
                tmp = sym(zeros(self.dof,self.dof,c));
            end
            res = tmp;
            if (r == self.dof)
                for i = 1 : c
                    res(:,:,i) = self.jacobd_fun(q(:,i));
                end
            else
                fprintf("Each row of q is a joint variable, with multiple columns representing different joint states.\n")
            end
        end

        function x = ik(self, y, x0)
            [x, ~, ~, ~, ~] = lsqnonlin(@(x)CF(self,x,y), x0, self.ik_lb, self.ik_ub, self.options);
        end
        
        function [F,J] = CF(self, x, y)
            F = y - self.fk(x);
            if nargout > 1
               J = -self.jacob(x);
            end
        end
        
        function jSym = getJacob(self, fkSym, qSym)
            jSym = sym(zeros(self.dof));
            for i = 1 : self.dof
                for j = 1 : self.dof
                    jSym(i, j) = utils.derivative(fkSym(i, :), qSym(j, 1));
                end
            end
            jSym = simplify(expand(jSym),100);
        end

        function jdSym = getJacobDot(self, jSym, qSym)
            jdSym = sym(zeros(self.dof));
            for i = 1 : self.dof
                for j = 1 : self.dof
                    jdSym(i, j) = utils.derivative(jSym(i, j), qSym(j, 1));
                end
            end
            jdSym = simplify(expand(jdSym),100);
        end
    end
end
