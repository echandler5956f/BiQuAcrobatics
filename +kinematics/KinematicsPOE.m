classdef KinematicsPOE < kinematics.KinematicsInterface

    methods
        function self = KinematicsPOE(PM, M, omegaList, pList, qSym, bounds, space, dof)
                self.dof = dof;
                self.space = space;
                self.ik_lb = bounds(:,1);
                self.ik_ub = bounds(:,2);
                self.options = optimoptions('lsqnonlin');
                self.options.StepTolerance = 1e-12;
                self.options.Algorithm = 'levenberg-marquardt';
                self.options.SpecifyObjectiveGradient = true;
                self.options.InitDamping = 1e-10;
                self.options.Display = 'none';
                self.generateKinematics(PM, M, omegaList, pList, qSym);
        end

        function generateKinematics(self, PM, M, omegaList, pList, qSym)
            if ((self.dof ~= size(qSym,1)) || (self.dof ~= size(omegaList,2)))
                fprintf("Number of degrees of freedom has to match the number of rows in qSym, as well as the number of columns in omegaList.\n");
            else
                wholeFKSym = self.getFK(PM, M, omegaList, pList, qSym);
                usefulFKSym = self.getUsefulFKSym(wholeFKSym);
                self.fk_fun = matlabFunction(usefulFKSym,'Vars',{qSym});
                jSym = self.getJacob(usefulFKSym, qSym);
                self.jacob_fun = matlabFunction(jSym,'Vars',{qSym});
                jdSym = self.getJacobDot(jSym, qSym);
                self.jacobd_fun = matlabFunction(jdSym,'Vars',{qSym});
            end
        end

        function fkSym = getFK(self, PM, M, omegaList, pList, qSym)
            fkSym = eye(4);
            for i = self.dof: -1: 1
                fkSym = utils.MatrixExp6(utils.VecTose3(vpa([omegaList(:,i);cross(-omegaList(:,i),pList(:,i))]) * qSym(i))) * fkSym;
            end
            fkSym = simplify(expand(PM * fkSym * M),100);
        end
    end
end
