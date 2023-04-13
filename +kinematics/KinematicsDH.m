classdef KinematicsDH < kinematics.KinematicsInterface

    methods
        function self = KinematicsDH(PM, DHTable, qSym, bounds, space, dof)
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
                self.generateKinematics(PM, DHTable, qSym);
        end

        function generateKinematics(self, PM, DHTable, qSym)
            if (self.dof ~= size(qSym,1))
                fprintf("Number of degrees of freedom has to match the number of rows in qSym.\n");
            else
                wholeFKSym = self.getFK(PM, DHTable);
                usefulFKSym = self.getUsefulFKSym(wholeFKSym);
                self.fk_fun = matlabFunction(usefulFKSym,'Vars',{qSym});
                jSym = self.getJacob(usefulFKSym, qSym);
                self.jacob_fun = matlabFunction(jSym,'Vars',{qSym});
                jdSym = self.getJacobDot(jSym, qSym);
                self.jacobd_fun = matlabFunction(jdSym,'Vars',{qSym});
            end
        end

        function fkSym = getFK(self, PM, DHTable)
            fkSym = eye(4);
            for i = 1 : size(DHTable, 1)
                fkSym = fkSym * utils.tdh(DHTable(i, 1), DHTable(i, 2), DHTable(i, 3), DHTable(i, 4));
            end
            fkSym = simplify(expand(PM * fkSym),100);
        end
    end

end
