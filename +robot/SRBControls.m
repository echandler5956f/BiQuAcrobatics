classdef SRBControls

    properties
        F;
        p_feet;
        T;
    end

    methods
        function self = SRBControls(F, p_feet, T)
            self.F = F;
            self.p_feet = p_feet;
            self.T = T;
        end
    end
end
