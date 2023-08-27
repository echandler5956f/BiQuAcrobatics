classdef SRBStates

    properties
        p_body;
        dp_body;
        omega;
        domega;
    end

    methods
        function self = SRBStates(p_body, dp_body, omega, domega)
            self.p_body = p_body;
            self.dp_body = dp_body;
            self.omega = omega;
            self.domega = domega;
        end
    end
end
