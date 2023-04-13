function [L1, L2, L3, L4] = getPhaseOffsets(beta)
    bRange = [0.5, 0.75];

    Leg1 = [0.000, 0.000; 
            0.500, 0.750];
    
    Leg2 = [0.000, 0.000;
            0.000, 0.250;
            0.500, 0.500;
            1.000, 1.000];
    
    Leg3 = [0.000, 0.000;
            0.000, 0.500;
            0.500, 0.667;
            1.000, 1.000];
    
    Leg4 = [0.000, 0.250; 
            0.500, 1.000];

    L1phaseOffsets1 = 4.*((beta - bRange(1)).*Leg1(1,2) - (beta-bRange(2)).*Leg1(1,1));
    L1phaseOffsets2 = 4.*((beta - bRange(1)).*Leg1(2,2) - (beta-bRange(2)).*Leg1(2,1));
    L1 = [L1phaseOffsets1; L1phaseOffsets2; zeros(1,size(L1phaseOffsets1,2)); zeros(1,size(L1phaseOffsets1,2))]';

    L2phaseOffsets1 = 4.*((beta - bRange(1)).*Leg2(1,2) - (beta-bRange(2)).*Leg2(1,1));
    L2phaseOffsets2 = 4.*((beta - bRange(1)).*Leg2(2,2) - (beta-bRange(2)).*Leg2(2,1));
    L2phaseOffsets3 = 4.*((beta - bRange(1)).*Leg2(3,2) - (beta-bRange(2)).*Leg2(3,1));
    L2phaseOffsets4 = 4.*((beta - bRange(1)).*Leg2(4,2) - (beta-bRange(2)).*Leg2(4,1));
    L2 = [L2phaseOffsets1; L2phaseOffsets2; L2phaseOffsets3; L2phaseOffsets4]';

    L3phaseOffsets1 = 4.*((beta - bRange(1)).*Leg3(1,2) - (beta-bRange(2)).*Leg3(1,1));
    L3phaseOffsets2 = 4.*((beta - bRange(1)).*Leg3(2,2) - (beta-bRange(2)).*Leg3(2,1));
    L3phaseOffsets3 = 4.*((beta - bRange(1)).*Leg3(3,2) - (beta-bRange(2)).*Leg3(3,1));
    L3phaseOffsets4 = 4.*((beta - bRange(1)).*Leg3(4,2) - (beta-bRange(2)).*Leg3(4,1));
    L3 = [L3phaseOffsets1; L3phaseOffsets2; L3phaseOffsets3; L3phaseOffsets4]';

    L4phaseOffsets1 = 4.*((beta - bRange(1)).*Leg4(1,2) - (beta-bRange(2)).*Leg4(1,1));
    L4phaseOffsets2 = 4.*((beta - bRange(1)).*Leg4(2,2) - (beta-bRange(2)).*Leg4(2,1));
    L4 = [L4phaseOffsets1; L4phaseOffsets2; zeros(1,size(L1phaseOffsets1,2)); zeros(1,size(L1phaseOffsets1,2))]';
end