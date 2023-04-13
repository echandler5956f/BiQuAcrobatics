function theta = getNextLegJoints(q,i,offset)
    numSamples = size(q,2);
    t = i + floor(offset*numSamples);
    if t > numSamples
        t = t - numSamples;
    end
        theta = q(:,t);
end