%% Used for computing derivatives of arbitrarily sized vectors
function dldq = derivative(l,q)
    for i = 1:length(q)
        dldq(:,i) = diff(l,q(i));
    end
    dldq = dldq;
 end