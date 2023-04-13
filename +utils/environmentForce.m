function force = environmentForce(pos,kz,kpz,zlim,posdes)
    force = zeros(3,1);
    if pos(3) < zlim
        force(3,1) = kpz*kz/(kpz + kz)*(posdes(3)-zlim);
    end
end