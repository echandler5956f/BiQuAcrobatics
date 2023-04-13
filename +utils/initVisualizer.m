function initVisualizer(robot,qj)
    figure;
    ax = show(robot, qj, "PreservePlot", false,"Frames","on");
    hold on;
    ax.XLabel.String = "X (m)";
    ax.YLabel.String = "Y (m)";
    ax.ZLabel.String = "Z (m)";
    ax.XLim = [-0.5 3];
    ax.YLim = [-2 2];
    ax.ZLim = [-0.1 1];
    view(ax, 135, 25);
    scale = 20;
    % hack to draw a checkerboard with a surface
    [mX, mY] = meshgrid(-0.025*scale:0.025*scale:0.15*scale, -0.125*scale:0.025*scale:0.125*scale);
    J = checkerboard(1, 5, 4) > 0.5;
    J(:,:,2) = J(:,:);
    J(:,:,3) = J(:,:,1);
    J = cast(J, "double");
    checkSurf = surf(ax, mX, mY, 0 * mX, "FaceColor","flat");
    checkSurf.CDataMode = "manual";
    checkSurf.CData = J;
end