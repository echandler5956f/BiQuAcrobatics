function drawInterpolatedGait()
    figure;
    t = tiledlayout(4,1,'TileSpacing','compact');
    t.Title.String = 'Interpolated Kinematic Phase Diagram $\Psi(\alpha,\beta)$';
    t.Title.Interpreter = 'latex';
    t.Title.FontWeight = 'bold';
    t.Title.FontSize = 48;
    
    t.XLabel.String = 'Kinematic Phase $\Psi$';
    t.XLabel.Interpreter = 'latex';
    t.XLabel.FontSize = 48;
    
    t.YLabel.String = 'Duty Factor $\beta$';
    t.YLabel.Interpreter = 'latex';
    t.YLabel.FontSize = 48;

    size = 4096;
    beta = [0.5, 0.75];
    c = bone(size*4);

    [L1, L2, L3, L4] = utils.getPhaseOffsets(beta);
    t1=nexttile;
    tl1=patch([L1(1,1);L1(1,2);L1(2,2); L1(2,1)], [beta(1);beta(1);beta(2);beta(2)],[0;0;1;1],'EdgeColor','none');
    colormap(t1,c(3*size:4*size,:));
    axis([0 1 0.5 0.75]);
    xticks([]);
    yticks([]);

    t2=nexttile;
    hold on;
    tl2=patch([L2(1,1);L2(2,1);L2(2,2)], [beta(1);beta(2);beta(2)],[0;1;1],'EdgeColor','none');
    patch([L2(1,3);L2(1,4);L2(2,4);L2(2,3)], [beta(1);beta(1);beta(2);beta(2)],[0;0;1;1],'EdgeColor','none');
    colormap(t2,c(2*size:3*size,:));
    axis([0 1 0.5 0.75]);
    xticks([]);
    yticks([]);

    t3=nexttile;
    hold on;
    tl3=patch([L3(1,1);L3(2,1);L3(2,2)], [beta(1);beta(2);beta(2)],[0;1;1],'EdgeColor','none');
    patch([L3(1,3);L3(1,4);L3(2,4);L3(2,3)], [beta(1);beta(1);beta(2);beta(2)],[0;0;1;1],'EdgeColor','none');
    colormap(t3,c(size:2*size,:));
    axis([0 1 0.5 0.75]);
    xticks([]);
    yticks([]);

    t4=nexttile;
    tl4=patch([L4(1,1);L4(1,2);L4(2,2); L4(2,1)], [beta(1);beta(1);beta(2);beta(2)],[0;0;1;1],'EdgeColor','none');
    colormap(t4,c(1:size,:));
    axis([0 1 0.5 0.75]);
    xticks(linspace(0,1,13));
    xtickformat('%.2f');
    yticks(beta);
    lg = legend([tl1(1), tl2(1), tl3(1), tl4(1)], 'Leg 1','Leg 2','Leg 3','Leg 4'); 
    set(lg,'Interpreter','latex');
    set(lg,'FontSize',48);
    lg.Layout.Tile = 'east';
    hold off;
end