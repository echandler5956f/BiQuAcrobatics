function trajectoryVisualizer(q_b, q, qd, qdd, qddd, tSamples, wq, wqd, wqdd, wqddd, tPoints)
    figure;
    if isempty(qddd)
        tiledlayout(4,2);
    else
        tiledlayout(5,2);
        nexttile;
        hold on;
        plot(tSamples, qddd(1,:));
        if ~isempty(wqddd)
            plot(tPoints, wqddd(1,:),'ro');
        end
        title('Jerk $x^{f}_{g}$', 'Interpreter','latex')
        xlabel('$t_{swing}$', 'Interpreter','latex');
        ylabel('Jerk $x^{f}_{g}$', 'Interpreter','latex');
        nexttile;
        hold on;
        plot(tSamples, qddd(2,:));
        if ~isempty(wqddd)
            plot(tPoints, wqddd(w,:),'ro');
        end
        title('Jerk $y^{f}_{g}$', 'Interpreter','latex')
        xlabel('$t_{swing}$', 'Interpreter','latex');
        ylabel('Jerk $y^{f}_{g}$', 'Interpreter','latex');
    end
    nexttile;
    hold on;
    plot(tSamples, qdd(1,:));
    if ~isempty(wqdd)
            plot(tPoints, wqdd(1,:),'ro');
    end
    title('$\ddot x^{f}_{g}$', 'Interpreter','latex')
    xlabel('$t_{swing}$', 'Interpreter','latex');
    ylabel('$\ddot x^{f}_{g}$', 'Interpreter','latex');
    nexttile;
    hold on;
    plot(tSamples, qdd(2,:));
    if ~isempty(wqdd)
            plot(tPoints, wqdd(2,:),'ro');
    end
    title('$\ddot y^{f}_{g}$', 'Interpreter','latex')
    xlabel('$t_{swing}$', 'Interpreter','latex');
    ylabel('$\ddot y^{f}_{g}$', 'Interpreter','latex');
    nexttile;
    hold on;
    plot(tSamples, qd(1,:));
    if ~isempty(wqd)
            plot(tPoints, wqd(1,:),'ro');
    end
    title('$\dot x^{f}_{g}$', 'Interpreter','latex')
    xlabel('$t_{swing}$', 'Interpreter','latex');
    ylabel('$\dot x^{f}_{g}$', 'Interpreter','latex');
    nexttile;
    hold on;
    plot(tSamples, qd(2,:));
    if ~isempty(wqd)
            plot(tPoints, wqd(2,:),'ro');
    end
    title('$\dot y^{f}_{g}$', 'Interpreter','latex')
    xlabel('$t_{swing}$', 'Interpreter','latex');
    ylabel('$\dot y^{f}_{g}$', 'Interpreter','latex');
    nexttile;
    hold on;
    plot(tSamples, q(1,:));
    if ~isempty(wq)
            plot(tPoints, wq(1,:),'ro');
    end
    title('$x^{f}_{g}$', 'Interpreter','latex')
    xlabel('$t_{swing}$', 'Interpreter','latex');
    ylabel('$x^{f}_{g}$', 'Interpreter','latex');
    nexttile;
    hold on;
    plot(tSamples, q(2,:));
    if ~isempty(wq)
            plot(tPoints, wq(2,:),'ro');
    end
    title('$y^{f}_{g}$', 'Interpreter','latex')
    xlabel('$t_{swing}$', 'Interpreter','latex');
    ylabel('$y^{f}_{g}$', 'Interpreter','latex');
    nexttile;
    hold on;
    plot(q(1,:), q(2,:));
    if ~isempty(wq)
            plot(wq(1,:), wq(2,:),'ro');
    end
    title('$x^{f}_{g}$ vs $y^{f}_{g}$', 'Interpreter','latex')
    xlabel('$x^{f}_{g}$', 'Interpreter','latex');
    ylabel('$y^{f}_{g}$', 'Interpreter','latex');
    nexttile;
    hold on;
    if ~isempty(q_b)
         plot(q_b(1,:), q_b(2,:))
    end
    title('$x^{f}_{b}$ vs $y^{f}_{b}$', 'Interpreter','latex')
    xlabel('$x^{f}_{b}$', 'Interpreter','latex');
    ylabel('$y^{f}_{b}$', 'Interpreter','latex');
    hold off;
end