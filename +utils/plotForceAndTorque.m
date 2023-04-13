function plotForceAndTorque(time,forces,torques)
    figure('Name', 'Forces and Torques');
    title('Forces and Torques');

    subplot(2,1,1)
    hold on;
    for i = 1 : 3
        plot(time, forces(i,:));
    end
    xlabel('Time (Seconds)');
    ylabel('External Forces');
    title('External Forces vs Time');

    subplot(2,1,2)
    hold on;
    for i = 1 : 3
        plot(time, torques(i,:));
    end
    xlabel('Time (Seconds)');
    ylabel('Input Torques');
    title('Input Torques vs Time');
end