clear all;
% solve harmonic oscillator using ode45
% define the t vector 0->5 (100 points)
t = linspace(0, 10, 100);

% this is the harmonic defined for the harmonic oscillator
function g_dot = harmonicFunc(t, g, omega, gma)
        g_dot(1) = g(2); % v
        g_dot(2) = - 2*gma*g(2) - (omega^2)*g(1); % dv/dt
endfunction

% g is a  vector of v and dv/dt; where v = dx/dy.
omega = 2; % constant
filename = 'harmonic-oscillator-octave.pdf'
fopen(filename, 'w'); % drop the contents of the file.
for gma = [0.1, 2, 4]
        f = @(t, g)harmonicFunc(t, g, omega, gma); % defining a annon func.
        [_, x_v] = ode45(f, t, [2, 1]);
        figure('visible', 'off');

        subplot (3, 1, 1)
        plot(t, x_v(:, 1));
        title(strcat('Time(t) vs Position(x) [ \gamma=', num2str(gma), ', \omega=', num2str(omega),']'))
        grid();
        xlabel('Time(t) \rightarrow'); ylabel('Position(x) \rightarrow');

        subplot (3, 1, 2)
        plot(t, x_v(:, 2));
        title(strcat('Time(t) vs Velocity(v) [ \gamma=', num2str(gma), ', \omega=', num2str(omega),']'))
        grid();
        xlabel('Time(t) \rightarrow'); ylabel('Velocity(v) \rightarrow');

        subplot (3, 1, 3)
        plot(x_v(:, 1), x_v(:, 2));
        title(strcat('Position(x) vs Velocity(v) [ \gamma=', num2str(gma), ', \omega=', num2str(omega),']'))
        grid();
        xlabel('Position(x) \rightarrow'); ylabel('Velocity(v) \rightarrow');

        print (filename, '-append','-dpdf','-S595,842');
        close all;
        disp(strcat('Plotted For: [gamma=', num2str(gma), ',omega=', num2str(omega),']'))
endfor
