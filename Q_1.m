clc;
clear;
%run('vlfeat-0.9.18/toolbox/vl_setup')
%% 1.1
fprintf('\nwait few sec Q-1 is running...\n');

x = [1:500]';
alpha = 5;
beta = 5;
gama = 5;

y = [1:500]';
y = y + randn(size(y));

z = (alpha*x + beta*y + gama);
z = z + randn(size(z));

Xcolv = x(:); % Make X a column vector
Ycolv = y(:); % Make Y a column vector
Zcolv = z(:); % Make Z a column vector
Const = ones(size(Xcolv)); % Vector of ones for constant term
Coefficients = [Xcolv Ycolv Const]\Zcolv; % Find the coefficients
XCoeff = Coefficients(1); % X coefficient
YCoeff = Coefficients(2); % X coefficient
CCoeff = Coefficients(3); % constant term
% Using the above variables, z = XCoeff * x + YCoeff * y + CCoeff
L=plot3(x,y,z,'ro'); % Plot the original data points
set(L,'Markersize',3*get(L,'Markersize')) % Making the circle markers larger
set(L,'Markerfacecolor','b') % Filling in the markers
hold on
[xx, yy]=meshgrid(0:100:500,0:100:500); % Generating a regular grid for plotting
zz = XCoeff * xx + YCoeff * yy + CCoeff;
surf(xx,yy,zz) % Plotting the surface
title(sprintf('Plotting plane z=(%f)*x+(%f)*y+(%f)',XCoeff, YCoeff, CCoeff))

%% 1.2
A = [x y ones(size(z))]; % Ax = b
estimates = A\z % x = A\b
%% 1.3
absolute_error = abs([alpha - estimates(1); beta - estimates(2); gama - estimates(3)])


