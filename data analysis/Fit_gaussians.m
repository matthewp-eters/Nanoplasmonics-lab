function [fitResult, fwhm1] = Fit_gaussians(x, y)
% FIT_TWO_GAUSSIANS fits two Gaussians to a given wavelength spectrum and
% returns the fitting results and the FWHM of each Gaussian peak.

% Set initial conditions
a1 = 20;
b1 = 845;
c1 = 0.5;


startPoints = [a1, b1, c1];

% Define model function
gauss2 = fittype('a1*exp(-((x-b1)/c1)^2)');

% Fit the model to the data
fitResult = fit(x(x>800 & x<900), y(x>800 & x<900), gauss2, 'StartPoint', startPoints);

% Get FWHM of each Gaussian
fwhm1 = 2*sqrt(2*log(2))*fitResult.c1;


end
