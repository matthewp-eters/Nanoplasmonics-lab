function [fitResult, fwhm] = fit_Lorentzian(x, y)
% FIT_LORENTZIAN fits a Lorentzian function to a given wavelength spectrum,
% and returns the fitting results and the FWHM of the peak.

% Set initial conditions
a = 0.5;
b = 49;
c = 0.01;

startPoints = [a, b, c];

% Define model function
lorentz = fittype('a./(b^2 + (x-c).^2)');


% Fit the model to the data
fitResult = fit(x(x>800 & x<900), y(x>800 & x<900), lorentz, 'StartPoint', startPoints);

% Get FWHM of the peak
fwhm = 2*fitResult.b;
end
