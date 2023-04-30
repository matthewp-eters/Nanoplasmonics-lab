function [fc, Pxx, f] = compute_PSD(voltage, time, fs, start, ending, file_name)

% Extract the portion of the voltage signal within the specified start and end times
trapped = voltage(time > start & time < ending);

% Create a Hann window of the same length as the signal
window = hann(length(trapped));

% Compute the PSD
[Pxx, f] = pwelch(trapped, window, [], [], fs);

% Fit the PSD to a Lorentzian function
Pxx_fit = Pxx(f > 1 & f < 4000);
f_fit = f(f > 1 & f < 4000);
fitfun = fittype('A./(f.^2 +(fc^2))', 'independent', 'f', 'coefficients', {'A', 'fc'});
opts = fitoptions('Method', 'NonlinearLeastSquares', 'Lower', [0, 0], 'Upper', [Inf, Inf], 'StartPoint', [1, 1]);
[fitresult, gof] = fit(f_fit, Pxx_fit, fitfun, opts);

% Extract the fitted corner frequency
fc = fitresult.fc;


PS = PLOT_STANDARDS();

%% Trapping signal
figure;
% Get the handle of figure(n).
fig1_comps.fig = gcf;
fig1_comps.p1 = plot(time, voltage);
%========================================================
% ADD LABELS, LEGEND AND SPECIFY SPACING AND PADDING

% Add Global Labels and Title
fig1_comps.plotTitle = title(file_name);
fig1_comps.plotXLabel = xlabel('Time(s)');
fig1_comps.plotYLabel = ylabel('Voltage(V)');

%========================================================
% ADJUST FONT

set(gca, 'FontName', PS.DefaultFont, 'FontWeight', 'bold');
set([fig1_comps.plotTitle, fig1_comps.plotXLabel, fig1_comps.plotYLabel], 'FontName', PS.DefaultFont);
%set(fig1_comps.plotText, 'FontName', PS.DefaultFont);
set(gca, 'FontSize', PS.AxisNumbersFontSize);
set([fig1_comps.plotXLabel, fig1_comps.plotYLabel], 'FontSize', PS.AxisFontSize);
%set(fig1_comps.plotText, 'FontSize', PS.AxisFontSize);
set(fig1_comps.plotTitle, 'FontSize', PS.TitleFontSize, 'FontWeight' , 'bold');

ax = gca;
ax.YAxis.Limits = [0, 3.1];
ax.XAxis.Limits = [min(time), max(time)];
set(gca,'XAxisLocation', 'bottom', 'YAxisLocation', 'left');

%========================================================
% INSTANTLY IMPROVE AESTHETICS

% Set default properties for fign
STANDARDIZE_FIGURE(fig1_comps);

%set(fig1_comps.p1, 'LineStyle', 'none', 'Marker', 'o', 'MarkerSize', 6, 'MarkerEdgeColor', PS.Blue4, 'MarkerFaceColor', PS.Blue1);
set(fig1_comps.p1, 'Color', PS.Blue4);

axis square


%% PSD
figure;
% Get the handle of figure(n).
fig1_comps.fig = gcf;
fig1_comps.p1 = semilogx(f_fit, 10*log10(Pxx_fit));
hold on
fig1_comps.p2 = semilogx(f_fit, 10*log10(fitresult(f_fit)));
fig1_comps.p3 = plot(fc, 10*log10(fitresult(fc)));
fig1_comps.p4 = line([fc fc], [get(gca, 'YLim')]);
hold off


%========================================================
% ADD LABELS, LEGEND AND SPECIFY SPACING AND PADDING

% Add Global Labels and Title
fig1_comps.plotTitle = title(file_name);
fig1_comps.plotXLabel = xlabel('Frequency (Hz)');
fig1_comps.plotYLabel = ylabel('PSD (dB/Hz)');

%========================================================
% ADJUST FONT

set(gca, 'FontName', PS.DefaultFont, 'FontWeight', 'bold');
set([fig1_comps.plotTitle, fig1_comps.plotXLabel, fig1_comps.plotYLabel], 'FontName', PS.DefaultFont);
%set(fig1_comps.plotText, 'FontName', PS.DefaultFont);
set(gca, 'FontSize', PS.AxisNumbersFontSize);
set([fig1_comps.plotXLabel, fig1_comps.plotYLabel], 'FontSize', PS.AxisFontSize);
%set(fig1_comps.plotText, 'FontSize', PS.AxisFontSize);
set(fig1_comps.plotTitle, 'FontSize', PS.TitleFontSize, 'FontWeight' , 'bold');
ax = gca;
ax.XAxis.Limits = [min(f_fit), max(f_fit)];
set(gca,'XAxisLocation', 'bottom', 'YAxisLocation', 'left');

%text
% set position of the text
xpos = fc*1.25;
ypos = 0.975*10*log10(fitresult(fc));
% here we put 2 backslash \\pi, to espcape the backslash and interpret it
% as for literal character
text_string = sprintf('fc = %.2f Hz', fc);
fig1_comps.plotText = text(xpos, ypos, text_string, 'Interpreter', 'latex', 'Color', PS.MyBlack, 'FontSize', 30);
%========================================================
% INSTANTLY IMPROVE AESTHETICS

% Set default properties for fign
STANDARDIZE_FIGURE(fig1_comps);

%set(fig1_comps.p1, 'LineStyle', 'none', 'Marker', 'o', 'MarkerSize', 6, 'MarkerEdgeColor', PS.Blue4, 'MarkerFaceColor', PS.Blue1);
set(fig1_comps.p1, 'Color', PS.Blue4);
set(fig1_comps.p2, 'Color', PS.Red2);
set(fig1_comps.p3, 'Color', PS.Red3);
set(fig1_comps.p4, 'LineStyle', ':', 'Color', PS.Red3);

axis square

%% Histogram
figure;
% Get the handle of figure(n).
fig1_comps.fig = gcf;
fig1_comps.p1 = histogram(trapped / max(abs(trapped)), 'Normalization', 'pdf');
%========================================================
% ADD LABELS, LEGEND AND SPECIFY SPACING AND PADDING

% Add Global Labels and Title
fig1_comps.plotTitle = title(file_name);
fig1_comps.plotXLabel = xlabel('Signal');
fig1_comps.plotYLabel = ylabel('Counts');

%========================================================
% ADJUST FONT

set(gca, 'FontName', PS.DefaultFont, 'FontWeight', 'bold');
set([fig1_comps.plotTitle, fig1_comps.plotXLabel, fig1_comps.plotYLabel], 'FontName', PS.DefaultFont);
%set(fig1_comps.plotText, 'FontName', PS.DefaultFont);
set(gca, 'FontSize', PS.AxisNumbersFontSize);
set([fig1_comps.plotXLabel, fig1_comps.plotYLabel], 'FontSize', PS.AxisFontSize);
%set(fig1_comps.plotText, 'FontSize', PS.AxisFontSize);
set(fig1_comps.plotTitle, 'FontSize', PS.TitleFontSize, 'FontWeight' , 'bold');


%========================================================
% INSTANTLY IMPROVE AESTHETICS

% Set default properties for fign
STANDARDIZE_FIGURE(fig1_comps);

axis square
%set(fig1_comps.p1, 'LineStyle', 'none', 'Marker', 'o', 'MarkerSize', 6, 'MarkerEdgeColor', PS.Blue4, 'MarkerFaceColor', PS.Blue1);




end
