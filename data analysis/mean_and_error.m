% Input data
data = [49.2139 57.4558 53.7345 30.5227;
        58.1298 59.0121 89.1931  NaN;
        72.0612 71.7757 81.4804 NaN;
        35.2170 40.0867     NaN     NaN ];

% Compute mean and standard deviation for each sample
mean_data = nanmean(data, 2); % Compute mean along rows
std_data = nanstd(data, 0, 2); % Compute standard deviation along rows


% Define wavelengths and material
wavelengths = [1000, 800, 300, 0]; % in nm

% Plot the mean and standard deviation for each sample
figure;
% Get the handle of figure(n).
fig1_comps.fig = gcf;

hold on
fig1_comps.p1 = scatter(wavelengths, mean_data, 'filled');
fig1_comps.p2 = errorbar(wavelengths, mean_data, std_data, 'k.');

set(fig1_comps.p1, 'LineWidth', 20, 'Color', 'b');
set(fig1_comps.p2, 'LineWidth', 2, 'Color', 'k');


%========================================================
% ADD LABELS, LEGEND AND SPECIFY SPACING AND PADDING

% Add Global Labels and Title
fig1_comps.plotTitle = title('FWHM of 850nm peak');
fig1_comps.plotXLabel = xlabel('Hole Size (nm)');
fig1_comps.plotYLabel = ylabel('FWHM (nm)');

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
ax.XAxis.Limits = [-10, 1050];
set(gca,'XAxisLocation', 'bottom', 'YAxisLocation', 'left');



%========================================================
% INSTANTLY IMPROVE AESTHETICS

% Set default properties for fign
STANDARDIZE_FIGURE(fig1_comps);
axis square
