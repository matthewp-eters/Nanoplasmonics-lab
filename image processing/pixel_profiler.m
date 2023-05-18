%% paramaters
circle_sensitivity = 0.75;

prominence_horiz = 10; %peaks
prominence_vert = 0.5;
prominence_diag = 0.5;

span_horiz = 0.005;%smoothing
span_vert = 0.025;
span_diag = 0.005;

angle_deg = -45; % angle in degrees (measured from the x-axis)

%% Prompt the user to select two images
[filename1, pathname1] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg, *.png, *.bmp)'}, 'Select image 1');
[filename2, pathname2] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg, *.png, *.bmp)'}, 'Select image 2');

% Check if the user canceled the selection
if isequal(filename1,0) || isequal(filename2,0)
    disp('Image selection canceled.');
    return;
end

% Load the selected images
image1 = imread(fullfile(pathname1, filename1));
image2 = imread(fullfile(pathname2, filename2));

% Use the same image as the reference
reference = image1;


%% Find centers of circles

figure;
imshow(reference)
h = imrect;
wait(h);
rectPos = getPosition(h);

    % Update the particle position
    x = rectPos(1) + rectPos(3)/2;
    y = rectPos(2) + rectPos(4)/2;
    centers1 = [x y];
    radii1 = rectPos(3)-rectPos(1);

x1 = round(rectPos(1));
y1 = round(rectPos(2));
x2 = round(rectPos(1) + rectPos(3));
y2 = round(rectPos(2) + rectPos(4));

averageValues = zeros(2, 1);
% Extract the rectangular region for image 1
rectRegion1 = image1(y1:y2, x1:x2);
averageValues(1) = mean(rectRegion1(:));

% Extract the rectangular region for image 2
rectRegion2 = image2(y1:y2, x1:x2);
averageValues(2) = mean(rectRegion2(:));







%[centers1, radii1] = imfindcircles(reference, [5 20], 'Sensitivity', circle_sensitivity);
%[centers2, radii2] = imfindcircles(image2, [20 40], 'Sensitivity', circle_sensitivity);

% %% Show images with detected circles
% figure;
% subplot(1,2,1);
% imshow(image1);
% viscircles(centers1, radii1,'Color','b');
% title('Untrapped with Detected Circles');
% subplot(1,2,2);
% imshow(image2);
% viscircles(centers1, radii1,'Color','r');
% title('Trapped with Detected Circles');

%% Select center of fringes
center1 = round(centers1(1,:));
center2 = round(centers1(1,:));

%%convert to grayscale
pretrap = im2gray(image1);
posttrap = im2gray(image2);

%%get dimensions of matrix
[m,n] = size(pretrap);

%% Compute line through center of fringes
% Create an array of matrix values corresponding to a horizontal line
% passing through the point
num_pixels = 300;

% Horizontal line through center
center_row = center1(2);
line_pre_horizontal = double(pretrap(center_row, center1(1):(center1(1)+num_pixels-1)));
line_post_horizontal = double(posttrap(center_row, center1(1):(center1(1)+num_pixels-1)));

% Vertical line through center
center_col = center1(1);
line_pre_vertical = double(pretrap(center1(2):(center1(2)+num_pixels-1), center_col))';
line_post_vertical = double(posttrap(center1(2):(center1(2)+num_pixels-1), center_col))';

%smoothing
line_pre_horizontal = smooth(line_pre_horizontal, span_horiz, 'rloess');
line_post_horizontal = smooth(line_post_horizontal, span_horiz, 'rloess');
line_pre_vertical = smooth(line_pre_vertical, span_vert, 'rloess');
line_post_vertical = smooth(line_post_vertical, span_vert, 'rloess');





%% ANGLE DETECTION
% diagonal_pre_matrix = zeros(360, num_pixels);
% diagonal_post_matrix = zeros(360, num_pixels);
% for angle_deg = 0:359
%     % Compute slope and intercept of line
%     angle_rad = angle_deg * pi/180; % convert angle to radians
%     slope = tan(angle_rad);
%     intercept = center1(2) - slope*center1(1);
% 
%     % Determine x and y coordinates of all pixels on the line
%     if abs(slope) <= 1
%         % loop over x values
%         x = center1(1):center1(1)+num_pixels-1;
%         y = round(slope*x + intercept);
%     else
%         % loop over y values
%         y = center1(2):center1(2)+num_pixels-1;
%         x = round((y-intercept)/slope);
%     end
% 
%     % Remove points that are outside the matrix
%     valid_idx = (x>=1) & (x<=n) & (y>=1) & (y<=m);
%     x = x(valid_idx);
%     y = y(valid_idx);
% 
%     % Compute indices of pixels on the line in the matrix
%     idx = sub2ind(size(pretrap), y, x);
% 
%     % Compute array of single pixel values along the line
%     values_horiz = zeros(size(idx));
%     values_vert = zeros(size(idx));
%     for i = 1:numel(idx)
%         values_horiz(i) = pretrap(idx(i));
%         values_vert(i) = posttrap(idx(i));
%     end
% 
%     % Average values of multiple pixels corresponding to the same point on the line
%     unique_idx = unique(idx);
%     diagonal_pre_temp = zeros(1,num_pixels);
%     diagonal_post_temp = zeros(1,num_pixels);
% 
%     for i = 1:numel(unique_idx)
%         pixel_idx = unique_idx(i);
%         diagonal_pre_temp(i) = mean(values_horiz(idx==pixel_idx));
%         diagonal_post_temp(i) = mean(values_vert(idx==pixel_idx));
%     end
% 
%     diagonal_pre = smooth(diagonal_pre_temp, span_diag, 'rloess');
%     diagonal_post = smooth(diagonal_post_temp, span_diag, 'rloess');
% 
%     diagonal_pre_matrix(angle_deg+1,:) = diagonal_pre;
%     diagonal_post_matrix(angle_deg+1,:) = diagonal_post;
% end
% L = size(idx);
% % Compute the coordinates of the endpoints of the line
% x1 = center1(1) - L * cos(angle_rad);
% x2 = center1(1) + L * cos(angle_rad);
% y1 = center1(2) - L * sin(angle_rad);
% y2 = center1(2) + L * sin(angle_rad);

%% find the local peaks for each image
[pretrap_pks_horizontal,pre_locs_horiz, pre_width_horiz, pre_prom_horiz] = findpeaks(line_pre_horizontal, 'MinPeakProminence',prominence_horiz);
[pretrap_pks_vertical, pre_locs_vert, pre_width_vert, pre_prom_vert] = findpeaks(line_pre_vertical, 'MinPeakProminence',prominence_vert);
%[pretrap_pks_diagonal, pre_locs_diagonal, pre_width_diagonal, pre_prom_diagonal] = findpeaks(diagonal_pre, 'MinPeakProminence',prominence_diag);

[posttrap_pks_horizontal, post_locs_horiz, post_width_horiz, post_prom_horiz] = findpeaks(line_post_horizontal, 'MinPeakProminence',prominence_horiz);
[posttrap_pks_vertical, post_locs_vert, post_width_vert, post_prom_vert] = findpeaks(line_post_vertical, 'MinPeakProminence',prominence_vert);
%[posttrap_pks_diagonal, post_locs_diagonal, post_width_diagonal, post_prom_diagonal] = findpeaks(diagonal_post, 'MinPeakProminence',prominence_diag);

horiz_peaks_difference = [];
vert_peaks_difference = [];
horiz_peaks_pre = [];
vert_peaks_pre = [];
horiz_peaks_post = [];
vert_peaks_post = [];
for i = 1:3
    horiz_peaks_difference = [horiz_peaks_difference,pretrap_pks_horizontal(i) - posttrap_pks_horizontal(i)];
    vert_peaks_difference = [vert_peaks_difference,pretrap_pks_vertical(i) - posttrap_pks_vertical(i)];
    horiz_peaks_pre = [horiz_peaks_pre, pretrap_pks_horizontal(i) ];
    vert_peaks_pre = [vert_peaks_pre, pretrap_pks_vertical(i)];
    horiz_peaks_post = [horiz_peaks_post, posttrap_pks_horizontal(i)];
    vert_peaks_post = [vert_peaks_post, posttrap_pks_vertical(i)];
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figures
PS = PLOT_STANDARDS();

%% Peak difference
figure;
% Get the handle of figure(n).
fig1_comps.fig = gcf;
fig1_comps.p1 = scatter([1 2 3],horiz_peaks_difference, 200, "filled");
hold on
fig1_comps.p2 = scatter([1 2 3], vert_peaks_difference, 200, "filled");
hold off

%========================================================
% ADD LABELS, LEGEND AND SPECIFY SPACING AND PADDING
% Add Global Labels and Title
fig1_comps.plotTitle = title('Peak Intensity Differences');
fig1_comps.plotXLabel = xlabel('Peak');
fig1_comps.plotYLabel = ylabel('Intensity Difference');

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
ax.XAxis.Limits = [0.5, 3.5];
xticks([1 2 3])
set(gca,'XAxisLocation', 'bottom', 'YAxisLocation', 'left');
% ADD LEGEND
fig1_comps.plotLegend = legend('Horizontal', 'Vertical','Interpreter', 'none');
% Legend Properties
legendX0 = .7; legendY0 = .08; legendWidth = .1; legendHeight = .1;
set(fig1_comps.plotLegend, 'position', [legendX0, legendY0, legendWidth, ...
    legendHeight], 'Box', 'on');
set(fig1_comps.plotLegend, 'FontSize', PS.LegendFontSize, 'LineWidth', 1.5, ...
    'EdgeColor', PS.Red4);
%========================================================
% INSTANTLY IMPROVE AESTHETICS
% Set default properties for fign
STANDARDIZE_FIGURE(fig1_comps);
set(fig1_comps.p1,'MarkerEdgeColor', PS.DGreen4, 'MarkerFaceColor', PS.DGreen1);
set(fig1_comps.p2,'MarkerEdgeColor', PS.Purple4, 'MarkerFaceColor', PS.Purple1);
axis square





% horizontal pre and post trap peaks
figure;
% Get the handle of figure(n).
fig1_comps.fig = gcf;
fig1_comps.p1 = scatter([1 2 3], horiz_peaks_pre, 200, "filled");
hold on
fig1_comps.p2 = scatter([1 2 3], horiz_peaks_post, 200, "filled");
hold off

%========================================================
% ADD LABELS, LEGEND AND SPECIFY SPACING AND PADDING
% Add Global Labels and Title
fig1_comps.plotTitle = title('Horizontal Peaks');
fig1_comps.plotXLabel = xlabel('Peak');
fig1_comps.plotYLabel = ylabel('Intensity');

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
ax.XAxis.Limits = [0.5, 3.5];
xticks([1 2 3])
set(gca,'XAxisLocation', 'bottom', 'YAxisLocation', 'left');
% ADD LEGEND
fig1_comps.plotLegend = legend('Untrapped', 'Trapped','Interpreter', 'none');
% Legend Properties
legendX0 = .7; legendY0 = .08; legendWidth = .1; legendHeight = .1;
set(fig1_comps.plotLegend, 'position', [legendX0, legendY0, legendWidth, ...
    legendHeight], 'Box', 'on');
set(fig1_comps.plotLegend, 'FontSize', PS.LegendFontSize, 'LineWidth', 1.5, ...
    'EdgeColor', PS.Red4);
%========================================================
% INSTANTLY IMPROVE AESTHETICS
% Set default properties for fign
STANDARDIZE_FIGURE(fig1_comps);
set(fig1_comps.p1,'MarkerEdgeColor', PS.Blue4, 'MarkerFaceColor', PS.Blue1);
set(fig1_comps.p2,'MarkerEdgeColor', PS.Red4, 'MarkerFaceColor', PS.Red1);
axis square


% vertical pre and post trap peaks
figure;
% Get the handle of figure(n).
fig1_comps.fig = gcf;
fig1_comps.p1 = scatter([1 2 3], vert_peaks_pre, 200, "filled");
hold on
fig1_comps.p2 = scatter([1 2 3], vert_peaks_post, 200, "filled");
hold off

%========================================================
% ADD LABELS, LEGEND AND SPECIFY SPACING AND PADDING
% Add Global Labels and Title
fig1_comps.plotTitle = title('Vertical Peaks');
fig1_comps.plotXLabel = xlabel('Peak');
fig1_comps.plotYLabel = ylabel('Intensity');

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
ax.XAxis.Limits = [0.5, 3.5];
xticks([1 2 3])
set(gca,'XAxisLocation', 'bottom', 'YAxisLocation', 'left');
% ADD LEGEND
fig1_comps.plotLegend = legend('Untrapped', 'Trapped','Interpreter', 'none');
% Legend Properties
legendX0 = .7; legendY0 = .08; legendWidth = .1; legendHeight = .1;
set(fig1_comps.plotLegend, 'position', [legendX0, legendY0, legendWidth, ...
    legendHeight], 'Box', 'on');
set(fig1_comps.plotLegend, 'FontSize', PS.LegendFontSize, 'LineWidth', 1.5, ...
    'EdgeColor', PS.Red4);
%========================================================
% INSTANTLY IMPROVE AESTHETICS
% Set default properties for fign
STANDARDIZE_FIGURE(fig1_comps);
set(fig1_comps.p1,'MarkerEdgeColor', PS.Blue4, 'MarkerFaceColor', PS.Blue1);
set(fig1_comps.p2,'MarkerEdgeColor', PS.Red4, 'MarkerFaceColor', PS.Red1);
axis square




%% Center intensity
figure;
% Get the handle of figure(n).
fig1_comps.fig = gcf;
fig1_comps.p1 = bar(1, averageValues(1), 0.75);
hold on
fig1_comps.p2 = bar(2, averageValues(2), 0.75);
xticks([1, 2]);
xticklabels({'Untrapped', 'Trapped'});
%========================================================
% ADD LABELS, LEGEND AND SPECIFY SPACING AND PADDING
% Add Global Labels and Title
fig1_comps.plotTitle = title('Center Intensity');
fig1_comps.plotYLabel = ylabel('Average Pixel Intensity');

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
ax.YAxis.Limits = [0, 260];
set(gca,'XAxisLocation', 'bottom', 'YAxisLocation', 'left');
hold off
%========================================================
% INSTANTLY IMPROVE AESTHETICS
% Set default properties for fign
STANDARDIZE_FIGURE(fig1_comps);
set(fig1_comps.p1, 'EdgeColor', PS.Blue4, 'FaceColor', PS.Blue1);
set(fig1_comps.p2, 'EdgeColor', PS.Red4, 'FaceColor', PS.Red1);
axis square


%% remove untrapped frame from trapped

difference_frame = posttrap - pretrap;

%% plot difference lines
figure;
fig1_comps.fig = gcf;
fig1_comps.p1 = plot(line_pre_horizontal-line_post_horizontal);
%========================================================
% ADD LABELS, LEGEND AND SPECIFY SPACING AND PADDING
% Add Global Labels and Title
fig1_comps.plotTitle = title('Horizontal line difference between untrapped and trapped');
fig1_comps.plotXLabel = xlabel('Pixels From Center');
fig1_comps.plotYLabel = ylabel('Intensity');

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
set(gca,'XAxisLocation', 'bottom', 'YAxisLocation', 'left');
%========================================================
% INSTANTLY IMPROVE AESTHETICS
% Set default properties for fign
STANDARDIZE_FIGURE(fig1_comps);
set(fig1_comps.p1, 'LineWidth', 3, 'Color', PS.Blue4);
axis square



figure;
fig1_comps.fig = gcf;
fig1_comps.p1 = plot(line_pre_vertical-line_post_vertical);
%========================================================
% ADD LABELS, LEGEND AND SPECIFY SPACING AND PADDING
% Add Global Labels and Title
fig1_comps.plotTitle = title('Vertical line difference between untrapped and trapped');
fig1_comps.plotXLabel = xlabel('Pixels From Center');
fig1_comps.plotYLabel = ylabel('Intensity');

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
set(gca,'XAxisLocation', 'bottom', 'YAxisLocation', 'left');
%========================================================
% INSTANTLY IMPROVE AESTHETICS
% Set default properties for fign
STANDARDIZE_FIGURE(fig1_comps);
set(fig1_comps.p1, 'LineWidth', 3, 'Color', PS.Blue4);
axis square

% figure;
% plot(diagonal_pre-diagonal_post);
% title(angle_deg,'degree line difference between untrapped and trapped')





%% Plot pixel intensities horizontal

figure;
fig1_comps.fig = gcf;
fig1_comps.p1 = plot(line_pre_horizontal);
hold on;
fig1_comps.p2 = plot(line_post_horizontal);
hold off
%========================================================
% ADD LABELS, LEGEND AND SPECIFY SPACING AND PADDING
% Add Global Labels and Title
fig1_comps.plotTitle = title('Horizontal Line Pixel Intensities');
fig1_comps.plotXLabel = xlabel('Pixels From Center');
fig1_comps.plotYLabel = ylabel('Intensity');

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
ax.YAxis.Limits = [0, 260];
set(gca,'XAxisLocation', 'bottom', 'YAxisLocation', 'left');
% ADD LEGEND
fig1_comps.plotLegend = legend('Untrapped','Trapped','Interpreter', 'none');
% Legend Properties
legendX0 = .7; legendY0 = .08; legendWidth = .1; legendHeight = .1;
set(fig1_comps.plotLegend, 'position', [legendX0, legendY0, legendWidth, ...
    legendHeight], 'Box', 'on');
set(fig1_comps.plotLegend, 'FontSize', PS.LegendFontSize, 'LineWidth', 1.5, ...
    'EdgeColor', PS.Red4);
%========================================================
% INSTANTLY IMPROVE AESTHETICS
% Set default properties for fign
STANDARDIZE_FIGURE(fig1_comps);
set(fig1_comps.p1, 'LineWidth', 3, 'Color', PS.Blue4);
set(fig1_comps.p2, 'LineWidth', 3, 'Color', PS.Red4);
axis square


figure;
%plot(line_pre_horizontal,'b');
findpeaks(line_pre_horizontal, 'MinPeakProminence',prominence_horiz);
hold on;
%plot(line_post_horizontal,'r');
findpeaks(line_post_horizontal, 'MinPeakProminence',prominence_horiz);

plot(line_pre_horizontal, ':')
plot(line_post_horizontal, ':')

%xline(center1(1), 'linewidth', 2, 'color', 'r');
title('Horizontal Pixel Intensities Along Line');
xlabel('Pixel Index');
ylabel('Intensity');
legend('Untrapped', 'Peak','Trapped','Peak');
hold off

%% Plot pixel intensities vertical
figure;
fig1_comps.fig = gcf;
fig1_comps.p1 = plot(line_pre_vertical);
hold on;
fig1_comps.p2 = plot(line_post_vertical);
hold off
%========================================================
% ADD LABELS, LEGEND AND SPECIFY SPACING AND PADDING
% Add Global Labels and Title
fig1_comps.plotTitle = title('Vertical Line Pixel Intensities');
fig1_comps.plotXLabel = xlabel('Pixels From Center');
fig1_comps.plotYLabel = ylabel('Intensity');

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
ax.YAxis.Limits = [0, 260];

set(gca,'XAxisLocation', 'bottom', 'YAxisLocation', 'left');
% ADD LEGEND
fig1_comps.plotLegend = legend('Untrapped','Trapped','Interpreter', 'none');
% Legend Properties
legendX0 = .7; legendY0 = .08; legendWidth = .1; legendHeight = .1;
set(fig1_comps.plotLegend, 'position', [legendX0, legendY0, legendWidth, ...
    legendHeight], 'Box', 'on');
set(fig1_comps.plotLegend, 'FontSize', PS.LegendFontSize, 'LineWidth', 1.5, ...
    'EdgeColor', PS.Red4);
%========================================================
% INSTANTLY IMPROVE AESTHETICS
% Set default properties for fign
STANDARDIZE_FIGURE(fig1_comps);
set(fig1_comps.p1, 'LineWidth', 3, 'Color', PS.Blue4);
set(fig1_comps.p2, 'LineWidth', 3, 'Color', PS.Red4);
axis square





figure;
%plot(line_pre_vertical,'b');
findpeaks(line_pre_vertical, 'MinPeakProminence',prominence_vert);
hold on;
%plot(line_post_vertical,'r');
findpeaks(line_post_vertical, 'MinPeakProminence',prominence_vert);

plot(line_pre_vertical, ':')
plot(line_post_vertical, ':')

%xline(center1(2), 'linewidth', 2, 'color', 'r');
title('Vertical Pixel Intensities Along Line');
xlabel('Pixel Index');
ylabel('Intensity');
legend('Untrapped', 'Peak','Trapped','Peak');
hold off


%% Plot pixel intensities diagonal
% figure;
% %plot(diagonal_pre,'b');
% findpeaks(diagonal_pre, 'MinPeakProminence',prominence_diag);
% hold on;
% %plot(diagonal_post,'r');
% findpeaks(diagonal_post, 'MinPeakProminence',prominence_diag);
% 
% plot(diagonal_pre_temp, ':')
% plot(diagonal_post_temp, ':')
% 
% title(angle_deg,'degree Pixel Intensities Along Line');
% xlabel('Pixel Index');
% ylabel('Intensity');
% legend('Untrapped','Trapped');
% hold off

%% Show images with detected circles
% figure;
% subplot(1,2,1);
% imshow(image1);
% % xline(center1(1), 'linewidth', 2, 'color', 'r');
% % yline(center1(2), 'linewidth', 2, 'color', 'r');
% line([center1(1) center1(1)+num_pixels], [center1(2) center1(2)], 'Color', 'r', 'LineWidth', 2);
% line([center1(1) center1(1)], [center1(2) center1(2)+num_pixels], 'Color', 'r', 'LineWidth', 2);
% hold on
% plot([x1 x2], [y1 y2], 'linewidth', 2, 'color', 'r')
% hold off
% title('Untrapped');
% 
% subplot(1,2,2);
% imshow(image2);
% %xline(center1(1), 'linewidth', 2, 'color', 'r');
% %yline(center1(2), 'linewidth', 2, 'color', 'r');
% line([center1(1) center1(1)+num_pixels], [center1(2) center1(2)], 'Color', 'r', 'LineWidth', 2);
% line([center1(1) center1(1)], [center1(2) center1(2)+num_pixels], 'Color', 'r', 'LineWidth', 2);
% hold on
% plot([x1 x2], [y1 y2], 'linewidth', 2, 'color', 'r')
% hold off
% title('Trapped');

%% figure;
% %3D intensity plot of untrapped
% A1=image1(:,:,1);A2=image1(:,:,2);A3=image1(:,:,3);
% Y_CCIR601=.299*A1+.587*A2+.114*A3;
% surf(Y_CCIR601,'LineStyle','none');
% shading interp
% colorbar
% 
% figure;
% %3D intensity plot of trapped
% A1=image2(:,:,1);A2=image2(:,:,2);A3=image2(:,:,3);
% Y_CCIR601=.299*A1+.587*A2+.114*A3;
% surf(Y_CCIR601,'LineStyle','none');
% shading interp
% colorbar
% 
% figure;
% %3D intensity plot of difference frame
% subtracted = image2-image1;
% A1=subtracted(:,:,1);A2=subtracted(:,:,2);A3=subtracted(:,:,3);
% Y_CCIR601=.299*A1+.587*A2+.114*A3;
% surf(Y_CCIR601,'LineStyle','none');
% shading interp
% colorbar

figure;
imshow(difference_frame);
line([center1(1) center1(1)+num_pixels], [center1(2) center1(2)], 'Color', 'r', 'LineWidth', 2);
line([center1(1) center1(1)], [center1(2) center1(2)+num_pixels], 'Color', 'r', 'LineWidth', 2);
title('Background subtraction')







% VERIFICATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

% image1_gray = rgb2gray(image1);

% % Set the pixel values to white at the indices in line_pre_horizontal
% for i = 1:numel(line_pre_horizontal)
%     row = center1(2);
%     col = center1(1)+i;
%     image1_gray(row, col) = 255;
% end

% Set the pixel values to white at the indices in line_pre_vertical
% for i = 1:numel(line_pre_vertical)
%     row = center1(1)+i;
%     col = center1(2);
%     image1_gray(row, col) = 255;
% end

% Display the modified grayscale image
% imshow(image1_gray);

% % Create a random nxm matrix
% n = 5;
% m = 6;
% A = rand(n, m);
% 
% % Define the starting point and the angle of the diagonal line
% start_point = [2, 3]; % starting point (row, column)
% angle_degrees = 45; % angle in degrees
% 
% % Convert angle to a direction vector
% angle_radians = angle_degrees * pi / 180; % convert to radians
% direction = [cos(angle_radians), sin(angle_radians)]; % direction vector
% 
% % Calculate the length of the diagonal line
% length = min(n - start_point(1), m - start_point(2));
% 
% % Initialize array for diagonal line values
% diagonal_line = zeros(length, 1);
% 
% % Extract values along the diagonal line
% for i = 1:length
%     row = floor(start_point(1) + i * direction(1));
%     col = floor(start_point(2) + i * direction(2));
%     fprintf('row = %d, col = %d\n', row, col);
%     diagonal_line(i) = A(row, col);
% end
% 
% % Display the extracted diagonal line
% disp(diagonal_line);


