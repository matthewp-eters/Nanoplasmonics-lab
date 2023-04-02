%% paramaters
circle_sensitivity = 0.75;

prominence_horiz = 10; %peaks
prominence_vert = 0.5;
prominence_diag = 0.5;

span_horiz = 0.005;%smoothing
span_vert = 0.025;
span_diag = 0.005;

angle_deg = 45; % angle in degrees (measured from the x-axis)

% Load images
image1 = imread('ps20_frame_0162.jpg');
image2 = imread('ps20_frame_1942.jpg');

% Find centers of circles
[centers1, radii1] = imfindcircles(image1, [20 40], 'Sensitivity', circle_sensitivity);
[centers2, radii2] = imfindcircles(image2, [20 40], 'Sensitivity', circle_sensitivity);

%% Show images with detected circles
figure;
subplot(1,2,1);
imshow(image1);
viscircles(centers1, radii1,'Color','b');
title('Untrapped with Detected Circles');
subplot(1,2,2);
imshow(image2);
viscircles(centers2, radii2,'Color','r');
title('Trapped with Detected Circles');

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


line_pre_horizontal_temp = zeros(1, n); 
line_post_horizontal_temp = zeros(1, n);
for i = 1:n
    line_pre_horizontal_temp(i) = pretrap(center1(2), i);
    line_post_horizontal_temp(i) = posttrap(center1(2), i);
end
line_pre_horizontal = smooth(line_pre_horizontal_temp,span_horiz,'rloess');
line_post_horizontal = smooth(line_post_horizontal_temp,span_horiz,'rloess');


line_pre_vertical_temp = zeros(1, m); 
line_post_vertical_temp = zeros(1, m);
for i = 1:m
    line_pre_vertical_temp(i) = pretrap(center1(1), i);
    line_post_vertical_temp(i) = posttrap(center1(1), i);
end

line_pre_vertical = smooth(line_pre_vertical_temp,span_vert,'rloess');
line_post_vertical = smooth(line_post_vertical_temp,span_vert,'rloess');




%% ANGLE DETECTION

% Compute slope and intercept of line
angle_rad = angle_deg * pi/180; % convert angle to radians
slope = tan(angle_rad);
intercept = center1(2) - slope*center1(1);

% Determine x and y coordinates of all pixels on the line
if abs(slope) <= 1
    % loop over x values
    x = 1:n;
    y = round(slope*x + intercept);
else
    % loop over y values
    y = 1:m;
    x = round((y-intercept)/slope);
end

% Remove points that are outside the matrix
valid_idx = (x>=1) & (x<=n) & (y>=1) & (y<=m);
x = x(valid_idx);
y = y(valid_idx);

% Compute indices of pixels on the line in the matrix
idx = sub2ind(size(pretrap), y, x);

% Compute array of single pixel values along the line
values_horiz = zeros(size(idx));
values_vert = zeros(size(idx));
for i = 1:numel(idx)
    values_horiz(i) = pretrap(idx(i));
    values_vert(i) = posttrap(idx(i));
end

% Average values of multiple pixels corresponding to the same point on the line
unique_idx = unique(idx);
diagonal_pre_temp = zeros(size(unique_idx));
diagonal_post_temp = zeros(size(unique_idx));
for i = 1:numel(unique_idx)
    pixel_idx = unique_idx(i);
    diagonal_pre_temp(i) = mean(values_horiz(idx==pixel_idx));
    diagonal_post_temp(i) = mean(values_vert(idx==pixel_idx));
end

diagonal_pre = smooth(diagonal_pre_temp, span_diag, 'rloess');
diagonal_post = smooth(diagonal_post_temp, span_diag, 'rloess');


% Compute the coordinates of the endpoints of the line
x1 = center1(1) - L * cos(angle_rad);
x2 = center1(1) + L * cos(angle_rad);
y1 = center1(2) - L * sin(angle_rad);
y2 = center1(2) + L * sin(angle_rad);

%% find the local peaks for each image
[pretrap_pks_horizontal,pre_locs_horiz, pre_width_horiz, pre_prom_horiz] = findpeaks(line_pre_horizontal, 'MinPeakProminence',prominence_horiz);
[pretrap_pks_vertical, pre_locs_vert, pre_width_vert, pre_prom_vert] = findpeaks(line_pre_vertical, 'MinPeakProminence',prominence_vert);
[pretrap_pks_diagonal, pre_locs_diagonal, pre_width_diagonal, pre_prom_diagonal] = findpeaks(diagonal_pre, 'MinPeakProminence',prominence_diag);

[posttrap_pks_horizontal, post_locs_horiz, post_width_horiz, post_prom_horiz] = findpeaks(line_post_horizontal, 'MinPeakProminence',prominence_horiz);
[posttrap_pks_vertical, post_locs_vert, post_width_vert, post_prom_vert] = findpeaks(line_post_vertical, 'MinPeakProminence',prominence_vert);
[posttrap_pks_diagonal, post_locs_diagonal, post_width_diagonal, post_prom_diagonal] = findpeaks(diagonal_post, 'MinPeakProminence',prominence_diag);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figures
%% remove untrapped frame from trapped
difference_frame = posttrap - pretrap;

%% plot difference lines
figure;
plot(line_pre_horizontal-line_post_horizontal);
xline(center1(1), 'linewidth', 2, 'color', 'r');
title('Horizontal line difference between untrapped and trapped')

figure;
plot(line_pre_vertical-line_post_vertical);
xline(center1(2), 'linewidth', 2, 'color', 'r');
title('Vertical line difference between untrapped and trapped')

figure;
plot(diagonal_pre-diagonal_post);
xline(center1(2), 'linewidth', 2, 'color', 'r');
title(angle_deg,'degree line difference between untrapped and trapped')


%% Plot pixel intensities horizontal
figure;
%plot(line_pre_horizontal,'b');
findpeaks(line_pre_horizontal, 'MinPeakProminence',prominence_horiz);
hold on;
%plot(line_post_horizontal,'r');
findpeaks(line_post_horizontal, 'MinPeakProminence',prominence_horiz);

plot(line_pre_horizontal_temp, ':')
plot(line_post_horizontal_temp, ':')

xline(center1(1), 'linewidth', 2, 'color', 'r');
title('Horizontal Pixel Intensities Along Line');
xlabel('Pixel Index');
ylabel('Intensity');
legend('Image 1','Image 2');
hold off

%% Plot pixel intensities vertical
figure;
%plot(line_pre_vertical,'b');
findpeaks(line_pre_vertical, 'MinPeakProminence',prominence_vert);
hold on;
%plot(line_post_vertical,'r');
findpeaks(line_post_vertical, 'MinPeakProminence',prominence_vert);

plot(line_pre_vertical_temp, ':')
plot(line_post_vertical_temp, ':')

xline(center1(2), 'linewidth', 2, 'color', 'r');
title('Vertical Pixel Intensities Along Line');
xlabel('Pixel Index');
ylabel('Intensity');
legend('Untrapped','Trapped');
hold off


%% Plot pixel intensities diagonal
figure;
%plot(diagonal_pre,'b');
findpeaks(diagonal_pre, 'MinPeakProminence',prominence_diag);
hold on;
%plot(diagonal_post,'r');
findpeaks(diagonal_post, 'MinPeakProminence',prominence_diag);

plot(diagonal_pre_temp, ':')
plot(diagonal_post_temp, ':')

title(angle_deg,'degree Pixel Intensities Along Line');
xlabel('Pixel Index');
ylabel('Intensity');
legend('Untrapped','Trapped');
hold off

%% Show images with detected circles
figure;
subplot(1,2,1);
imshow(image1);
xline(center1(1), 'linewidth', 2, 'color', 'r');
yline(center1(2), 'linewidth', 2, 'color', 'r');
hold on
plot([x1 x2], [y1 y2], 'linewidth', 2, 'color', 'r')
hold off
title('Untrapped');

subplot(1,2,2);
imshow(image2);
xline(center1(1), 'linewidth', 2, 'color', 'r');
yline(center1(2), 'linewidth', 2, 'color', 'r');
hold on
plot([x1 x2], [y1 y2], 'linewidth', 2, 'color', 'r')
hold off
title('Trapped');

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
xline(center1(1), 'linewidth', 2, 'color', 'r');
yline(center1(2), 'linewidth', 2, 'color', 'r');
title('Background subtraction')


