% Load images
image1 = imread('ps20_frame_0162.jpg');
image2 = imread('ps20_frame_1942.jpg');

% Find centers of circles
[centers1, radii1] = imfindcircles(image1, [20 40], 'Sensitivity', 0.75);
[centers2, radii2] = imfindcircles(image2, [20 40], 'Sensitivity', 0.75);

% Show images with detected circles
figure;
subplot(1,2,1);
imshow(image1);
viscircles(centers1, radii1,'Color','b');
title('Untrapped with Detected Circles');
subplot(1,2,2);
imshow(image2);
viscircles(centers2, radii2,'Color','r');
title('Trapped with Detected Circles');

% Select center of fringes
center1 = round(centers1(1,:));
center2 = round(centers1(1,:));

%convert to grayscale
pretrap = im2gray(image1);
posttrap = im2gray(image2);

%get dimensions of matrix
[m,n] = size(pretrap);

% Compute line through center of fringes
% Create an array of matrix values corresponding to a horizontal line
% passing through the point
span_horiz = 0.01;
span_vert = 0.025;

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


%find the local peaks for each image
prominence_horiz = 10;
prominence_vert = 10;
[pretrap_pks_horizontal,pre_locs_horiz, pre_width_horiz, pre_prom_horiz] = findpeaks(line_pre_horizontal, 'MinPeakProminence',prominence_horiz);
[pretrap_pks_vertical, pre_locs_vert, pre_width_vert, pre_prom_vert] = findpeaks(line_pre_vertical, 'MinPeakProminence',prominence_vert);

[posttrap_pks_horizontal, post_locs_horiz, post_width_horiz, post_prom_horiz] = findpeaks(line_post_horizontal, 'MinPeakProminence',prominence_horiz);
[posttrap_pks_vertical, post_locs_vert, post_width_vert, post_prom_vert] = findpeaks(line_post_vertical, 'MinPeakProminence',prominence_vert);


%remove untrapped frame from trapped
difference_frame = posttrap - pretrap;

%plot difference lines
figure;
plot(line_pre_horizontal-line_post_horizontal);
xline(center1(1), 'linewidth', 2, 'color', 'r');
title('Horizontal line difference between untrapped and trapped')
figure;
plot(line_pre_vertical-line_post_vertical);
xline(center1(2), 'linewidth', 2, 'color', 'r');
title('Vertical line difference between untrapped and trapped')

% Plot pixel intensities
figure;
plot(line_pre_horizontal,'b');
findpeaks(line_pre_horizontal, 'MinPeakProminence',prominence);
hold on;
plot(line_post_horizontal,'r');
findpeaks(line_post_horizontal, 'MinPeakProminence',prominence);
title('Horizontal Pixel Intensities Along Line');
xlabel('Pixel Index');
ylabel('Intensity');
legend('Image 1','Image 2');
hold off

% Plot pixel intensities
figure;
plot(line_pre_vertical,'b');
findpeaks(line_pre_vertical, 'MinPeakProminence',prominence);
hold on;
plot(line_post_vertical,'r');
findpeaks(line_post_vertical, 'MinPeakProminence',prominence);
title('Vertical Pixel Intensities Along Line');
xlabel('Pixel Index');
ylabel('Intensity');
legend('Untrapped','Trapped');
hold off

% % Show images with detected circles
% figure;
% subplot(1,2,1);
% imshow(image1);
% xline(center1(1), 'linewidth', 2, 'color', 'r');
% yline(center1(2), 'linewidth', 2, 'color', 'r');
% title('Untrapped');
% 
% subplot(1,2,2);
% imshow(image2);
% xline(center1(1), 'linewidth', 2, 'color', 'r');
% yline(center1(2), 'linewidth', 2, 'color', 'r');
% title('Trapped');

% figure;
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


