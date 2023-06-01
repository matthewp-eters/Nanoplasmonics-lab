function estMotionOF(isEnabled)

if isEnabled
%% Set up video reader
[filename, filepath] = uigetfile('*.avi', 'Select the input video file');
videoPath = fullfile(filepath, filename);

videoReader = vision.VideoFileReader(videoPath, 'ImageColorSpace', 'Intensity');

% Generate output file name
[~, baseFilename, ~] = fileparts(filename);
outputFilename = [baseFilename '_tracked'];

vidfile = VideoWriter(outputFilename, 'MPEG-4');
open(vidfile);

% Get frame rate
frameRate = videoReader.info.VideoFrameRate;

% Calculate time interval
timeInterval = 1 / frameRate;
%% Initialize tracking
img = step(videoReader);
figure;
imshow(img)
h = imrect;
wait(h);
rectPos = getPosition(h);

%% Set up optical flow
of = opticalFlowHS;
of.Smoothness = 0.00001;

%% line profiling
% Read the first frame
firstFrame = step(videoReader);

% Update the particle position
center_rectx = rectPos(1) + rectPos(3)/2;
center_recty = rectPos(2) + rectPos(4)/2;
centers1 = [center_rectx center_recty];
center1 = round(centers1(1,:));
    
[height, width, ~] = size(firstFrame); % replace with height and width of first frame
    
% Compute the vertical line
vertical_line = [center1(1)*ones(height, 1), (1:height)'];

% Compute the horizontal line
horizontal_line = [(1:width)', center1(2)*ones(width, 1)];
vertical_profiles = [];
horizontal_profiles = [];

%% Loop algorithm
i = 1;
frames = [];
while(~isDone(videoReader))
    vidFrame = step(videoReader);
    flowField = estimateFlow(of, vidFrame);
      
    % Extract pixel values along the vertical and horizontal lines for the current frame
    vertical_profile = improfile(vidFrame, vertical_line(:, 1), vertical_line(:, 2), 'bicubic');
    horizontal_profile = improfile(vidFrame, horizontal_line(:, 1), horizontal_line(:, 2), 'bicubic');
    
    % Store the profiles in the respective matrices
    vertical_profiles = [vertical_profiles; vertical_profile'];
    horizontal_profiles = [horizontal_profiles; horizontal_profile'];
    
    % Extract the flow vectors for the particle
    indices = zeros(size(vidFrame));
    indices(rectPos(2):rectPos(2)+rectPos(4)-1,rectPos(1):rectPos(1)+rectPos(3)-1) = 1;
    [y,x] = find(indices);
    vx = mean(flowField.Vx(sub2ind(size(flowField.Vx),y,x)));
    vy = mean(flowField.Vy(sub2ind(size(flowField.Vy),y,x)));
    
    % Update the particle position
    x = rectPos(1) + rectPos(3)/2 + vx;
    y = rectPos(2) + rectPos(4)/2 + vy;

    % Store the new particle position in an array
    positions(i,:) = [x y];
    
    subplot(1,2,1)
    plot(flowField, ...
        'DecimationFactor', [10,10],...
        'ScaleFactor', 100);
    title('Optical Flow')
    drawnow;
    subplot(1,2,2)
    imshow(vidFrame)
        
    % plot the positions on the video frame
    hold on
    plot(positions(i,1), positions(i,2), '-o', 'Color', 'b', 'MarkerSize', 2)
    hold off
    
    thisFrame = getframe(gcf);
    writeVideo(vidfile, thisFrame);
    
    frames = [frames, i];
    i = i+1;
end

close(vidfile);
positions = positions(2:end,:);
frames(1) = [];

% Create a column vector of x and y coordinates
xy = positions';

% Fit an ellipse to the trajectory data
fit_result = fit_ellipse(xy(1, :), xy(2, :));

% Extract the ellipse parameters from the fit result
x_center = mean(positions(:, 1));
y_center = mean(positions(:, 2));
a = fit_result.long_axis;
b = fit_result.short_axis;
phi = fit_result.phi;

% Generate the ellipse points
theta = linspace(0, 2*pi, 100);
x_ellipse = x_center + a*cos(theta)*cos(phi) - b*sin(theta)*sin(phi);
y_ellipse = y_center + a*cos(theta)*sin(phi) + b*sin(theta)*cos(phi);

%% MSD
% Calculate displacement between successive frames
displacements = diff(positions(end-149:end, :));

velocities = vecnorm(displacements, 2, 2) * frameRate;

% Calculate squared displacement
sq_displacements = sum(displacements.^2, 2);

% Calculate the MSD for each time lag
max_lag = size(sq_displacements, 1) - 1;
msd = zeros(max_lag, 1);
for lag = 1:max_lag
    msd(lag) = mean(sq_displacements(1:end-lag));
end
diffusion_coefficient = mean(msd) / (6 * timeInterval)

%% Plot the trajectory of the particle
figure
scatter(positions(:,1), positions(:,2))
% Plot the ellipse
hold on;
plot(x_ellipse, y_ellipse, 'r', 'LineWidth', 2);
hold off;
axis equal

figure
plot3(positions(:,1), positions(:,2), frames)



% Plot MSD versus time lag
figure;
plot((1:max_lag)', msd);
xlabel('Time Lag');
ylabel('MSD');

% Rotate the heatmaps
vertical_profiles = transpose(vertical_profiles);
horizontal_profiles = transpose(horizontal_profiles);

% Plot heatmaps for the vertical and horizontal profiles
figure;
imagesc(vertical_profiles);
title('Vertical Line Profiles Heatmap');
xlabel('Frame');
ylabel('Distance');

figure;
imagesc(horizontal_profiles);
title('Horizontal Line Profiles Heatmap');
xlabel('Frame');
ylabel('Distance');



%% Clean up
release(videoReader);

else
    return
end

% diffusion_coefficient = CA011
% 
%     0.4535
% 
% diffusion_coefficient = 20nm011
% 
%     0.0331
% 
% diffusion_coefficient = BSA 121
% 
%     1.5562
% 
% diffusion_coefficient = BSA 013
% 
%     1.6247

