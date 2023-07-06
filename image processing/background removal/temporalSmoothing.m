% Allow user to select multiple video files
[filenames, filepath] = uigetfile('*.mp4', 'Select video files', 'MultiSelect', 'on');

% Check if a single file or multiple files are selected
if ~iscell(filenames)
    filenames = {filenames}; % Convert single file selection to cell array
end
vidPlayer = vision.DeployableVideoPlayer;

allCentroids = cell(1, numel(filenames));

for i = 1:numel(filenames)
    % Get the current video file path
    filename = filenames{i};
    videoPath = fullfile(filepath, filename);

    % Create a unique output file name for each video
    [~, baseFilename, ~] = fileparts(filename);
    outputFilename = [baseFilename '_temporal_smooth'];

    % Create video reader for the current video file
    videoReader = vision.VideoFileReader(videoPath, 'ImageColorSpace', 'Intensity');

    % Create video writer for the current output video file
    vidfile = VideoWriter(outputFilename, 'MPEG-4');
    open(vidfile);

% Read the first frame
currentFrame = step(videoReader);
previousFrames = zeros(size(currentFrame), 'like', currentFrame);
for i = 1:4
    previousFrames = previousFrames + currentFrame;
    currentFrame = step(videoReader);
end
previousFrames = previousFrames / 4;

    % Read the first frame
    numFrames = 1;
    previousFrame = step(videoReader);
    bboxes = cell(1,numFrames);
    centroids = cell(1,numFrames);
    ind = 0;
    frameCount = 0;
    minBlobArea = 400; % Minimum blob size, in pixels, to be considered as a detection
    detectorObjects = setupDetectorObjects(minBlobArea);
while ~isDone(videoReader)
    % Compute the difference between the current frame and the previous average frame
    differenceFrame = imsubtract(currentFrame, previousFrames);

    % Normalize the difference frame to the range of 0 to 1
    normalizedFrame = mat2gray(differenceFrame);

    % Apply a Gaussian filter to the normalized frame
    filteredFrame = imgaussfilt(normalizedFrame, 24); % Adjust the standard deviation (sigma) as needed
    filteredFrame = medfilt2(filteredFrame);
    %varFrame = stdfilt(filteredFrame);
    frameCount = frameCount + 1; % Increment frame count

%     [M, N] = size(filteredFrame);
% 
    % % Getting Fourier Transform of the input_image
    % % using MATLAB library function fft2 (2D fast fourier transform)  
    % FT_img = fft2(double(filteredFrame));
    % 
    % D0 = 50; % one can change this value accordingly
    % 
    % % Designing filter
    % u = 0:(M-1);
    % idx = find(u>M/2);
    % u(idx) = u(idx)-M;
    % v = 0:(N-1);
    % idy = find(v>N/2);
    % v(idy) = v(idy)-N;
    % 
    % % MATLAB library function meshgrid(v, u) returns
    % % 2D grid which contains the coordinates of vectors
    % % v and u. Matrix V with each row is a copy 
    % % of v, and matrix U with each column is a copy of u
    % [V, U] = meshgrid(v, u);
    % 
    % % Calculating Euclidean Distance
    % D = sqrt(U.^2+V.^2);
    % 
    % % Comparing with the cut-off frequency and 
    % % determining the filtering mask
    % H = double(D <= D0);
    % 
    % % Convolution between the Fourier Transformed image and the mask
    % G = H.*FT_img;
    % 
    % % Getting the resultant image by Inverse Fourier Transform
    % % of the convoluted image using MATLAB library function
    % % ifft2 (2D inverse fast fourier transform)  
    % output_image = mat2gray(real(ifft2(double(G))));

            % Detect blobs in the video frame
    [centroids{frameCount}, bboxes{frameCount}] = detectBlobs(detectorObjects, normalizedFrame);

    % Annotate frame with blobs
    filteredFrame = insertShape(filteredFrame, "rectangle", bboxes{frameCount}, ...
        'Color', 'magenta', 'LineWidth', 4);

    % Add frame count in the top right corner
    filteredFrame = insertText(filteredFrame, [0,0], ['Frame: ', num2str(frameCount)], ...
        'BoxColor', 'black', 'TextColor', 'yellow', 'BoxOpacity', 1);

    step(vidPlayer, filteredFrame);

    % Write the difference frame to the output video file
    writeVideo(vidfile, filteredFrame);
    
    % Update the previous average frame with the current frame
    previousFrames = previousFrames - previousFrames / 4 + currentFrame / 4;
    
    % Read the next frame
    currentFrame = step(videoReader);
end

% Close the video reader and writer
release(videoReader);
close(vidfile);

    % Store the centroid coordinates for this video
    allCentroids{i} = vertcat(centroids{:});

end

figure;
allCentroids = vertcat(allCentroids{:});
scatter(allCentroids(:, 1), allCentroids(:, 2), 'filled');
axis ij;  % Flip the y-axis to match the image coordinate system
title('Blob CenterCoordinates');
xlabel('X-coordinate');
ylabel('Y-coordinate');

function detectorObjects = setupDetectorObjects(minBlobArea)
% Create System objects for foreground detection and blob analysis

detectorObjects.detector = vision.ForegroundDetector('NumGaussians', 3, ...
    'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);

detectorObjects.blobAnalyzer = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', true, 'CentroidOutputPort', true, 'MinimumBlobArea', minBlobArea);
end

function [centroids, bboxes] = detectBlobs(detectorObjects, frame)
% Detect foreground.
mask = detectorObjects.detector.step(frame);

% Apply morphological operations to remove noise and fill in holes.
mask = imopen(mask, strel('rectangle', [6, 6]));
mask = imclose(mask, strel('rectangle', [50, 50]));
mask = imfill(mask, 'holes');

% Perform blob analysis to find connected components.
[~, centroids, bboxes] = detectorObjects.blobAnalyzer.step(mask);
end
