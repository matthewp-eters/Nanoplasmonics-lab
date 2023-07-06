

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
    outputFilename = [baseFilename '_frame_average'];

    % Create video reader for the current video file
    videoReader = vision.VideoFileReader(videoPath, 'ImageColorSpace', 'Intensity');

    % Create video writer for the current output video file
    vidfile = VideoWriter(outputFilename, 'MPEG-4');
    open(vidfile);

    % Get the frame rate
    frameRate = videoReader.info.VideoFrameRate;
    numFrames = floor(18 * frameRate); % Specify the number of frames for averaging
  
    bgModel = [];
    for j = 1:numFrames
        frame = step(videoReader);
        if isempty(bgModel)
            bgModel = double(frame);
        else
            bgModel = bgModel + double(frame);
        end
    end
    bgModel = bgModel / numFrames;

    % Reset the video reader to the beginning
    reset(videoReader);
bboxes = cell(1,numFrames);
centroids = cell(1,numFrames);
ind = 0;
frameCount = 0;
minBlobArea = 400; % Minimum blob size, in pixels, to be considered as a detection
detectorObjects = setupDetectorObjects(minBlobArea);
    while ~isDone(videoReader)
        % Read the current frame
        frame = step(videoReader);
        
        % Compute the difference between the current frame and the average frame
        differenceFrame = imsubtract(double(frame), bgModel);

        % Normalize the difference frame to the range of 0 to 1
        normalizedFrame = mat2gray(differenceFrame);
        

        frameCount = frameCount + 1; % Increment frame count
    
        % Detect blocs in the video frame
        [centroids{frameCount}, bboxes{frameCount}] = detectBlobs(detectorObjects, normalizedFrame);
    
           
	    % Annotate frame with blobs
        normalizedFrame = insertShape(normalizedFrame,"rectangle",bboxes{frameCount}, ...
            Color="magenta",LineWidth=4);
    
        % Add frame count in the top right corner
        normalizedFrame = insertText(normalizedFrame,[0,0],"Frame: "+int2str(frameCount), ...
            BoxColor="black",TextColor="yellow",BoxOpacity=1);
    
        % Display Video
        step(vidPlayer,normalizedFrame);

        % Write the difference frame to the output video file
        writeVideo(vidfile, normalizedFrame);
    end
    % Store the centroid coordinates for this video
    allCentroids{i} = vertcat(centroids{:});
    % Close the video reader and writer
    release(videoReader);
    close(vidfile);
end

% Plot scatter plot of centroid coordinates
figure;
allCentroids = vertcat(allCentroids{:});
scatter(allCentroids(:, 1), allCentroids(:, 2), 'filled');
axis ij;  % Flip the y-axis to match the image coordinate system
title('Blob CenterCoordinates');
xlabel('X-coordinate');
ylabel('Y-coordinate');
function detectorObjects = setupDetectorObjects(minBlobArea)
% Create System objects for foreground detection and blob analysis

detectorObjects.detector = vision.ForegroundDetector(NumGaussians = 3, ...
    NumTrainingFrames = 40, MinimumBackgroundRatio = 0.7);

detectorObjects.blobAnalyzer = vision.BlobAnalysis(BoundingBoxOutputPort = true, ...
    AreaOutputPort = true, CentroidOutputPort = true, MinimumBlobArea = minBlobArea);
end

function [centroids, bboxes] = detectBlobs(detectorObjects, frame)
% Expected uncertainty (noise) for the blob centroid.

% Detect foreground.
mask = detectorObjects.detector.step(frame);

% Apply morphological operations to remove noise and fill in holes.
mask = imopen(mask, strel(rectangle = [6, 6]));
mask = imclose(mask, strel(rectangle = [50, 50]));
mask = imfill(mask, "holes");

% Perform blob analysis to find connected components.
[~, centroids, bboxes] = detectorObjects.blobAnalyzer.step(mask);
end

