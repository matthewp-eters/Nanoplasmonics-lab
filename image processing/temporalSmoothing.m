% Allow user to select multiple video files
[filenames, filepath] = uigetfile('*.mp4', 'Select video files', 'MultiSelect', 'on');

% Check if a single file or multiple files are selected
if ~iscell(filenames)
    filenames = {filenames}; % Convert single file selection to cell array
end

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
previousFrames = zeros(size(currentFrame), 'like', currentFrame);
for i = 1:4
    previousFrames = previousFrames + currentFrame;
    currentFrame = step(videoReader);
end
previousFrames = previousFrames / 4;

while ~isDone(videoReader)
    % Compute the difference between the current frame and the previous average frame
    differenceFrame = imsubtract(currentFrame, previousFrames);

    % Normalize the difference frame to the range of 0 to 1
    normalizedFrame = mat2gray(differenceFrame);
    
    % Write the difference frame to the output video file
    writeVideo(vidfile, normalizedFrame);
    
    % Update the previous average frame with the current frame
    previousFrames = previousFrames - previousFrames / 4 + currentFrame / 4;
    
    % Read the next frame
    currentFrame = step(videoReader);
end

% Close the video reader and writer
release(videoReader);
close(vidfile);


end