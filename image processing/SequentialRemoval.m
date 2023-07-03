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
    outputFilename = [baseFilename '_sequential_removal'];

    % Create video reader for the current video file
    videoReader = vision.VideoFileReader(videoPath, 'ImageColorSpace', 'Intensity');

    % Create video writer for the current output video file
    vidfile = VideoWriter(outputFilename, 'MPEG-4');
    open(vidfile);

% Read the first frame
previousFrame = step(videoReader);

while ~isDone(videoReader)
    % Read the current frame
    currentFrame = step(videoReader);
    
    % Compute the difference between the current frame and the previous frame
    differenceFrame = imsubtract(currentFrame, previousFrame);
    
    % Normalize the difference frame to the range of 0 to 1
    normalizedFrame = mat2gray(differenceFrame);
    
    % Apply a Gaussian filter to the normalized frame
    filteredFrame = imgaussfilt(normalizedFrame, 2); % Adjust the standard deviation (sigma) as needed
    
    % Write the filtered frame to the output video file
    writeVideo(vidfile, filteredFrame);
    
    % Update the previous frame with the current frame for the next iteration
    previousFrame = currentFrame;
end

% Close the video reader and writer
release(videoReader);
close(vidfile);

end
