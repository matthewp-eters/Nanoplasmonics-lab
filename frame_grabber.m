% Prompt the user to select a video file
[filename, filepath] = uigetfile({'*.mp4;*.avi', 'Video Files (*.mp4,*.avi)'}, 'Select a video file');
if filename == 0
    % User cancelled the file selection dialog
    return;
end
videoFile = fullfile(filepath, filename);

% Open the video file
vidObj = VideoReader(videoFile);

% Get information about the video file
nFrames = vidObj.NumFrames;    % Number of frames in the video
vidHeight = vidObj.Height;     % Height of each frame
vidWidth = vidObj.Width;       % Width of each frame

% Set up a figure to display the video frames
fig = figure;
axesHandle = axes('Parent', fig);

% Create a video player object
playerObj = vision.VideoPlayer;

% Set up a pause button
pauseButton = uicontrol('Style', 'pushbutton', 'String', 'Pause', ...
    'Units', 'normalized', 'Position', [0.9 0.9 0.1 0.1], ...
    'Callback', 'paused = true;');

% Initialize the pause flag
paused = false;

% Set up a counter to keep track of the number of frames saved
numFramesSaved = 0;

% Get the video filename without the extension
[~, videoFilename, ~] = fileparts(videoFile);

% Get the folder where the video is located
videoFolder = fileparts(videoFile);

% Loop through the frames and display them in the figure
for i = 1:nFrames
    % Read the current frame
    frame = read(vidObj, i);
    
    % Display the current frame in the video player object
    step(playerObj, frame);
    
    % Check if the user pressed the pause button
    if paused
        % Construct the filename for the current frame
        filename = fullfile(videoFolder, sprintf('%s_frame_%04d.jpg', videoFilename, i));
        
        % Save the current frame as a JPEG file
        imwrite(frame, filename);
        
        % Increment the counter
        numFramesSaved = numFramesSaved + 1;
        
        % Reset the pause flag
        paused = false;
    end
    
    % Pause the video for a short time
    pause(0.01);
end

% Close the video player object and release the video reader object
release(playerObj);
release(vidObj);
