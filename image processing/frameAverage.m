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

    while ~isDone(videoReader)
        % Read the current frame
        frame = step(videoReader);
        
        % Compute the difference between the current frame and the average frame
        differenceFrame = imsubtract(double(frame), bgModel);

        % Normalize the difference frame to the range of 0 to 1
        normalizedFrame = mat2gray(differenceFrame);

        % Write the difference frame to the output video file
        writeVideo(vidfile, normalizedFrame);
    end

    % Close the video reader and writer
    release(videoReader);
    close(vidfile);
end
