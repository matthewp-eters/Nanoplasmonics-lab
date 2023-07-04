[filename, filepath] = uigetfile({'*.avi;*.mp4', 'Video Files (*.avi, *.mp4)'}, 'Select the input video file');
videoPath = fullfile(filepath, filename);

% Generate output file name
[~, baseFilename, ~] = fileparts(filename);
outputFilename = [baseFilename '_removed'];

% Set output path and format
outputFormat = 'mp4';  % Set the desired output format
outputPath = fullfile(filepath, [outputFilename '.' outputFormat]);

vidfile = VideoWriter(outputPath, 'MPEG-4');
open(vidfile);

column_change = 100;                % Specify the column_change value
run = true;                        % Set to true to run the code

if run
    % Load the video
    video_source = VideoReader(videoPath);
    
    % Step 1: Accumulate frames for background generation
    num_background_frames = column_change; % Specify the number of frames to use for background generation
    background_frames = {};
    frame_count = 0;
    
    while frame_count < num_background_frames
        frame = readFrame(video_source);
        if isempty(frame)
            break;
        end
        background_frames{end+1} = frame;
        frame_count = frame_count + 1;
    end
    
    % Step 2: Calculate the average background
    average_background = uint8(mean(cat(4, background_frames{:}), 4));
    
    % Step 3: Apply background subtraction to all frames and convert to black and white
    video_source.CurrentTime = 0; % Reset the video capture to the beginning
    
    % Get video properties
    fps = video_source.FrameRate;
    frame_size = [video_source.Width, video_source.Height];
    
    output_video = VideoWriter(outputPath, 'MPEG-4');
    output_video.FrameRate = fps;
    open(output_video);
    
    while hasFrame(video_source)
        frame = readFrame(video_source);
        
        % Subtract the average background from each frame
        subtracted_frame = frame - average_background;
        
        % Convert the frame and subtracted_frame to black and white (grayscale)
        gray_frame = rgb2gray(frame);
        gray_subtracted_frame = rgb2gray(subtracted_frame);
        
        % Horizontally stack the frames
        hstacked_frames = cat(2, gray_frame, gray_subtracted_frame);
        
        last_frame = gray_subtracted_frame;
        
        imshow(hstacked_frames);
        
        % Write the processed frame to the output video file
        normalized_frame = gray_subtracted_frame / max(gray_subtracted_frame(:));
        writeVideo(output_video, normalized_frame);

        
        if isempty(get(0,'CurrentFigure')) % Check if the figure is closed
            break;
        end
    end
    
    % Release video source and output video
    close(output_video);
    
else
end
