function removeBackground(variance, numFrames, isEnabled)

if isEnabled
    [filename, filepath] = uigetfile('*.avi', 'Select the input video file');
    videoPath = fullfile(filepath, filename);
    
    [~, name, ext] = fileparts(filename);
    outputVideoName = [name, '_removed', ext];
    outputVideoPath = fullfile(filepath, outputVideoName);

    videoSource = VideoReader(videoPath);
    outputVideo = VideoWriter(outputVideoPath, 'Motion JPEG AVI');
    outputVideo.FrameRate = videoSource.FrameRate;
    open(outputVideo);

    % Step 1: Training the background model
    bgModel = trainBackgroundModel(videoSource, numFrames);
    
    % Step 2: Removing the background from every frame and saving the processed video
    videoSource.CurrentTime = 0; % Reset the video source to the beginning
    while hasFrame(videoSource)
        frame = readFrame(videoSource);
        processedFrame = removeBackgroundFromFrame(frame, bgModel, variance);
        writeVideo(outputVideo, processedFrame);
    end

    close(outputVideo);

else
    return
end

function bgModel = trainBackgroundModel(videoSource, numFrames)
    bgModel = [];
    for i = 1:numFrames
        frame = readFrame(videoSource);
        if isempty(bgModel)
            bgModel = double(frame);
        else
            bgModel = bgModel + double(frame);
        end
    end
    bgModel = bgModel / numFrames;
end

function processedFrame = removeBackgroundFromFrame(frame, bgModel, variance)
    diff = abs(double(frame) - bgModel);
    dist = sqrt(sum(diff .^ 2, 3));
    fgMask = dist > variance;
    processedFrame = frame;
    processedFrame(fgMask) = 0;
end
end

