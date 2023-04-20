% Create a pop-up window to select one or more files
[files, path] = uigetfile({'*.csv;*.xlsx', 'CSV or Excel Files (*.csv, *.xlsx)'}, 'Select Data File(s)', 'MultiSelect', 'on');

% If the user cancels the file selection, exit the script
if isequal(files,0)
    disp('File selection canceled.')
    return
end

% Initialize max value
max_val = 0;

% Loop over each selected file
for i = 1:length(files)
    % Determine the file type based on the file extension
    if endsWith(files{i}, '.csv')
        % Load data from CSV file
        data = readmatrix(fullfile(path, files{i}));
    elseif endsWith(files{i}, '.xlsx')
        % Load data from XLSX file
        [num, txt, raw] = xlsread(fullfile(path, files{i}));
        data = num;
    else
        % Display an error message if the file type is not supported
        error('Unsupported file type. Please select a CSV or XLSX file.')
    end

    % Extract the two columns of data
    x = data(:, 1);
    y = data(:, 2);

    % Find the maximum value of the current file
    max_val_file = max(y);

    % Update the overall maximum value
    max_val = max(max_val, max_val_file);

    % Normalize the data by the maximum value
    y_norm = y / max_val_file;

    % Plot the normalized data with a different color for each file
    hold on;
    plot(x, y_norm, 'Color', rand(1,3), 'LineWidth', 3);
end

% Add axis labels and a legend
xlabel('X');
ylabel('Y');
legend(files, 'Interpreter', 'none');

% Set the y-axis limit to 1 (the maximum normalized value)
ylim([0 1]);

% Set the x-axis limit to the minimum and maximum values of all the files
xlim([min(x) max(x)]);
