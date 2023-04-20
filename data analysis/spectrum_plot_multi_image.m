% Create a pop-up window to select one or more files
[files, path] = uigetfile({'*.csv;*.xlsx', 'CSV or Excel Files (*.csv, *.xlsx)'}, 'Select Data File(s)', 'MultiSelect', 'on');

% If the user cancels the file selection, exit the script
if isequal(files,0)
    disp('File selection canceled.')
    return
end

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

    % Create a new figure window for each file
    figure;

    % Plot the data and set axis labels and title
    plot(x, y);
    xlabel('Wavelength (nm)');
    ylabel('Counts');
    ylim([0 max(y(i))])
    xlim([min(x(i)) max(x(i))])
    title(files{i}, 'Interpreter', 'none');
end
