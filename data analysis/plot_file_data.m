function [file_data] = plot_file_data()
% Prompt user to select files
[file_names,file_path] = uigetfile({'*.csv;*.txt;*.tdms', 'Supported Files (*.csv,*.txt,*.tdms)'; '*.*', 'All Files (*.*)'}, 'Select files', 'MultiSelect', 'on');

% Check if the user cancelled the dialog
if isequal(file_names,0) || isequal(file_path,0)
    error('File selection cancelled')
end

% Convert file_names to cell array if it is a character array
if ischar(file_names)
    file_names = {file_names};
end

% Initialize file data array
num_files = length(file_names);
file_data = cell(1, num_files);

% Loop through each file
for i = 1:num_files
    % Determine file type based on file extension
    [~,~,ext] = fileparts(file_names{i});
    switch lower(ext)
        case '.csv'
            data = readmatrix(fullfile(file_path, file_names{i}),'NumHeaderLines',7);
            time = data(:,1);
            voltage = data(:,3);
        case '.txt'
            data = readmatrix(fullfile(file_path, file_names{i}),'NumHeaderLines',16);
            voltage = data(:,1);
            time_step = 0.00001;
            time = ((0:length(voltage)-1)*time_step)';
        case '.tdms'
             import TDMExcelAddIn.*
             file = TDMS_readTDMSFile(fullfile(file_path, file_names{i}));
            x_data = file.Data.MeasuredData(1).Data;
            y_data = file.Data.MeasuredData(2).Data;
            time = x_data;
            voltage = y_data;
        otherwise
            error('File type not supported');
    end
    
    % Store file data in array
    file_data{i}.time = time;
    file_data{i}.voltage = voltage;
    file_data{i}.fs = 100000;
    file_data{i}.name = file_names{i};

%     % Plot the data
%     figure
%     plot(time, voltage, "LineWidth",1)
%     hold on;
%     plot(time, lowpass(voltage,1,file_data{i}.fs))
%     xt = get(gca, 'XTick');
%     set(gca, 'XTick',xt, 'XTickLabel',xt/100000)
%     xlabel('Time (s)', 'FontSize',30,'FontWeight','bold')
%     ylabel('Voltage (V)', 'FontSize',30,'FontWeight','bold')
%     ax = gca; 
%     ax.FontSize = 16;
%     title(file_names{i})
%     hold off
end

end
