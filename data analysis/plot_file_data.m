%[time, voltage] = plot_file_data();
%function [time, voltage] = plot_file_data()
% This function reads data from a text, csv, or tdms file and plots it

% Prompt user to select a file
[file_name,file_path] = uigetfile({'*.csv;*.txt;*.tdms', 'Supported Files (*.csv,*.txt,*.tdms)'; '*.*', 'All Files (*.*)'}, 'Select a file');

% Check if the user cancelled the dialog
if isequal(file_name,0) || isequal(file_path,0)
    error('File selection cancelled')
end

% Determine file type based on file extension
[~,~,ext] = fileparts(file_name);
switch lower(ext)
    case '.csv'
        data = readmatrix(fullfile(file_path, file_name),'NumHeaderLines',7);
        time = data(:,1);
        voltage = data(:,3);
    case '.txt'
        data = readmatrix(fullfile(file_path, file_name),'NumHeaderLines',16);
        voltage = data(:,1);
        time_step = 0.00001;
        time = (0:length(voltage)-1)*time_step;
    case '.tdms'
         import TDMExcelAddIn.*
         file = TDMS_readTDMSFile(fullfile(file_path, file_name));
        x_data = file.Data.MeasuredData(1).Data;
        y_data = file.Data.MeasuredData(2).Data;
        time = x_data;
        voltage = y_data;
    otherwise
        error('File type not supported');
end

% Plot the data
figure
plot(time, voltage, "LineWidth",1)
xt = get(gca, 'XTick');
set(gca, 'XTick',xt, 'XTickLabel',xt/500000)
xlabel('Time (s)', 'FontSize',30,'FontWeight','bold')
ylabel('Voltage (V)', 'FontSize',30,'FontWeight','bold')
ax = gca; 
ax.FontSize = 16;

%end
