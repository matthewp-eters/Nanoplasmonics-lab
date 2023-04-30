clear all
% Call plot_file_data function to get data
file_info = plot_file_data();

start = 70;
ending = 79;
% Loop through the cell array of structs and compute PSD for each file
for i = 1:numel(file_info)
    % Get data for the i-th file
    data = file_info{i};
    time = data.time;
    voltage = data.voltage;
    fs = data.fs;
    name = data.name;
    % Call compute_PSD function to compute PSD
    compute_PSD(voltage, time, fs, start, ending, name);
end
