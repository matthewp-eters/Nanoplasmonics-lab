clear all;
[files, path] = uigetfile({'*.csv;*.xlsx', 'CSV or Excel Files (*.csv, *.xlsx)'}, 'Select Data File(s)', 'MultiSelect', 'on');

% If the user cancels the file selection, exit the script
if isequal(files,0)
    disp('File selection canceled.')
    return
end

% If only one file is selected, convert the output to a cell array
if ~iscell(files)
    files = {files};
end

norm_factor = 1;
normarray = ones(length(files{1}),1);
colors = [ "#008000", "#228B22", "#808000", "#556B2F", "#FFA500", "#FF8C00", "#FF6347", "#0000CD", "#00008B", "#000080", "#E6E6FA", "#DDA0DD"];
%colors = [ "#008000", "#228B22", "#808000", "#556B2F", "#E6E6FA","#DDA0DD"]; %1um
%colors = [ "#FFA500", "#FF8C00", "#FF6347", "#E6E6FA", "#DDA0DD"]; %300nm
%colors = ["#0000CD", "#00008B", "#000080", "#E6E6FA", "#DDA0DD"]; %800nm
%colors = ["#E6E6FA", "#DDA0DD"]; %gold
%colors = ["#008000", "#228B22", "#FFA500", "#FF8C00" ]; %clean fibres
%colors = ['r', 'g', 'b', 'k']; %IR

% Ask the user if they want to select a reference file
select_ref_file = questdlg('Do you want to select a reference file to normalize the data?', 'Reference file selection', 'Yes', 'No', 'No');

if strcmp(select_ref_file, 'Yes')
    % Select the reference file
    [ref_file, ref_path] = uigetfile({'*.csv;*.xlsx', 'CSV or Excel Files (*.csv, *.xlsx)'}, 'Select the reference file');
    
    % If the user cancels the file selection, exit the script
    if isequal(ref_file,0)
        disp('Reference file selection canceled.')
        return
    end
    
    % Determine the file type based on the file extension
    if endsWith(ref_file, '.csv')
        ref_data = readmatrix(fullfile(ref_path, ref_file));
    elseif endsWith(ref_file, '.xlsx')
        % Load data from XLSX file
        [num, txt, raw] = xlsread(fullfile(ref_path, ref_file));
        ref_data = num;
    else
        % Display an error message if the file type is not supported
        error('Unsupported file type. Please select a CSV or XLSX file.')
    end
    
    % Extract the two columns of data
    ref_x = ref_data(:, 1);
    ref_y = ref_data(:, 2);
    % Find the maximum value of the reference data
    ref_max = max(ref_y);
    normref = ref_y/max(ref_y);
    
end

figure;
% Get the handle of figure(n).
fig1_comps.fig = gcf;
% Loop over each selected file
for i = 1:length(files)
    % Determine the file type based on the file extension
    if endsWith(files{i}, '.csv')
        opts = detectImportOptions(fullfile(path, files{i})); % detect import options
        opts.DataLines = [17,inf]; % skip the first 16 lines
        data = readmatrix(fullfile(path, files{i}), opts);
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

    normY = y/max(y);
        
     if strcmp(select_ref_file, 'Yes')
         DivRefY = y./ref_y;
        %DivRefY = normY ./ normref;
        DivRefNorm = DivRefY/max(DivRefY(ref_x >700 & ref_x <1000));
        %DivRefNorm = DivRefY/max(DivRefY);

     else
        DivRefNorm = normY;
     end

   [fitResult, fwhm1] =  fit_Lorentzian(x, DivRefNorm);
     fwhm1
    PS = PLOT_STANDARDS();
    %% Spectrum
    hold on;
    fig1_comps.p1(i) = plot(x, smooth(DivRefNorm));
    fig1_comps.p2(i) = plot(fitResult);
    
    %set(fig1_comps.p1, 'LineStyle', 'none', 'Marker', 'o', 'MarkerSize', 6, 'MarkerEdgeColor', PS.Blue4, 'MarkerFaceColor', PS.Blue1);
    set(fig1_comps.p1(i), 'LineWidth', 3, 'Color', colors(i));
    set(fig1_comps.p2(i), 'LineWidth', 3, 'Color', colors(i));

    

end
hold off;
% ADD LEGEND
fig1_comps.plotLegend = legend(files,'Interpreter', 'none');
% Legend Properties
legendX0 = .7; legendY0 = .08; legendWidth = .1; legendHeight = .1;
set(fig1_comps.plotLegend, 'position', [legendX0, legendY0, legendWidth, ...
    legendHeight], 'Box', 'on');
set(fig1_comps.plotLegend, 'FontSize', PS.LegendFontSize, 'LineWidth', 1.5, ...
    'EdgeColor', PS.Red4);
%========================================================
% ADD LABELS, LEGEND AND SPECIFY SPACING AND PADDING

% Add Global Labels and Title
fig1_comps.plotTitle = title('DNH on Fibre');
fig1_comps.plotXLabel = xlabel('Wavelength (nm)');
fig1_comps.plotYLabel = ylabel('Counts');

%========================================================
% ADJUST FONT

set(gca, 'FontName', PS.DefaultFont, 'FontWeight', 'bold');
set([fig1_comps.plotTitle, fig1_comps.plotXLabel, fig1_comps.plotYLabel], 'FontName', PS.DefaultFont);
%set(fig1_comps.plotText, 'FontName', PS.DefaultFont);
set(gca, 'FontSize', PS.AxisNumbersFontSize);
set([fig1_comps.plotXLabel, fig1_comps.plotYLabel], 'FontSize', PS.AxisFontSize);
%set(fig1_comps.plotText, 'FontSize', PS.AxisFontSize);
set(fig1_comps.plotTitle, 'FontSize', PS.TitleFontSize, 'FontWeight' , 'bold');

ax = gca;
ax.YAxis.Limits = [0, 1];
ax.XAxis.Limits = [700, 1000];
set(gca,'XAxisLocation', 'bottom', 'YAxisLocation', 'left');



%========================================================
% INSTANTLY IMPROVE AESTHETICS

% Set default properties for fign
STANDARDIZE_FIGURE(fig1_comps);
axis square

