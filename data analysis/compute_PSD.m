function [fc, Pxx, f] = compute_PSD(voltage, time, fs, start, ending, file_name)

% Extract the portion of the voltage signal within the specified start and end times
trapped = voltage(time > start & time < ending);

% Create a Hann window of the same length as the signal
window = hann(length(trapped));

% Compute the PSD
[Pxx, f] = pwelch(trapped, window, [], [], fs);

% Fit the PSD to a Lorentzian function
Pxx_fit = Pxx(f > 1 & f < 4000);
f_fit = f(f > 1 & f < 4000);
fitfun = fittype('A./(f.^2 +(fc^2))', 'independent', 'f', 'coefficients', {'A', 'fc'});
opts = fitoptions('Method', 'NonlinearLeastSquares', 'Lower', [0, 0], 'Upper', [Inf, Inf], 'StartPoint', [1, 1]);
[fitresult, gof] = fit(f_fit, Pxx_fit, fitfun, opts);

% Extract the fitted corner frequency
fc = fitresult.fc;

% Plot the results
figure;
subplot(1, 3, 1)
semilogx(f_fit, 10*log10(Pxx_fit), 'b');
hold on;
semilogx(f_fit, 10*log10(fitresult(f_fit)), 'r');
fc_text = sprintf('fc = %.2f Hz', fc);
fc_text_pos = [fc*1.25, 0.975*10*log10(fitresult(fc))];
text(fc_text_pos(1), fc_text_pos(2), fc_text, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
line([fc fc], [get(gca, 'YLim')], 'LineStyle', ':', 'Color', 'r');
plot(fc, 10*log10(fitresult(fc)), 'ro', 'MarkerFaceColor', 'r');
xlabel('Frequency (Hz)', 'FontSize',30,'FontWeight','bold');
ylabel('Power spectral density (dB/Hz)', 'FontSize',30,'FontWeight','bold');
title_text = sprintf('%s PSD', file_name);
title(title_text, 'FontSize',30,'FontWeight','bold');
legend('PSD', 'Fitted Lorentzian');
ax = gca; 
ax.FontSize = 16;

subplot(1,3,2)
histogram(trapped / max(abs(trapped)), 'Normalization', 'pdf');
xlabel('Signal', 'FontSize',30,'FontWeight','bold');
ylabel('Counts', 'FontSize',30,'FontWeight','bold');
title_text1 = sprintf('%s Histogram', file_name);
title(title_text1, 'FontSize',30,'FontWeight','bold');
ax = gca; 
ax.FontSize = 16;

subplot(1,3,3)
plot(time, voltage, "LineWidth",1)
hold on; 
plot(time, lowpass(voltage,0.1,fs))
xt = get(gca, 'XTick');
set(gca, 'XTick',xt, 'XTickLabel',xt/100000)
xlabel('Time (s)', 'FontSize',30,'FontWeight','bold')
ylabel('Voltage (V)', 'FontSize',30,'FontWeight','bold')
ax = gca; 
ax.FontSize = 16;
title(file_name, 'FontSize',30,'FontWeight','bold')
hold off
end
