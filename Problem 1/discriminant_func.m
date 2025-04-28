% trained_1d_discriminants_annotated.m

clear; close all; clc;
rng(4);  % For reproducibility

% 1D, non-overlapping class data
n = 25;
x1 = randn(n,1)*0.4 + 1;   % Class 1 centered at 1
x2 = randn(n,1)*0.4 + 4;   % Class 2 centered at 4

X = [x1; x2];
y = [ones(n,1); 2*ones(n,1)];

% Darker Colors
color1 = [0.9, 0.4, 0];    % Darker Orange (Class 1)
color2 = [0.2, 0.6, 0.9];  % Darker Light Blue (Class 2)

% X-range for plotting discriminant functions
xrange = linspace(min(X)-1, max(X)+1, 200)';

% Create figure
figure; hold on;

% Plot class data points
scatter(x1, zeros(n,1), 30, color1, 'filled', 'DisplayName', 'Class 1');
scatter(x2, zeros(n,1), 30, color2, 'filled', 'DisplayName', 'Class 2');

% Estimate means and shared variance (LDA)
mu1 = mean(x1);
mu2 = mean(x2);
sigma2 = var([x1; x2]);  % shared variance

% Discriminant functions
g1 = (xrange * mu1 - 0.5 * mu1^2) / sigma2;
g2 = (xrange * mu2 - 0.5 * mu2^2) / sigma2;

% Plot discriminant functions
plot(xrange, g1, '-', 'Color', color1, 'LineWidth', 2, 'DisplayName', 'g₁(x)');
plot(xrange, g2, '-', 'Color', color2, 'LineWidth', 2, 'DisplayName', 'g₂(x)');

% Add class text labels
text(mean(x1), 0.4, 'Class 1', 'HorizontalAlignment', 'center', 'Color', color1, 'FontSize', 12);
text(mean(x2), 0.4, 'Class 2', 'HorizontalAlignment', 'center', 'Color', color2, 'FontSize', 12);

% Labels and styling
% title('Trained Discriminant Functions (1D)');
xlabel('x'); ylabel('g_i(x)');
grid on; ylim([-1 5]); xlim([0 5]);

% Add legend
legend('Location', 'northwest', FontSize=13);
