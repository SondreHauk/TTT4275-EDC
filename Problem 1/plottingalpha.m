% Learning rates to test
alphas = [0.05, 0.01, 0.005];
%colors = ['r-', 'g', 'b']; % Colors for plotting

figure(1)
subplot(2,1,1); hold on; % First subplot for gradient norm
title('Convergence of Gradient Norm');
xlabel('Iterations');
ylabel('Gradient Norm');
grid on;

subplot(2,1,2); hold on; % Second subplot for MSE
title('Convergence of MSE');
xlabel('Iterations');
ylabel('Mean Squared Error');
grid on;

for i = 1:length(alphas)
    [~, gradNorms, mseValues] = plotalpha(C, F+1, x_train, training_set_spes, alphas(i));
    
    % Plot gradient norm
    subplot(2,1,1);
    plot(1:length(gradNorms), gradNorms, 'LineWidth', 2);
    
    % Plot MSE
    subplot(2,1,2);
    plot(1:length(mseValues), mseValues, 'LineWidth', 2);
    
end
legend('Alpha = 0.05', 'Alpha = 0.01', 'Alpha = 0.005');

function [W, gradNorms, mseValues] = plotalpha(N, M, x, t, alpha)
    W = zeros(N, M);
    %threshold = 5e-3;
    %W_prev = W + 1;
    %it = 0;
    
    gradNorms = []; % Store gradient norms
    mseValues = []; % Store MSE values

    %while max(abs(W - W_prev), [], 'all') > threshold
    for i = 0:1999
        W_prev = W;
        grad = gradientofMSE(W_prev, x, t);
        W = W_prev - alpha * grad;
        
        % Compute MSE
        mse = meanSquaredError(W, x, t);
        
        gradNorms = [gradNorms; norm(grad, 'fro')]; % Store norm of gradient
        mseValues = [mseValues; mse]; % Store MSE
        
        %it = it + 1;
    end
    
    fprintf("Alpha: %.3f, Iterations: %d\n", alpha, i);
end

function gradWMSE = gradientofMSE(W, x, t)
    sigmoid = @(x) 1./(1 + exp(-x));
    g = sigmoid(W * x');
    gradWMSE = ((g - t) .* g .* (1 - g)) * x;
end

function mse = meanSquaredError(W, x, t)
    sigmoid = @(x) 1./(1 + exp(-x));
    predictions = sigmoid(W * x');
    predictions = predictions';

    if size(t, 1) == 1
       t = t'; 
    end

    mse = mean((predictions' - t).^2, 'all');
end
