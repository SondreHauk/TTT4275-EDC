function W = trainLinearClassifier(N, M, x, t)

    W = zeros(N,M+1);
    alpha = 0.005;     % 0.005 chosen as step length
    threshold = 10e-5; % 10e-5 chosen as threshold
    W_prev = W + 1;

    while max(abs(W - W_prev), [], 'all') > threshold
        W_prev = W;
        W = W_prev - alpha * gradientofMSE(W_prev, x, t);
    end
end

function gradWMSE = gradientofMSE(W, x, t)

    sigmoid = @(x) 1./(1 + exp(-x));
    g = sigmoid(W * x');
    gradWMSE = ((g - t) .* g .* (1 - g)) * x;

end