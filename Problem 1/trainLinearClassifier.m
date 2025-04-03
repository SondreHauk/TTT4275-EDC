function W = trainLinearClassifier(N, M, x, t) 
    W = zeros(N,M+1);
    alpha = 0.02;

    threshold = 10e-4;
    W_prev = W + 1;
    it = 0;
    while max(abs(W - W_prev), [], 'all') > threshold
        W_prev = W;
        W = W_prev - alpha * gradientofMSE(W_prev, x, t);
        it = it + 1;
    end
    fprintf("It: %d", it);
end

function gradWMSE = gradientofMSE(W, x, t)
    sigmoid = @(x) 1./(1 + exp(-x));
    g = sigmoid(W * x');
    gradWMSE = ((g - t) .* g .* (1 - g)) * x;
end