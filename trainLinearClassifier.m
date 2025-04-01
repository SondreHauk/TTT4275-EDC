function W = trainLinearClassifier(N, M, x, t) 
    W = ones(N,M) + rand(N,M);
    alpha = 0.005;

    threshold = 10e-6;
    W_prev = W + 1;
    it = 0;
    while max(abs(W - W_prev), [], 'all') > threshold
        W_prev = W;
        W = W_prev - alpha * gradientofMSE(W_prev, x, t);
        it = it + 1;
    end
    %fprintf("It: %d", it);
end

function gradWMSE = gradientofMSE(W, x, t)
    sigmoid = @(x) 1./(1 + exp(-x));
    g = sigmoid(W * x');
    disp(size(g));
    gradWMSE = ((g - t) .* g .* (1 - g)) * x;
end

% W = [-1.1645    1.5908   -0.3936    0.3653    0.3097;
    %       0.4948    0.5265   -2.1763    2.4124   -0.5420;
    %      -1.2158    0.2648   -1.1957    0.7402    0.1223];