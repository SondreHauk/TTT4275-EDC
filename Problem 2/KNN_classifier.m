function KNN = KNN_classifier( ...
    test_data, ...
    training_data, ...
    training_label, ...
    chunk_len, ...
    K)

    N_tr = size(training_data, 1);
    N_te = size(test_data, 1);
    
    KNN = -1*ones(N_te, 1);
    KNN_dist_temp = inf(chunk_len, K);
    KNN_label_temp = -1 * ones(chunk_len, K);
    

    for i = 0:(N_te/chunk_len - 1)
        chunk_test_data = test_data(chunk_len*i+1 : chunk_len*(i+1), :);
    
        for j = 0:(N_tr/chunk_len - 1)
            chunk_training_data = training_data(chunk_len*j+1 : chunk_len*(j+1), :);
            chunk_training_label = training_label(chunk_len*j+1 : chunk_len*(j+1), :);
            
            % Find K shortest distances of the chunks.
            E = (max(sum(chunk_training_data.^2, 2)' + sum(chunk_test_data.^2, 2) - 2 * (chunk_test_data * chunk_training_data'), 0));

            [distance, idx] = mink(E, K, 2);
            
            %Compare the distances of test chunk with the training chunks.
            comparison_matrix = [distance KNN_dist_temp];
            comparison_label = [chunk_training_label(idx), KNN_label_temp];
            [KNN_dist_temp, idx_] = mink(comparison_matrix, K, 2);
            
            % Extract and store labels where a shorter distance is found
            [rowCount, ~] = size(idx_);
            rowIdx = repmat((1:rowCount)', 1, K);
            linearIdx = sub2ind(size(comparison_label), rowIdx, idx_);
            KNN_label_temp = comparison_label(linearIdx);
            %fprintf("[Inner]\t Finished %d av %d\n", i*N_tr/chunk_len + j, N_te*N_tr/chunk_len^2);
        end
        
        % Label temp now represents the K closest reference points and
        % their labels.
        
        % Finds the majority vote
        m = mode(KNN_label_temp, 2); 

        % Checks for tie in decision
        mode_is_tied = false(size(KNN_label_temp, 1), 1);
        for r = 1:size(KNN_label_temp, 1)
            row = KNN_label_temp(r, :);
            u = unique(row);
            counts = histc(row, u);
            max_count = max(counts);
            if sum(counts == max_count) > 1
                mode_is_tied(r) = true;
            end
        end
        m(mode_is_tied) = KNN_label_temp(mode_is_tied, 1);

        
        % fprintf("[Outer]\t Finished %d av %d\n", i+1, N_te/chunk_len);

        %Store the test chunk in the KNN matrix. 
        KNN(chunk_len*i+1:(i+1)*chunk_len) = m;
        KNN_dist_temp =  inf(chunk_len, K);
        KNN_label_temp = -1 * ones(chunk_len, K);
    end
end