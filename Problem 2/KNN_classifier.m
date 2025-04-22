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
            
            %E = dist(chunk_training_data, chunk_test_data');
            E = dist(chunk_test_data, chunk_training_data');
            [distance, idx] = mink(E, K, 2);
            
            comparison_matrix = [distance KNN_dist_temp];
            comparison_label = [chunk_training_label(idx), KNN_label_temp];
            [KNN_dist_temp, idx_] = mink(comparison_matrix, K, 2);
            [rowCount, ~] = size(idx_);
            rowIdx = repmat((1:rowCount)', 1, K);

            linearIdx = sub2ind(size(comparison_label), rowIdx, idx_);
            KNN_label_temp = comparison_label(linearIdx);

        end
  
        m = mode(KNN_label_temp, 2); 
        mode_is_tied = false(size(KNN_label_temp, 1), 1);
        
        for r = 1:size(KNN_label_temp, 1)
            label_counts = histcounts(KNN_label_temp(r, :), 'BinMethod', 'integers');
            max_count = max(label_counts);
            num_max = sum(label_counts == max_count);
            if num_max > 1
                mode_is_tied(r) = true;
            end
        end

        % Resolve ties by choosing label of nearest neighbor (first column)
        m(mode_is_tied) = KNN_label_temp(mode_is_tied, 1);

        KNN(chunk_len*i+1:(i+1)*chunk_len) = m;
        KNN_dist_temp =  inf(chunk_len, 1);
        KNN_label_temp = -1 * ones(chunk_len,1);
    end
end