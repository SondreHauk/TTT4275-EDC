function nearest_neighboor = nearest_neighbor_classifier(test_data, training_data, training_label)
    N_tr = size(training_set, 1);
    N_te = size(test_set, 1);

    nearest_neighbor = -1*ones(N_te, 1);
    nearest_neighbor_temp_dist =  inf(chunk_len, 1);
    nearest_neighbor_temp_label = -1 * ones(chunk_len,1);


for i = 0:(N_te/chunk_len - 1)
    chunk_test_data = test_data(chunk_len*i+1 : chunk_len*(i+1), :);

    for j = 0:(N_tr/chunk_len - 1)
        chunk_training_data = training_data(chunk_len*j+1 : chunk_len*(j+1), :);
        chunk_training_label = training_label(chunk_len*j+1 : chunk_len*(j+1), :);
        
        % E = dist(chunk_training_data, chunk_test_data');
        E = dist(chunk_test_data, chunk_training_data');
        [distance, idx] = min(E, [], 2);
        
        update_idx = distance < nearest_neighbor_temp_dist;
        nearest_neighbor_temp_dist(update_idx) = distance(update_idx);        
        nearest_neighbor_temp_label(update_idx) = chunk_training_label(idx(update_idx));
        
    end
    fprintf("[NN, line 40] Temp label\n");
    disp(nearest_neighbor_temp_label);
    disp("")

    nearest_neighbor(chunk_len*i+1:(i+1)*chunk_len) = nearest_neighbor_temp_label;
    nearest_neighbor_temp_dist =  inf(chunk_len, 1);
    nearest_neighbor_temp_label = -1 * ones(chunk_len,1);
end


end