%% Cleaning up Workspace
%clear
close all
clc

%% Init - load data and set variable

data = load('/Users/johnbirgermorud/Documents/Kyb 6 sem/Estimering/Prosjekt/TTT4275-EDC/MNist_ttt4275/data_all.mat');
training_label = data.trainlab;
training_data = data.trainv;
test_label = data.testlab;
test_data = data.testv;

N_te = 10000;
N_tr = 60000;
chunk_len = 1000;

%% Nearest neighboor clasification

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
        nearest_neighbor_temp_label(update_idx) = chunk_training_label(update_idx);
        
    end
    fprintf("[NN, line 40] Temp label\n");
    disp(nearest_neighbor_temp_label);
    disp("")

    nearest_neighbor(i*chunk_len+1:(i+1)*chunk_len) = nearest_neighbor_temp_label;
    nearest_neighbor_temp_dist =  inf(chunk_len, 1);
    nearest_neighbor_temp_label = -1 * ones(chunk_len,1);
end