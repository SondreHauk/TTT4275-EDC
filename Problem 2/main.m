%% Cleaning up Workspace
clear
close all
clc

%% Init - load data and set variable

data = load('data_all.mat');
training_label = data.trainlab;
training_data = data.trainv;
test_label = data.testlab;
test_data = data.testv;

N_te = 10000;
N_tr = 60000;
chunk_len = 1000;
N_pixels = 784;

%% Nearest neighboor clasification
% Run the classifier and save the result. 
% nearest_neighboor = nearest_neighbor_classifier(test_data, training_data, training_label, chunk_len);
% save("classified_images.mat", "nearest_neighboor");

% Load result from file instead of running the function
nearest_neighboor = load("classified_images.mat").nearest_neighbor;


%% Plot misclassified images

%plot_misclassified_images(test_label, test_data, nearest_neighboor, 10);


%% Clustering
M = 64; %Number of clusters for each class
N_cluster_vectors = 6000;
N_classes = 10;

%% Sort training data
training_data_sorted = zeros(60000, N_pixels);
training_label_sorted = zeros(60000, 1);

C = zeros(M*N_classes, N_pixels);
C_label = zeros(M*N_classes, 1);

for i = 0:9
    update_idx = (training_label == i);
    j = 1:60000;
    idx_of_value_i = j(update_idx);
    K = training_data(idx_of_value_i, :); % All data of class i

    [idx, c] = kmeans(K, M);
    C(i*M+1:(i+1)*M, :) = c;
    C_label(i*M+1:(i+1)*M, :) = i;
end

%% Run nearest_neighbor_classifier with clustered data set
chunk_len_clustered = 40;
nearest_neighboor_clustered = nearest_neighbor_classifier(test_data, C, C_label, chunk_len_clustered);


%% K nearest neighbors
K = 1;
KNN = KNN_classifier(test_data, C, C_label, chunk_len_clustered, K);

%% Sort values into sorted matrix
% for i = 0:9
%     update_idx = (training_label == i);
%     idx = 1:60000;
%     idx_of_value_i = idx(update_idx);
%     training_data_sorted(start_pos:end_pos, :) ... 
%         = training_data(idx_of_value_i, :);
%     training_label_sorted(start_pos:end_pos, :) ... 
%         = training_label(idx_of_value_i, :);
%     start_pos = end_pos + 1;
%     end_pos = end_pos + sum(training_label == i+1);
% end

