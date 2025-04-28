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
nearest_neighboor = nearest_neighbor_classifier(test_data, training_data, training_label, chunk_len);
save("classified_images.mat", "nearest_neighboor");

% Load result from file instead of running the function
%nearest_neighboor = load("classified_images.mat").nearest_neighbor;


%% Plot misclassified images
%plot_misclassified_images(test_label, test_data, nearest_neighboor, 100);


%% Clustering
M = 64; %Number of clusters for each class
N_classes = 10; 
% Initializing the reduced-size matrices
C = zeros(M*N_classes, N_pixels);
C_label = zeros(M*N_classes, 1);

rng(42); % Set seed for reproducability
for i = 0:9
    % Extracts data of class i
    j = 1:N_tr;
    idx_of_value_i = j(training_label == i);
    K = training_data(idx_of_value_i, :); 

    % Use k-means clustering to reduce data set,
    % and store it into C and C_label.
    [idx, c] = kmeans(K, M, 'Replicates',3);
    C(i*M+1:(i+1)*M, :) = c;
    C_label(i*M+1:(i+1)*M, :) = i;
end

%% Run nearest_neighbor_classifier with clustered data set
chunk_len_clustered = 40;
nearest_neighboor_clustered = nearest_neighbor_classifier(test_data, C, C_label, chunk_len_clustered);

%% K nearest neighbor without clustered data set
K = 7;
KNN_without_clustering = KNN_classifier(test_data, training_data, training_label, chunk_len, K);
error_KNN_without_clustering = sum(KNN_without_clustering == test_label);
error_rate_KNN_wo_clustering = (10000-error_KNN_without_clustering)/10000;
%% K nearest neighbors with clustered data set
chunk_len_clustered = 40;
K = 4;
tic
KNN_with_clustering = KNN_classifier(test_data, C, C_label, chunk_len_clustered, K);
running_time = toc;
error_KNN_with_clustering = (N_te-sum((KNN_without_clustering == test_label)))/N_te;
error_rate_KNN_with_clustering = (10000-error_KNN_with_clustering)/10000;