%% Cleaning up Workspace
clear
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
% Run the classifier and save the result. 
% nearest_neighboor = nearest_neighbor_classifier(test_data, training_data, training_label);
% save("classified_images.mat", "nearest_neighboor");

% Load result from file instead of running the function
nearest_neighboor = load("classified_images.mat").nearest_neighbor;


%% Plot misclassified images
test_idx = 1:N_te;
misclassified_idx = test_idx(nearest_neighboor ~= test_label);

N_pictures_to_plot = 5;
for i=1:N_pictures_to_plot
    misclassified_image = zeros(28,28);
    misclassified_image(:) = test_data(misclassified_idx(i), :);
    image(misclassified_image');
    clc;
    fprintf("Picture number: %d\n", i);
    fprintf("Classifier says it is a: %d\n", nearest_neighboor(misclassified_idx(i)));
    fprintf("The real value is a: %d\n", test_label(misclassified_idx(i)));
    pause(8);
end
