%% Clear Matlab Workspace & clean
close all;
clear;
clc;

%% ------------ Task 1 ------------ %%
load fisheriris;

N_da =          50;   % Length of data set
N_tr =          30;   % Length of trainings set
N_te = N_da - N_tr;   % Length of test set

C = 3; % number of classes
F = 4; % number of features

training_set_feat = [
    meas(1:N_tr, :);
    meas(N_da+1:N_da + N_tr, :);
    meas(2*N_da+1:2*N_da+N_tr, :)];

training_set_label = [
    species(1:N_tr); 
    species(N_da+1:N_da + N_tr); 
    species(2*N_da+1:2*N_da+N_tr)];

test_set_feat = [
    meas(N_tr+1:N_da, :); 
    meas(N_da+N_tr+1:2*N_da, :); 
    meas(2*N_da+N_tr+1:3*N_da, :)];

test_set_label = [
    species(N_tr+1:N_da); 
    species(N_da+N_tr+1:2*N_da); 
    species(2*N_da+N_tr+1:3*N_da)];

% Define Setosa, Versicolor and Virinica as
Se = [1 0 0]';
Ve = [0 1 0]';
Vi = [0 0 1]';

%% Create training and test sets
training_set_spes = [
    kron(ones(1, N_tr), Se), ...
    kron(ones(1, N_tr), Ve), ...
    kron(ones(1, N_tr), Vi)];

x_train = [training_set_feat, ones(size(training_set_feat,1),1)];
x_test = [test_set_feat, ones(size(test_set_feat,1),1)];

%% Train Classifier
W = trainLinearClassifier(C, F, x_train, training_set_spes);

% save('W.mat', 'W')
% W = load("W.mat").W;

%% Predicted label for training and test set
sigmoid = @(x) 1./(1 + exp(-x));

[~, pred_labels_train] = max(sigmoid(W * x_train'), [], 1);
[~, pred_labels_test] = max(sigmoid(W * x_test'), [], 1);

%% Confusion matrixes for test and training set
class_labels = {'Setosa', 'Versicolor', 'Virginica'};

true_labels_train = [
    1 * ones(1,N_tr), ...
    2 * ones(1,N_tr), ...
    3 * ones(1,N_tr)];

true_labels_test = [
    1 * ones(1,N_te), ...
    2 * ones(1,N_te), ...
    3 * ones(1,N_te)];

true_labels_train = categorical(class_labels(true_labels_train));
pred_labels_train = categorical(class_labels(pred_labels_train));

true_labels_test = categorical(class_labels(true_labels_test));
pred_labels_test = categorical(class_labels(pred_labels_test));

cm_train = confusionchart(true_labels_train, pred_labels_train, ...
    "Title","Training set");

%cm_test = confusionchart(true_labels_test, pred_labels_test, ...
%  "Title","Test set");

%% ------------ Task 2 ------------ %%
p1_c1 = meas(1:N_da,1);
p1_c2 = meas(N_da+1:2*N_da,1);
p1_c3 = meas(2*N_da+1:3*N_da,1);

p2_c1 = meas(1:N_da,2);
p2_c2 = meas(N_da+1:2*N_da,2);
p2_c3 = meas(2*N_da+1:3*N_da,2);

p3_c1 = meas(1:N_da,3);
p3_c2 = meas(N_da+1:2*N_da,3);
p3_c3 = meas(2*N_da:3*N_da,3);

p4_c1 = meas(1:N_da,4);
p4_c2 = meas(N_da+1:2*N_da,4);
p4_c3 = meas(2*N_da+1:3*N_da,4);

%% Plot histograms
B = 25;
figure;
% Sepal length
subplot(2,2,1);
histogram(p1_c1, B); hold on; histogram(p1_c2, B); hold on; histogram(p1_c3, B);
xlabel('cm'); ylabel('frequency'); subtitle('Sepal length', 'FontSize',14);
xlim([0 8]);

% Sepal width
subplot(2,2,2);
histogram(p2_c1, B); hold on; histogram(p2_c2, B); hold on; histogram(p2_c3, B);
xlabel('cm'); ylabel('frequency'); subtitle('Sepal width', 'FontSize',14);
xlim([0 8]);

% Petal length
subplot(2,2,3); 
histogram(p3_c1, B); hold on; histogram(p3_c2, B); hold on; histogram(p3_c3, B);
xlabel('cm'); ylabel('frequency'); subtitle('Petal length', 'FontSize',14);
xlim([0 8]);

% Petal width
subplot(2,2,4);
histogram(p4_c1, 15); hold on; histogram(p4_c2, 15); hold on; histogram(p4_c3, 15);
xlabel('cm'); ylabel('frequency'); subtitle('Petal width', 'FontSize',14);
xlim([0 8]); % Set x-axis limits from 0 to 8 cm
legend('Setosa', 'Versicolor', 'Virginica', 'FontSize',14);

%% Train classifier with 3 features
% By inspection: The sepal width (feature 2) is the feature with most overlap between
% the classes and is therefore removed for the following training.

training_set_meas_3 = [training_set_feat(:,1) training_set_feat(:, 3:4)]; % Remove feature 2
test_set_meas_3 = [test_set_feat(:,1) test_set_feat(:, 3:4)]; % Remove feature 2

F = 3; % features reduced to three

% Training set
x_train_3 = [training_set_meas_3 ones(size(training_set_meas_3,1),1)];
x_test_3 = [test_set_meas_3 ones(size(test_set_meas_3,1),1)];

W = trainLinearClassifier(C, F, x_train_3, training_set_spes);

sigmoid = @(x) 1./(1 + exp(-x));
[~, pred_labels_train] = max(sigmoid(W * x_train_3'), [], 1);
[~, pred_labels_test] = max(sigmoid(W * x_test_3'), [], 1);

pred_labels_train = categorical(class_labels(pred_labels_train));
pred_labels_test = categorical(class_labels(pred_labels_test));

cm_train = confusionchart(true_labels_train, pred_labels_train, ...
    "Title","Training set");

%cm_test = confusionchart(true_labels_test, pred_labels_test, ...
%    "Title", "Test set");


%% Train classifier with 2 features
% By inspection: The sepal length (feature 1) is the next feature with overlap between
% the classes and is therefore be removed for the following training.

training_set_meas_2 = [training_set_meas_3(:, 2:3)]; % Remove feature 1 and 2
test_set_meas_2 = [test_set_meas_3(:, 2:3)]; % Remove feature 1 and 2

F = 2; % features reduced to two

x_train_2 = [training_set_meas_2, ones(size(training_set_meas_2,1),1)];
x_test_2 = [test_set_meas_2, ones(size(test_set_meas_2,1),1)];

W = trainLinearClassifier(C, F, x_train_2, training_set_spes);

[~, pred_labels_train] = max(sigmoid(W * x_train_2'), [], 1);
[~, pred_labels_test] = max(sigmoid(W * x_test_2'), [], 1);

pred_labels_train = categorical(class_labels(pred_labels_train));
pred_labels_test = categorical(class_labels(pred_labels_test));

cm_train = confusionchart(true_labels_train, pred_labels_train, ...
    "Title", "Training set");

cm_test = confusionchart(true_labels_test, pred_labels_test, ...
   "Title", "Test set");

%% Train classifier with 1 feature
% By inspection: The petal width (feature 4) is the next feature with most overlap between
% the classes and is therefore be removed for the following training. The
% only remaining feature is now the petal length.

training_set_meas_1 = [training_set_meas_2(:, 1)];
test_set_meas_1 = [test_set_meas_2(:, 1)];

F = 1; % Features reduces to one

x_train_1 = [training_set_meas_1, ones(size(training_set_meas_1,1),1)];
x_test_1 = [test_set_meas_1, ones(size(test_set_meas_1,1),1)];

W = trainLinearClassifier(C, F, x_train_1, training_set_spes);

[~, pred_labels_train] = max(sigmoid(W * x_train_1'), [], 1);
[~, pred_labels_test] = max(sigmoid(W * x_test_1'), [], 1);

pred_labels_train = categorical(class_labels(pred_labels_train));
pred_labels_test = categorical(class_labels(pred_labels_test));

cm_train = confusionchart(true_labels_train, pred_labels_train, ...
    "Title", "Training set");

cm_test = confusionchart(true_labels_test, pred_labels_test, ...
   "Title", "Test set");