%% Clear Matlab Workspace & clean
close all;
clear;
clc;

%% ------------ Task 1 ------------ %%
load fisheriris;

N_da = 50;          % Length of data set
N_tr = 20;          % Length of trainings set
N_te = N_da-N_tr;   % Length of test set

C = 3; % number of classes
F = 4; % number of features

% first N of each species (setosa, versicolor, virginica)
training_set_meas = [meas(1:N_tr, :); meas(N_da+1:N_da + N_tr, :); meas(2*N_da+1:2*N_da+N_tr, :)];
training_set_spec = [species(1:N_tr); species(N_da+1:N_da + N_tr); species(2*N_da+1:2*N_da+N_tr)];

% last 50-N of each species (setosa, versicolor, virginica)
test_set_meas = [meas(N_tr+1:N_da, :); meas(N_da+N_tr+1:2*N_da, :); meas(2*N_da+N_tr+1:3*N_da, :)];
test_set_spec = [species(N_tr+1:N_da); species(N_da+N_tr+1:2*N_da); species(2*N_da+N_tr+1:3*N_da)];

% Define Setosa, Versicolor and Virinica as
Se = [1 0 0]';
Ve = [0 1 0]';
Vi = [0 0 1]';

%% Create training and test sets
training_set_spes = [kron(ones(1, N_tr), Se), kron(ones(1, N_tr), Ve), kron(ones(1, N_tr), Vi)];
x = [training_set_meas, ones(size(training_set_meas,1),1)];
x_test = [test_set_meas, ones(size(test_set_meas,1),1)];

%% Train Classifier
W = trainLinearClassifier(C, F+1, x, training_set_spes);

% save('W.mat', 'W')
% W = load("W.mat").W;

%% Display result of classification test
sigmoid = @(x) 1./(1 + exp(-x));
[~, predicted_labels] = max(sigmoid(W * x_test'), [], 1);
% disp(predicted_labels);

%% Make Confusion matrix
C = 3;
true_labels = [1 * ones(1,N_te), 2 * ones(1,N_te), 3 * ones(1,N_te)];

% Create confusion matrix
cm = confusionmat(true_labels, predicted_labels);
class_labels = {'Setosa', 'Versicolor', 'Virginica'};

fprintf("[task 1, line 47]\nConfusion mat task 1: \n")
disp(cm');

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
title('Histogram of the four features occuring in the three classes');
subplot(411);
histogram(p1_c1, B); hold on; histogram(p1_c2, B); hold on; histogram(p1_c3, B);
xlabel('cm'); ylabel('frequency'); subtitle('Sepal length');
subplot(412);
histogram(p2_c1, B); hold on; histogram(p2_c2, B); hold on; histogram(p2_c3, B);
xlabel('cm'); ylabel('frequency'); subtitle('Sepal width');
subplot(413); 
histogram(p3_c1, B); hold on; histogram(p3_c2, B); hold on; histogram(p3_c3, B);
xlabel('cm'); ylabel('frequency'); subtitle('Petal length');
subplot(414);
histogram(p4_c1, B); hold on; histogram(p4_c2, B); hold on; histogram(p4_c3, B);
xlabel('cm'); ylabel('frequency'); subtitle('Petal width');

%% Train classifier with 3 features
% By inspection: The sepal width (feature 2) is the feature with most overlap between
% the classes and is therefore removed for the following training.

training_set_meas_3 = [training_set_meas(:,1) training_set_meas(:, 3:4)]; % Remove feature 2
test_set_meas_3 = [test_set_meas(:,1) test_set_meas(:, 3:4)]; % Remove feature 2

F = 3; % features reduced by one

% Training set
x = [training_set_meas_3 ones(size(training_set_meas_3,1),1)];
x_test_3 = [test_set_meas_3 ones(size(test_set_meas_3,1),1)];
t = [kron(ones(1, N_tr), Se), kron(ones(1, N_tr), Ve), kron(ones(1, N_tr), Vi)];


W3 = trainLinearClassifier(C, F+1, x, t);

% Display result of classification test
sigmoid = @(x) 1./(1 + exp(-x));

[~, predicted_labels] = max(sigmoid(W3 * x_test_3'), [], 1);
%disp(predicted_labels);

% Make Confusion matrix

C = 3;
true_labels = [1 * ones(1,N_te), 2 * ones(1,N_te), 3 * ones(1,N_te)];

% Create confusion matrix
cm = confusionmat(true_labels, predicted_labels);
class_labels = {'Setosa', 'Versicolor', 'Virginica'};

fprintf("[task 2, line 111]\n Confusion matrix task 2, 3 labels\n")
disp(cm');
%% Train classifier with 2 features
% By inspection: The sepal length (feature 1) is the next feature with overlap between
% the classes and is therefore be removed for the following training.
training_set_meas_2 = [training_set_meas_3(:, 2:3)]; % Remove feature 1 and 2
test_set_meas_2 = [test_set_meas_3(:, 2:3)]; % Remove feature 1 and 2


F = 2; % features reduced by two
x = [training_set_meas_2, ones(size(training_set_meas_2,1),1)];
x_test_2 = [test_set_meas_2, ones(size(test_set_meas_2,1),1)];
t = [kron(ones(1, N_tr), Se), kron(ones(1, N_tr), Ve), kron(ones(1, N_tr), Vi)];

W2 = trainLinearClassifier(C, F+1, x, t);
[~, predicted_labels] = max(sigmoid(W2 * x_test_2'), [], 1);
cm = confusionmat(true_labels, predicted_labels);
fprintf("[task 2, line 129]\n Confusion matrix task 2, 2 labels\n")
disp(cm');

%% Train classifier with 1 feature
% By inspection: The petal width (feature 4) is the next feature with most overlap between
% the classes and is therefore be removed for the following training. The
% only remaining feature is now the petal length.

F = 1;
training_set_meas_1 = [training_set_meas_2(:, 1)];
test_set_meas_1 = [test_set_meas_2(:, 1)];

x = [training_set_meas_1, ones(size(training_set_meas_1,1),1)];
x_test_1 = [test_set_meas_1, ones(size(test_set_meas_1,1),1)];

t = [kron(ones(1, N_tr), Se), kron(ones(1, N_tr), Ve), kron(ones(1, N_tr), Vi)];
W1 = trainLinearClassifier(C, F+1, x, t);
[~, predicted_labels] = max(sigmoid(W1 * x_test_1'), [], 1);
cm = confusionmat(true_labels, predicted_labels);
fprintf("[task 2, line 142]\n Confusion matrix task 2, 1 label\n")
disp(cm');