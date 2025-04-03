%% Clear Matlab Workspace & clean
close all;
clear;
clc;

%% ------------ Task 1 ------------ %%

% Use last 30 for training and 20 first for testing.

load fisheriris;

N_da =          50;   % Length of data set
N_te =          20;   % Length of test set
N_tr = N_da - N_te;   % Length of training set

C = 3; % number of classes
F = 4; % number of features

% choose the first 20 of each species for testing
test_set_feat = [
    meas(1:N_te, :);
    meas(N_da+1:N_da + N_te, :);
    meas(2*N_da+1:2*N_da+N_te, :)];

test_set_label = [
    species(1:N_te); 
    species(N_da+1:N_da + N_te); 
    species(2*N_da+1:2*N_da+N_te)];

% choose the last 30 of each species for training
training_set_feat = [
    meas(N_te+1:N_da, :); 
    meas(N_da+N_te+1:2*N_da, :); 
    meas(2*N_da+N_te+1:3*N_da, :)];

training_set_label = [
    species(N_te+1:N_da); 
    species(N_da+N_te+1:2*N_da); 
    species(2*N_da+N_te+1:3*N_da)];

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

cm_test = confusionchart(true_labels_test, pred_labels_test, ...
  "Title","Test set");