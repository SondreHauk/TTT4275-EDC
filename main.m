%% ------------ Task 1 ------------ %%
load fisheriris;

% first 30 of each species (setosa, versicolor, virginica)
training_set_meas = [meas(1:30, :); meas(51:80, :); meas(101:130, :)];
training_set_spec = [species(1:30); species(51:80); species(101:130)];

% last 20 of each species (setosa, versicolor, virginica)
test_set_meas = [meas(31:50, :); meas(81:100, :); meas(131:150, :)];
test_set_spec = [species(31:50); species(81:100); species(131:150)];

% Define Setosa, Versicolor and Virinica as
Se = [1 0 0]';
Ve = [0 1 0]';
Vi = [0 0 1]';

disp("test")
%% Create training and test sets
training_set_spes = [kron(ones(1, 30), Se), kron(ones(1, 30), Ve), kron(ones(1, 30), Vi)];
x = [training_set_meas, ones(size(training_set_meas,1),1)];
x_test = [test_set_meas, ones(size(test_set_meas,1),1)];

%% Train Classifier
% W = trainLinearClassifier(x, training_set_spes);
% save('W.mat', 'W')
W = load("W.mat").W;

%% Display result of classification test
sigmoid = @(x) 1./(1 + exp(-x));
[~, predicted_labels] = max(sigmoid(W * x_test'), [], 1);
disp(predicted_labels);

%% Make Confusion matrix
N = 20;
C = 3;
true_labels = [1 * ones(1,N), 2 * ones(1,N), 3 * ones(1,N)];

% Create confusion matrix
cm = confusionmat(true_labels, predicted_labels);
class_labels = {'Setosa', 'Versicolor', 'Virginica'};