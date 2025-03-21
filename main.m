%% ------------ Task 1 ------------ %%
load fisheriris;

C = 3; % number of classes
F = 4; % number of features

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

%% Create training and test sets
t = [kron(ones(1, 30), Se), kron(ones(1, 30), Ve), kron(ones(1, 30), Vi)]; % training set species
x = [training_set_meas, ones(size(training_set_meas,1),1)]; % traning set measurements
x_test = [test_set_meas, ones(size(test_set_meas,1),1)]; % test set measurements 

%% Train Classifier
W = trainLinearClassifier(C, F+1, x, t);
% save('W.mat', 'W')
% W = load("W.mat").W;

%% Display result of classification test
sigmoid = @(x) 1./(1 + exp(-x));
[~, predicted_labels] = max(sigmoid(W * x_test'), [], 1);
% disp(predicted_labels);

%% Make Confusion matrix
N = 20;
C = 3;
true_labels = [1 * ones(1,N), 2 * ones(1,N), 3 * ones(1,N)];

% Create confusion matrix
cm = confusionmat(true_labels, predicted_labels)
class_labels = {'Setosa', 'Versicolor', 'Virginica'};


%% ------------ Task 2 ------------ %%
p1_c1 = meas(1:50,1);
p1_c2 = meas(51:100,1);
p1_c3 = meas(101:150,1);

p2_c1 = meas(1:50,2);
p2_c2 = meas(51:100,2);
p2_c3 = meas(101:150,2);

p3_c1 = meas(1:50,3);
p3_c2 = meas(51:100,3);
p3_c3 = meas(101:150,3);

p4_c1 = meas(1:50,4);
p4_c2 = meas(51:100,4);
p4_c3 = meas(101:150,4);

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

test_set_meas_3 = [test_set_meas(:,1) test_set_meas(:, 3:4)]; % Remove feature 2
F = 3; % features reduced by one

% Training set
x = [test_set_meas_3 ones(size(test_set_meas_3,1),1)];
t = [kron(ones(1, 30), Se), kron(ones(1, 30), Ve), kron(ones(1, 30), Vi)];

W3 = trainLinearClassifier(C, F+1, x, t);

% Display result of classification test
sigmoid = @(x) 1./(1 + exp(-x));
[~, predicted_labels] = max(sigmoid(W * x_test'), [], 1);
disp(predicted_labels);

% Make Confusion matrix
N = 20;
C = 3;
true_labels = [1 * ones(1,N), 2 * ones(1,N), 3 * ones(1,N)];

% Create confusion matrix
cm = confusionmat(true_labels, predicted_labels)
class_labels = {'Setosa', 'Versicolor', 'Virginica'};


%% Train classifier with 2 features
% By inspection: The sepal length (feature 1) is the next feature with overlap between
% the classes and is therefore be removed for the following training.
test_set_meas_2 = [test_set_meas_3(:, 2:3)]; % Remove feature 1 and 2
F = 2; % features reduced by two

%% Train classifier with 1 feature
% By inspection: The petal width (feature 4) is the next feature with most overlap between
% the classes and is therefore be removed for the following training. The
% only remaining feature is now the petal length.
test_set_meas_1 = [test_set_meas_2(:, 1)];