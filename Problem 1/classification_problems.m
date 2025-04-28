% classification_problems.m

% Clear workspace and figures
clear; close all; clc;

% Set random seed for reproducibility
rng(1);

% Create figure
figure;

%% 1. Linearly Separable
subplot(1,3,1);
hold on;

% Generate two Gaussian blobs linearly separable
mu1 = [1 1]; sigma1 = 0.3;
mu2 = [3 3]; sigma2 = 0.3;

class1 = mvnrnd(mu1, sigma1*eye(2), 100);
class2 = mvnrnd(mu2, sigma2*eye(2), 100);

scatter(class1(:,1), class1(:,2), 30, 'o', 'filled');
scatter(class2(:,1), class2(:,2), 30, 'b', 'filled');
title('Linearly Separable');
axis equal; grid on;

%% 2. Non-Linearly Separable
subplot(1,3,2);
hold on;

% Make a circular (concentric) dataset
theta = linspace(0, 2*pi, 100)';
r1 = 0.5 + 0.1*rand(100,1);
r2 = 1.0 + 0.1*rand(100,1);

class1 = [r1.*cos(theta), r1.*sin(theta)];
class2 = [r2.*cos(theta), r2.*sin(theta)];

scatter(class1(:,1), class1(:,2), 30, 'r', 'filled');
scatter(class2(:,1), class2(:,2), 30, 'b', 'filled');
title('Non-Linearly Separable');
axis equal; grid on;

%% 3. Non-Separable
subplot(1,3,3);
hold on;

% Two overlapping Gaussian distributions
mu1 = [0 0]; sigma1 = 1.0;
mu2 = [1 1]; sigma2 = 1.0;

class1 = mvnrnd(mu1, sigma1*eye(2), 100);
class2 = mvnrnd(mu2, sigma2*eye(2), 100);

scatter(class1(:,1), class1(:,2), 30, 'r', 'filled');
scatter(class2(:,1), class2(:,2), 30, 'b', 'filled');
title('Non-Separable');
axis equal; grid on;

% Overall title
sgtitle('Classification Problem Examples');
