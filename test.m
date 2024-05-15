clear all;
close all;
clc

rng(0);

nbr1 = 100;
nbr2 = nbr1;

mu1 = [5, 3.4, 0.0];
mu2 = [-1, -2.1, 0.0];
sigma1 = diag([0.5, 0.2, 1.5]);
sigma2 = diag([0.6, 0.21, 1.5]);

data1 = mvnrnd(mu1, sigma1, 100);
data2 = mvnrnd(mu2, sigma2, 100);

figure(); hold on;
scatter3(data1(:, 1), data1(:, 2), data1(:, 3))
scatter3(data2(:, 1), data2(:, 2), data2(:, 3))
title('Original Data')
hold off;

data = [data1; data2];
label = [zeros(nbr1, 1); ones(nbr2,1)];

% PCA

%[y,x_hat,~,~,U_hat] = func_pca(data', 2);
[y,x_hat,~,~,U_hat] = func_pca_original(data, 2);

%y = y';
%x_hat = x_hat';

figure(); hold on;
scatter(y(:, 1), y(:, 2))
title('Principal Component Domain')
hold off;

figure(); 
scatter3(x_hat(:, 1), x_hat(:, 2), x_hat(:, 3))
title('Principal Component Truncated Reconstruction')
hold off;

%% LDA

y1_mean = mean( y(1:50, :) );
y2_mean = mean( y(101:150, :) );
y_mean = mean( [y(1:50, :); y(101:150, :)] );

Sw = 0.01 * ( (y(1:50, :)-y1_mean)' * (y(1:50, :)-y1_mean) + (y(101:150, :)-y2_mean)' * (y(101:150, :)-y2_mean) ); 
Sb = 0.01 * ( (y(1:50, :)-y_mean)' * (y(1:50, :)-y_mean) + (y(101:150, :)-y_mean)' * (y(101:150, :)-y_mean) );

q = Sw \ Sb;
a = U_hat'*q;

a' * data(55, :)'
a' * data(155, :)'

q' * y1_mean'
q' * y2_mean'




