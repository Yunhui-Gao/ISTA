% *************************************************************************
% * This code applies the iterative shrinkage / thresholding algorithms
%   (ISTAs) to solve the LASSO problem:
%
%      min { 1 / (2*n_samples) || A x - y ||_2^2 + lambda*|| x ||_1 }.
%       x
% 
%   where x is a an array of shape (n_features, 1), y is an array of shape
%   (n_samples, 1), and A is an array of shape (n_samples, n_features).
%   lambda is the regularization coefficient.
% *************************************************************************
% * Author : Yunhui Gao
% * Date   : 2021/04/20
% *************************************************************************

%% generate data
clear;clc;
close all;

% load source functions
addpath(genpath('../src'))

rng(0)  % random seed, for reproducibility

n_samples = 10;     % number of samples
n_features = 5;     % number of features

A = zeros(n_samples,n_features);
for i = 1:n_features
    A(:,i) = exprnd(i,n_samples,1);
end
x = [0, 2, 0, -3, 0]';   % x has only two nonzero components
e = randn(n_samples,1)*0.1;     % Gaussian noise
y = A*x + e;    % observation

%% LASSO regression
lambda = 10.^(-5:1:5);    % sequence of regularization parameters
x_est = zeros(n_features,length(lambda));   % estimate of x
for i = 1:length(lambda)
    x_est(:,i) = ISTA(y,A,n_samples*lambda(i),...   % try ISTA, TwIST, or FISTA
        'max_iter',1000,...
        'tol',1e-10);
end

%% display result
colors = ['r';'g';'b';'m';'c'];
figure
h1 = semilogx(lambda,x_est(1,:),'-s','linewidth',1.5,'color',colors(1),'markerfacecolor',colors(1)); hold on
h2 = semilogx(lambda,x_est(2,:),'-^','linewidth',1.5,'color',colors(2),'markerfacecolor',colors(2)); hold on
h3 = semilogx(lambda,x_est(3,:),'-o','linewidth',1.5,'color',colors(3),'markerfacecolor',colors(3)); hold on
h4 = semilogx(lambda,x_est(4,:),'-d','linewidth',1.5,'color',colors(4),'markerfacecolor',colors(4)); hold on
h5 = semilogx(lambda,x_est(5,:),'-*','linewidth',1.5,'color',colors(5),'markerfacecolor',colors(5)); hold on
semilogx(lambda,x(2)*ones(size(lambda)),'--','linewidth',1.5,'color',colors(2)); hold on
semilogx(lambda,x(4)*ones(size(lambda)),'--','linewidth',1.5,'color',colors(4));
ylim([-4,3])
grid on
legend([h1 h2 h3 h4 h5],'$x_1$','$x_2$','$x_3$','$x_4$','$x_5$','interpreter','latex','fontsize',14,'location','southeast')
xlabel('$\lambda$','interpreter','latex','fontsize',18)
ylabel('$x_i$','interpreter','latex','fontsize',18)
title('LASSO regression','interpreter','latex','fontsize',18)
set(gcf,'unit','normalized','position',[0.15,0.25,0.7,0.5])