% *************************************************************************
% * This code applies the iterative shrinkage / thresholding algorithms
%   (ISTAs) to solve the image deblurring problem:
%
%           min { 0.5*|| A(x) - y ||_2^2 + lambda*|| x ||_TV }.
%            x
% 
%   where x and y are two-dimensional arrays representing the estimate for
%   the deblurred image and the observed blurry image, respectively. A(x)
%   denotes the blurring linear operator, and lambda is the regularization
%   weight.
% *************************************************************************
% * Author : Yunhui Gao
% * Date   : 2021/04/20
% *************************************************************************

%% 
% =========================================================================
% Initialization
% =========================================================================

clear;clc;
close all;

% load source functions and test image
addpath(genpath('../src'))
img = im2double(imread('../data/cameraman.tif'));

% Gaussian kernel
kernel_size = 9;
kernel_sigma = 4;
kernel = fspecial('gaussian',[kernel_size,kernel_size], kernel_sigma);

% define function handles
A  = @(x) imgcrop((imfilter(x,kernel)),kernel_size);    % forward linear operator A
AT = @(x) (imfilter(zeropad(x,kernel_size),kernel));    % transpose (adjoint operator) of A

% forward model
x = zeropad(img,kernel_size);
y = A(x);
y = y + normrnd(0, 1e-3, size(y));  % add Gaussian noise

% display the image
figure
subplot(1,2,1),imshow(img,[])
title('Ground truth','fontsize',16,'interpreter','latex')
subplot(1,2,2),imshow(y,[])
title('Observation','fontsize',16,'interpreter','latex')
set(gcf,'unit','normalized','position',[0.25,0.3,0.5,0.4])

%% 
% =========================================================================
% Run the deblurring algorithm
% =========================================================================

rng(0)  % random seed, for reproducibility

n_iters = 10;    % number of iterations to solve the denoising subproblem
penalty = @(x) normTVi(x);   % isotropic TV norm as the penalty function
prox_op = @(x,gamma) proxTVi(x,gamma,n_iters);       % proximity operator

x_init = rand(size(x));      % random initialization
n_iters = 500;               % number of iterations
lambda = 2e-5;
[x,n_iters,J_vals,~] = TwIST(y,A,lambda,...   % try ISTA, TwIST, or FISTA
    'AT',           AT,...
    'initializer',  x_init,...
    'prox_op',      prox_op,...
    'penalty',      penalty,...
    'eta',          2,...
    'Lip',          1.5,... 
    'max_iter',     n_iters,...
    'min_iter',     n_iters,...
    'verbose',      true);

figure,imshow(imgcrop(x,kernel_size),[])
title(['Reconstruction after ',num2str(n_iters),' iterations'],'interpreter','latex')

%% 
% =========================================================================
% Comparison of different algorithms
% =========================================================================

% parameters
n_iters = 200;
lambda = 2e-5;

[x_ista,~,J_ista,~] = ISTA(y,A,lambda,...       % ISTA
    'AT',           AT,...
    'initializer',  2,...
    'prox_op',      prox_op,...
    'penalty',      penalty,...
    'max_iter',     n_iters,...
    'min_iter',     n_iters,...
    'verbose',      true);

[x_fista,~,J_fista,~] = FISTA(y,A,lambda,...    % FISTA
    'AT',           AT,...
    'initializer',  2,...
    'prox_op',      prox_op,...
    'penalty',      penalty,...
    'max_iter',     n_iters,...
    'min_iter',     n_iters,...
    'verbose',      true);

[x_twist,~,J_twist,~] = TwIST(y,A,lambda,...    % TwIST
    'AT',           AT,...
    'initializer',  2,...
    'prox_op',      prox_op,...
    'penalty',      penalty,...
    'max_iter',     n_iters,...
    'min_iter',     n_iters,...
    'verbose',      true);

%% display the results

figure
subplot(1,3,1),imshow(x_ista,[])
title(['ISTA ($J(\mathbf{x}) = $',num2str(J_ista(201),'%4.3f'),')'],'interpreter','latex','fontsize',14)
subplot(1,3,2),imshow(x_fista,[])
title(['FISTA ($J(\mathbf{x}) = $',num2str(J_fista(201),'%4.3f'),')'],'interpreter','latex','fontsize',14)
subplot(1,3,3),imshow(x_twist,[])
title(['TwIST ($J(\mathbf{x}) = $',num2str(J_twist(201),'%4.3f'),')'],'interpreter','latex','fontsize',14)
sgtitle('Reconstruction with 200 iterations','fontsize',18,'interpreter','latex')
set(gcf,'unit','normalized','position',[0.15,0.25,0.7,0.45])

% plot the curves
step = 5;
colors = ['r';'g';'b';'m';'c'];
figure
semilogy(0:step:n_iters,J_ista(1:step:(n_iters+1)),'-s','linewidth',1.5,'color',colors(1),'markerfacecolor',colors(1),'markersize',6)
hold on,semilogy(0:step:n_iters,J_fista(1:step:(n_iters+1)),'-^','linewidth',1.5,'color',colors(2),'markerfacecolor',colors(2),'markersize',6)
hold on,semilogy(0:step:n_iters,J_twist(1:step:(n_iters+1)),'-o','linewidth',1.5,'color',colors(3),'markerfacecolor',colors(3),'markersize',6)
legend('ISTA','FISTA','TwIST','interpreter','latex','fontsize',14)
xlabel('Iterations','interpreter','latex','fontsize',18)
ylabel('Objective function $J(\mathbf{x})$','interpreter','latex','fontsize',18)
title('Convergence behavior','fontsize',18,'interpreter','latex')
grid on
set(gcf,'unit','normalized','position',[0.15,0.25,0.7,0.45])

%% 
% =========================================================================
% Auxiliary functions
% =========================================================================

function u = imgcrop(x,cropsize)
% =========================================================================
% Crop the central part of the image.
% -------------------------------------------------------------------------
% Input:    - x        : Original image.
%           - cropsize : Cropping pixel number along each dimension.
% Output:   - u        : Cropped image.
% =========================================================================
u = x(cropsize+1:end-cropsize,cropsize+1:end-cropsize);
end

function u = zeropad(x,padsize)
% =========================================================================
% Zero-pad the image.
% -------------------------------------------------------------------------
% Input:    - x        : Original image.
%           - padsize  : Padding pixel number along each dimension.
% Output:   - u        : Zero-padded image.
% =========================================================================
u = padarray(x,[padsize,padsize],0);
end
