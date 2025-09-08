% This demo compares the denoising result of different algorithms in the
% paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; warning('off');
% addpath('omp_box');
randn('seed',0); rand('seed',0);

%% read the ground truth image and generate the noisy image
im_name = 'images/noisyimage_5';   % input image
std_noise = 5;     % standard deviation of Gaussian noise
im_gt = double(imread(im_name));    % ground truth image
% im_n = im_gt + randn(size(im_gt)) * std_noise;  % noisy image with Gaussian noise
disp(['noisy image =  ', im_name, ' + Gaussian noise w/ s.t.d. = ', num2str(std_noise)]);

%% set parameters
algos = ["Ours"];
% algos = ["LNDL","Ours","FODL"]; % the algorithms for running
sz_patch = 16;  % patch size,  %%%% ONLY SQUARE PATCH IS SUPPORTED
n_atom = 256;       % number of atoms in dictionary, i.e., dictioanry size
n_sample = 40000;   % number of training samples
n_iter = 30;    % maxiaum number of iterations in dictionary learning
lambda = 6500;  % model parameter cotrolling sparisty regularization
epsilon = sqrt(sz_patch^2) * std_noise * 1.1;   % target error for omp
UA=1; % number of columns of fixed dictionary A in FODL
lambda1=std_noise*3.5*2.7; % The denoising parameter used in FODL
topK = 10; % 's' in the paper
mu = 6500; % parameter in dictionary proximal term
lambda2= 6800; % parameter in coefficient matrix proximal term

%% denoising
% generate training data
A = im2cols(im_n,sz_patch);
idx = randperm(size(A,2));
Y = A(:,idx(1:n_sample));

% Setting parameters for l0d1
opts_dl.theta = 1;  % maximum step size of C
opts_dl.mu = 1e-3;  % maximum step size of D
opts_dl.n_iter = n_iter;

% Setting the initialization for dictionary learning for 
D0 = dct2dict(sz_patch,n_atom);     % initial dictionary
C0 = omp2(D0'*Y,sum(Y.^2,1),D0'*D0,epsilon);    % initial sparse code

% initialization for ICCV13 since the dictionary size is different(sz_patch)
DD = dct2dict(sz_patch,sz_patch*sz_patch);

% dictionary learning
% Draw the noisy im
figure; subplot(2,2,1); 
imshow(uint8(im_n)); 
title('noisy image');
Dict=cell(1,size(algos,2));

for k = 1 : size(algos,2)
    disp('====================================================================');
    disp('learning dictionary...');

    D=selectDlFunc(Y, lambda, D0, C0, opts_dl, UA, std_noise, DD, algos(1,k), topK, mu, lambda2);
    disp('====================================================================');
    disp('Denoising...');    
    im_r=selectDenoisedFunc(D, A, sz_patch, size(im_gt),epsilon,lambda1, algos(1,k));
    pnsr=10 * log10(255^2 /mean((im_r(:)-im_gt(:)).^2));
    
    %presenting results
    subplot(2,2,k+1);
    imshow(uint8(im_r));
    str2=sprintf("%s with psnr: %f", algos(1,k), 10 * log10(255^2 /mean((im_r(:)-im_gt(:)).^2)));
    title(str2);  
    Dict{1,k}=D;
end

%% Show dictionary
for k = 1 : size(algos,2)
    str=sprintf("%s learned dictionary", algos(1,k));
    figure; dictshow(Dict{1,k}); title(str); 
end
