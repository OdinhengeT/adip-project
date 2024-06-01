clear all
close all
clc

addpath('./data');
addpath('C:/Toolchains/diplib/share/DIPimage');

load oasis_residual_dataset_subs_05_20150309T105732_19924
load oasis_dataset_subs_05_20150309T105732_19924


%% Prepare Data
sz = size(resid_vol);   % size of data
labels = stats.CDR*2;   % get group label: 0 is controls, 1 is MCI patients
age = stats.Age;

% apply mask, get one matrix with nvoxels x nsubjects
data_masked=reshape(vol,prod(sz(1:3)),[]);
% or run this line for residuals
%tmp=reshape(resid_vol,prod(sz(1:3)),[]);

data_masked=data_masked(mask(:)>0,:);

%% Divide into Modeling and Validation Data

% 20% Validation Data
mod_nbr_lab0 = 98; % out of 122
mod_nbr_lab1 = 48; % out of 60

idx_lab0 = find(labels==0);
idx_lab1 = find(labels==1);

idx_lab0 = idx_lab0( randperm( length( idx_lab0 )));
idx_lab1 = idx_lab1( randperm( length( idx_lab1 )));

mod_idx = [ idx_lab0( 1:mod_nbr_lab0 ); idx_lab1( 1:mod_nbr_lab1 )];
mod_labels = labels(mod_idx);
mod_data = data_masked(:, mod_idx);

val_idx = [ idx_lab0( (1+mod_nbr_lab0):end); idx_lab1( (1+mod_nbr_lab1:end) )];
val_labels = labels(val_idx);
val_data = data_masked(:, val_idx);

clearvars idx_lab0 idx_lab1

%% Modeling Data
%PCA

pca_nbr_components = 115;

[x_hat, M_hat, y_hat, V_hat, Lambda_hat, ~, ~, ~] = func_pca(mod_data, pca_nbr_components);

% LDA

[q, labels_means, Sw, Sb, mean_projections] = func_lda(y_hat, mod_labels);

% NN Classification

mod_labels_classified = func_nn_classifier(q' * y_hat, mean_projections, unique(mod_labels));

mod_res = mod_labels_classified - mod_labels;

mod_success_rate = sum(mod_res == 0)/length(mod_res)

%% Validation Data

val_y_hat = V_hat' * (val_data - mean(val_data, 1) - M_hat);

val_labels_classified = func_nn_classifier(q' * val_y_hat, mean_projections, unique(val_labels));

val_res = val_labels_classified - val_labels;

val_success_rate = sum(val_res == 0)/length(val_res)


%% View volume 
bool_plot_volume = false;

if bool_plot_volume
    subject = 1;
    volumeViewer(vol(:,:,:,subject))
    
    Vol_new = reshape(x_hat, sz);
    volumeViewer(Vol_new(:,:,:,subject));
end
