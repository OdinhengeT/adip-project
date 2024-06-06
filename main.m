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
data = reshape(vol, prod(sz(1:3)), []);
% or run this line for residuals
%tmp=reshape(resid_vol,prod(sz(1:3)),[]);

data_masked = data(mask(:),:);

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

clearvars idx_lab0 idx_lab1 mod_nbr_lab0 mod_nbr_lab1

% Estimate transform: mean_sample and a 

pca_nbr_components = 100;

[mean_sample, a, mean_projections] = func_estimate_transform(mod_data, mod_labels, pca_nbr_components);

% Modeling Data

mod_projections = a' * (mod_data - mean(mod_data, 1) - mean_sample);

mod_labels_classified = func_nn_classifier(mod_projections, mean_projections, unique(mod_labels));

mod_res = mod_labels_classified - mod_labels;
mod_success_rate = sum(mod_res == 0) / length(mod_res)

% Validation Data

val_projections = a' * (val_data - mean(val_data, 1) - mean_sample);

val_labels_classified = func_nn_classifier(a' * val_data, mean_projections, unique(val_labels));

val_res = val_labels_classified - val_labels;
val_success_rate = sum(val_res == 0) / length(val_res)

%% Shaving

minimal_size = 50; % S_min

% Set up loop variables
nbr_iterations = 7;

nbr_voxels = zeros(nbr_iterations+1, 1);
mod_success_rates = zeros(nbr_iterations+1, 1);
val_success_rates = zeros(nbr_iterations+1, 1);

nbr_voxels(1) = sum( mask(:) );
mod_success_rates(1) = mod_success_rate;
val_success_rates(1) = val_success_rate;

% Logical array indicating values still kept after shaving
a_kept_expanded = mask(:);

for i = 2:nbr_iterations+1
    
    % Estimate transform
    [mean_sample, a, mean_projections,~,~] = func_estimate_transform(data(a_kept_expanded, mod_idx), mod_labels, pca_nbr_components);
    
    % Expand everything
    a_expanded = zeros( numel(mask), 1 );
    a_expanded( a_kept_expanded ) = a;

    % Prepare modeling and validation data for evaluation post shaving
    mod_data_prepped = data(:, mod_idx) - mean( data(a_kept_expanded, mod_idx), 1 );
    mod_data_prepped(a_kept_expanded, :) = mod_data_prepped(a_kept_expanded, :) - mean_sample;
    
    val_data_prepped = data(:, val_idx) - mean( data(a_kept_expanded, val_idx), 1 );
    val_data_prepped(a_kept_expanded, :) = val_data_prepped(a_kept_expanded, :) - mean_sample;
    
    % Reshape a_extended to be the same shape as the original MRI data
    a_vol = reshape(a_expanded, [60, 72, 60]);
    a_kept_vol = reshape(a_kept_expanded, [60, 72, 60]);
    
    % Perform Shaving
    a_kept_vol = func_shaving(a_vol, a_kept_vol, minimal_size);

    % Calculate the shaved a_vol
    a_vol = a_vol .* a_kept_vol;

    % Reshape to expanded form
    a_expanded = reshape(a_vol, prod(sz(1:3)),[]);
    a_kept_expanded = reshape(a_kept_vol, prod(sz(1:3)), []);
    
    % Evaluate
    mod_labels_classified = func_nn_classifier(a_expanded' * mod_data_prepped, mean_projections, unique(mod_labels));
    val_labels_classified = func_nn_classifier(a_expanded' * val_data_prepped, mean_projections, unique(val_labels));

    % Save results
    nbr_voxels(i) = sum( a_kept_expanded );
    
    mod_res = mod_labels_classified - mod_labels;
    mod_success_rates(i) = sum(mod_res == 0) / length(mod_res);
    
    val_res = val_labels_classified - val_labels;
    val_success_rates(i) = sum(val_res == 0) / length(val_res);
end

%% View volume 
bool_plot_volume = true;

if bool_plot_volume
    volumeViewer(a_vol)
    
    %Vol_new = reshape(x_hat, sz);
    %volumeViewer(Vol_new(:,:,:,subject));
end
