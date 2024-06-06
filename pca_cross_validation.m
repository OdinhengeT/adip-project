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
data = reshape(vol,prod(sz(1:3)),[]);
% or run this line for residuals
%tmp=reshape(resid_vol,prod(sz(1:3)),[]);

data_masked = data(mask(:),:);

%% Run Cross-Validation for nbr of Principal Components

% 20% Validation Data
mod_nbr_lab0 = 98; % out of 122
mod_nbr_lab1 = 48; % out of 60

idx_lab_0 = find(labels==0);
idx_lab_1 = find(labels==1);

nbr_repeats = 100;
pca_nbr_components_max = mod_nbr_lab0 + mod_nbr_lab1 - 1;

mod_fwd_error_rates = zeros(pca_nbr_components_max, nbr_repeats);
val_fwd_error_rates = zeros(pca_nbr_components_max, nbr_repeats);

mod_bwd_error_rates = zeros(pca_nbr_components_max, nbr_repeats);
val_bwd_error_rates = zeros(pca_nbr_components_max, nbr_repeats);

parfor i = 1:pca_nbr_components_max
    for j = 1:nbr_repeats
        
        % Randomize Modeling and Validation Data
       
        idx_lab0 = idx_lab_0( randperm( length( idx_lab_0 )));
        idx_lab1 = idx_lab_1( randperm( length( idx_lab_1 )));

        mod_idx = [ idx_lab0( 1:mod_nbr_lab0 ); idx_lab1( 1:mod_nbr_lab1 )];
        mod_labels = labels(mod_idx);
        mod_data = data_masked(:, mod_idx);

        val_idx = [ idx_lab0( (1+mod_nbr_lab0):end); idx_lab1( (1+mod_nbr_lab1:end) )];
        val_labels = labels(val_idx);
        val_data = data_masked(:, val_idx);
        
        % Estimate Transforms
        
        % PCA
        [mean_sample, P, ~, principal_components, ~, ~, ~, ~] = func_pca(mod_data, pca_nbr_components_max, true);
        
        P_fwd = P(:, 1:i);
        P_bwd = P(:, i:end);
        
        mod_pcs_fwd = principal_components(1:i, :); %P_fwd' * ( (mod_data - mean(mod_data, 1)) - mean_sample);
        mod_pcs_bwd = P_bwd' * ( (mod_data - mean(mod_data, 1)) - mean_sample);
        
        % LDA
        [q_fwd, mean_projections_fwd, ~, ~, ~] = func_lda(mod_pcs_fwd, mod_labels);
        [q_bwd, mean_projections_bwd, ~, ~, ~] = func_lda(mod_pcs_bwd, mod_labels);

        % NN Classification
        mod_classified_fwd = func_nn_classifier(q_fwd' * mod_pcs_fwd, mean_projections_fwd, unique(mod_labels));
        mod_classified_bwd = func_nn_classifier(q_bwd' * mod_pcs_bwd, mean_projections_bwd, unique(mod_labels));
        
        mod_res_fwd = mod_classified_fwd - mod_labels;
        mod_res_bwd = mod_classified_bwd - mod_labels;

        mod_fwd_error_rates(i, j) = 1 - ( sum(mod_res_fwd == 0)/length(mod_res_fwd) );
        mod_bwd_error_rates(i, j) = 1 - ( sum(mod_res_bwd == 0)/length(mod_res_bwd) );
        
        
        % Validation Data

        val_pcs_fwd = P_fwd' * ( (val_data - mean(val_data, 1)) - mean_sample);
        val_pcs_bwd = P_bwd' * ( (val_data - mean(val_data, 1)) - mean_sample);

        val_classified_fwd = func_nn_classifier(q_fwd' * val_pcs_fwd, mean_projections_fwd, unique(val_labels));
        val_classified_bwd = func_nn_classifier(q_bwd' * val_pcs_bwd, mean_projections_bwd, unique(val_labels));

        val_res_fwd = val_classified_fwd - val_labels;
        val_res_bwd = val_classified_bwd - val_labels;

        val_fwd_error_rates(i, j) = 1 - ( sum(val_res_fwd == 0)/length(val_res_fwd) );
        val_bwd_error_rates(i, j) = 1 - ( sum(val_res_bwd == 0)/length(val_res_bwd) );
       
    end
end 

mod_bwd_error_rates = flip( mod_bwd_error_rates );
val_bwd_error_rates = flip( val_bwd_error_rates );


%% Plot results

figure(); hold on; 
errorbar(1:pca_nbr_components_max, 100*mean(mod_fwd_error_rates, 2), 100*std(mod_fwd_error_rates, 0, 2))
errorbar(1:pca_nbr_components_max, 100*mean(mod_bwd_error_rates, 2), 100*std(mod_bwd_error_rates, 0, 2))
ylim([-5, 65])
title('Error Rates on Modelings Data')
xlabel('Number of Principal Components')
ylabel('Classification Error [%]')
legend('Forward', 'Backward')
grid on; hold off;

figure(); hold on; 
errorbar(1:pca_nbr_components_max, 100*mean(val_fwd_error_rates, 2), 100*std(val_fwd_error_rates, 0, 2))
errorbar(1:pca_nbr_components_max, 100*mean(val_bwd_error_rates, 2), 100*std(val_bwd_error_rates, 0, 2))
ylim([-5, 65])
title('Error Rates on Validation Data')
xlabel('Number of Principal Components')
ylabel('Classification Error [%]')
legend('Forward', 'Backward')
grid on; hold off;


%% Plot result2

figure(); hold on; 
plot(1:pca_nbr_components_max, 100*mean(mod_fwd_error_rates, 2))
plot(1:pca_nbr_components_max, 100*mean(mod_bwd_error_rates, 2))
ylim([-5, 65])
title('Error Rates on Modelings Data')
xlabel('Number of Principal Components')
ylabel('Classification Error [%]')
legend('Forward', 'Backward')
grid on; hold off;

figure(); hold on; 
plot(1:pca_nbr_components_max, 100*mean(val_fwd_error_rates, 2))
plot(1:pca_nbr_components_max, 100*mean(val_bwd_error_rates, 2))
ylim([-5, 65])
title('Error Rates on Validation Data')
xlabel('Number of Principal Components')
ylabel('Classification Error [%]')
legend('Forward', 'Backward')
grid on; hold off;












