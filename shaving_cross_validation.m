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

%% Run Cross-Validation for iterations of Shaving

% 20% Validation Data
mod_nbr_lab0 = 98; % out of 122
mod_nbr_lab1 = 48; % out of 60

idx_lab_0 = find(labels==0);
idx_lab_1 = find(labels==1);

% Parameters
pca_nbr_components = 100;
minimal_size = 50;

% Interation Parameters
nbr_repeats = 100;
nbr_iterations_max = 10;

% Initialize result matrices
nbr_voxels = zeros(nbr_iterations_max+1, nbr_repeats);

% sha = shave last
mod_error_rates_sha = zeros(nbr_iterations_max+1, nbr_repeats);
val_error_rates_sha = zeros(nbr_iterations_max+1, nbr_repeats);

% est = esstimation last
mod_error_rates_est = zeros(nbr_iterations_max+1, nbr_repeats);
val_error_rates_est = zeros(nbr_iterations_max+1, nbr_repeats);

for j = 1:nbr_repeats
    
    % Randomize Modeling and Validation Data Selection
    idx_lab0 = idx_lab_0( randperm( length( idx_lab_0 )));
    idx_lab1 = idx_lab_1( randperm( length( idx_lab_1 )));

    mod_idx = [ idx_lab0( 1:mod_nbr_lab0 ); idx_lab1( 1:mod_nbr_lab1 )];
    mod_labels = labels(mod_idx);
    mod_data = data(:, mod_idx);

    val_idx = [ idx_lab0( (1+mod_nbr_lab0):end); idx_lab1( (1+mod_nbr_lab1:end) )];
    val_labels = labels(val_idx);
    val_data = data(:, val_idx);
    
    % Inital Estimate (No Shaving)
    a_kept_expanded = mask(:);
    
    [mean_sample, a, mean_projections,~,~] = func_estimate_transform(mod_data(a_kept_expanded, :), mod_labels, pca_nbr_components);
    
    % Expand a
    a_expanded = zeros( numel(mask), 1 );
    a_expanded( a_kept_expanded ) = a;
    
    % Prepare modeling and validation data for evaluation post shaving
    mod_data_prepped = mod_data - mean( mod_data(a_kept_expanded, :), 1 );
    mod_data_prepped(a_kept_expanded, :) = mod_data_prepped(a_kept_expanded, :) - mean_sample;

    val_data_prepped = val_data - mean( val_data(a_kept_expanded, :), 1 );
    val_data_prepped(a_kept_expanded, :) = val_data_prepped(a_kept_expanded, :) - mean_sample;
    
    % Evaluate
    mod_labels_classified = func_nn_classifier(a_expanded' * mod_data_prepped, mean_projections, unique(mod_labels));
    val_labels_classified = func_nn_classifier(a_expanded' * val_data_prepped, mean_projections, unique(val_labels));
    
    % Save results
    nbr_voxels(1, j) = sum( a_kept_expanded );

    mod_res = mod_labels_classified - mod_labels;
    mod_error_rates_est(1, j) = 1 - sum(mod_res == 0) / length(mod_res);
    mod_error_rates_sha(1, j) = 1 - sum(mod_res == 0) / length(mod_res);

    val_res = val_labels_classified - val_labels;
    val_error_rates_est(1, j) = 1 - sum(val_res == 0) / length(val_res);
    val_error_rates_sha(1, j) = 1 - sum(val_res == 0) / length(val_res);
    
    % Shave
    for i = 1 + (1:nbr_iterations_max)
        
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
        nbr_voxels(i, j) = sum( a_kept_expanded );

        mod_res = mod_labels_classified - mod_labels;
        mod_error_rates_sha(i, j) = 1 - sum(mod_res == 0) / length(mod_res);

        val_res = val_labels_classified - val_labels;
        val_error_rates_sha(i, j) = 1 - sum(val_res == 0) / length(val_res);
        
        % Estimate new transform
        [mean_sample, a, mean_projections,~,~] = func_estimate_transform(data(a_kept_expanded, mod_idx), mod_labels, pca_nbr_components);

        % Expand a
        a_expanded = zeros( numel(mask), 1 );
        a_expanded( a_kept_expanded ) = a;

        % Prepare modeling and validation data for evaluation post shaving
        mod_data_prepped = mod_data - mean( mod_data(a_kept_expanded, :), 1 );
        mod_data_prepped(a_kept_expanded, :) = mod_data_prepped(a_kept_expanded, :) - mean_sample;

        val_data_prepped = val_data - mean( val_data(a_kept_expanded, :), 1 );
        val_data_prepped(a_kept_expanded, :) = val_data_prepped(a_kept_expanded, :) - mean_sample;
        
        % Evaluate
        mod_labels_classified = func_nn_classifier(a_expanded' * mod_data_prepped, mean_projections, unique(mod_labels));
        val_labels_classified = func_nn_classifier(a_expanded' * val_data_prepped, mean_projections, unique(val_labels));

        % Save results
        nbr_voxels(i, j) = sum( a_kept_expanded );

        mod_res = mod_labels_classified - mod_labels;
        mod_error_rates_est(i, j) = 1 - sum(mod_res == 0) / length(mod_res);

        val_res = val_labels_classified - val_labels;
        val_error_rates_est(i, j) = 1 - sum(val_res == 0) / length(val_res);

    end
end 

%% Plot Estimation Last

figure();
ax=axes;
ax.XScale='log';
xlim([0.0005, 1.05])
ylim([-5, 50])
hold on
xbar = std( nbr_voxels./nbr_voxels(1,1), 0, 2 ); 
ybar = 100*std(mod_error_rates_est, 0, 2);
errorbar(mean(nbr_voxels, 2)./nbr_voxels(1,1), 100*mean(mod_error_rates_est, 2), ybar, ybar, xbar, xbar, 'k')
scatter( 100,100 , 'or', 'filled' )
for r = 1:nbr_repeats
    scatter(nbr_voxels(:, r)./nbr_voxels(1), 100*mod_error_rates_est(:, r), 'or', 'filled', 'MarkerFaceAlpha', 0.05,'MarkerEdgeAlpha', 0.05 )
end
errorbar(mean(nbr_voxels, 2)./nbr_voxels(1,1), 100*mean(mod_error_rates_est, 2), ybar, ybar, xbar, xbar, 'k')
xticks([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
xticklabels({'0.05', '0.1', '0.02', '0.5', '1', '2', '5', '10', '20', '50', '100'})
legend('Mean per Iteration of Shaving', 'Scatter of all aquired data points')
title('Estimation Last: Error Rates on Modelings Data')
xlabel('Fraction of Voxels Retained in Shaving [%]')
ylabel('Classification Error [%]')
grid on; hold off;

figure();
ax=axes;
ax.XScale='log';
xlim([0.0005, 1.05])
ylim([-5, 70])
hold on
xbar = std( nbr_voxels./nbr_voxels(1,1), 0, 2 ); 
ybar = 100*std(val_error_rates_est, 0, 2);
errorbar(mean(nbr_voxels, 2)./nbr_voxels(1,1), 100*mean(val_error_rates_est, 2), ybar, ybar, xbar, xbar, 'k')
scatter( 100,100 , 'or', 'filled' )
for r = 1:nbr_repeats
    scatter(nbr_voxels(:, r)./nbr_voxels(1), 100*val_error_rates_est(:, r), 'or', 'filled', 'MarkerFaceAlpha', 0.05,'MarkerEdgeAlpha', 0.05 )
end
errorbar(mean(nbr_voxels, 2)./nbr_voxels(1,1), 100*mean(val_error_rates_est, 2), ybar, ybar, xbar, xbar, 'k')
xticks([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
xticklabels({'0.05', '0.1', '0.02', '0.5', '1', '2', '5', '10', '20', '50', '100'})
legend('Mean per Iteration of Shaving', 'Scatter of all aquired data points')
title('Estimation Last: Error Rates on Validation Data')
xlabel('Fraction of Voxels Retained in Shaving')
ylabel('Classification Error [%]')
grid on; hold off;


%% Plot Shaving Last

figure();
ax=axes;
ax.XScale='log';
xlim([0.0005, 1.05])
ylim([-5, 50])
hold on
xbar = std( nbr_voxels./nbr_voxels(1,1), 0, 2 ); 
ybar = 100*std(mod_error_rates_sha, 0, 2);
errorbar(mean(nbr_voxels, 2)./nbr_voxels(1,1), 100*mean(mod_error_rates_sha, 2), ybar, ybar, xbar, xbar, 'k')
scatter( 100,100 , 'or', 'filled' )
for r = 1:nbr_repeats
    scatter(nbr_voxels(:, r)./nbr_voxels(1), 100*mod_error_rates_sha(:, r), 'or', 'filled', 'MarkerFaceAlpha', 0.05,'MarkerEdgeAlpha', 0.05 )
end
errorbar(mean(nbr_voxels, 2)./nbr_voxels(1,1), 100*mean(mod_error_rates_sha, 2), ybar, ybar, xbar, xbar, 'k')
xticks([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
xticklabels({'0.05', '0.1', '0.02', '0.5', '1', '2', '5', '10', '20', '50', '100'})
legend('Mean per Iteration of Shaving', 'Scatter of all aquired data points')
title('Shaving Last: Error Rates on Modelings Data')
xlabel('Fraction of Voxels Retained in Shaving [%]')
ylabel('Classification Error [%]')
grid on; hold off;

figure();
ax=axes;
ax.XScale='log';
xlim([0.0005, 1.05])
ylim([-5, 70])
hold on
xbar = std( nbr_voxels./nbr_voxels(1,1), 0, 2 ); 
ybar = 100*std(val_error_rates_sha, 0, 2);
errorbar(mean(nbr_voxels, 2)./nbr_voxels(1,1), 100*mean(val_error_rates_sha, 2), ybar, ybar, xbar, xbar, 'k')
scatter( 100,100 , 'or', 'filled' )
for r = 1:nbr_repeats
    scatter(nbr_voxels(:, r)./nbr_voxels(1), 100*val_error_rates_sha(:, r), 'or', 'filled', 'MarkerFaceAlpha', 0.05,'MarkerEdgeAlpha', 0.05 )
end
errorbar(mean(nbr_voxels, 2)./nbr_voxels(1,1), 100*mean(val_error_rates_sha, 2), ybar, ybar, xbar, xbar, 'k')
xticks([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
xticklabels({'0.05', '0.1', '0.02', '0.5', '1', '2', '5', '10', '20', '50', '100'})
legend('Mean per Iteration of Shaving', 'Scatter of all aquired data points')
title('Shaving Last: Error Rates on Validation Data')
xlabel('Fraction of Voxels Retained in Shaving')
ylabel('Classification Error [%]')
grid on; hold off;