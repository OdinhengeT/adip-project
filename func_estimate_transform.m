function [mean_sample, a, mean_projections, P, q] = func_estimate_transform(data, labels, pca_nbr_components)

    % PCA
    [mean_sample, P, ~, principal_components, ~, ~, ~, ~] = func_pca(data, pca_nbr_components, true);
    
    % LDA
    [q, mean_projections, ~, ~, ~] = func_lda(principal_components, labels);
    
    % Calculate a, transform from data to decision-space
    a = P * q;
end