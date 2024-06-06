function [...
    mean_sample, ...
    P, ... 
    Lambda, ...
    principal_components, ...
    data_reconstructed, ...
    principal_components_full, ...
    P_full, ...
    Lambda_full ...
    ...
    ] = func_pca(data, nbr_principal_components, bool_fast)
    if nargin < 3; bool_fast = false; end

   %n = size(data, 1);  % Number of values per sample
    m = size(data, 2);  % Number of samples

    if nbr_principal_components <= 0 ; error("func_pca: input 'nbr_components' must be greater than zero."); end
    if nbr_principal_components > m-1
        error("func_pca: input 'nbr_components' must be at most one less than the dimensionallity of the data."); 
    end
    
    % Remove the mean value of each sample from each sample (sample_mean)
    data_sample_mean = mean(data, 1);
    U = data - data_sample_mean; % U is the sample matrix
   
    % Calculate the mean of all samples and remove it from each sample (mean_sample)
    mean_sample = mean(U, 2);
    U = U - mean_sample;
    
    % Eigen-decomposition (with the trick)
    [P_full, Lambda_full] = eig(U'*U/m);
    Lambda_full = diag(Lambda_full);
    
    % Normalize eigenvectors
    P_full = U * P_full * diag( 1./sqrt(m .* Lambda_full) ); 

    % Sort (Matlab gives no guarantees that returned eigenvalues are in a select order)
    [Lambda_full, ind_sort] = sort(Lambda_full, 'descend');
    P_full = P_full(:, ind_sort);
    
    % Truncate 
    P = P_full(:, 1:nbr_principal_components); % Transform from (data - mean_sample) to principal components
    Lambda = Lambda_full(1:nbr_principal_components);
    
    if bool_fast
        % Truncated vector of principal components
        principal_components = P' * U;
        
        % Ignore these
        principal_components_full = NaN;
        data_reconstructed = NaN;
    else 
        % Vector of principal components
        principal_components_full = P_full' * U;

        % Truncated vector of principal components
        principal_components = principal_components_full(1:nbr_principal_components, :);

        % Reconstruction of Data
        data_reconstructed = P * principal_components + mean_sample + data_mean;  
    end
end