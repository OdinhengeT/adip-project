function [y, x_hat, V, Lambda, V_hat, Lambda_hat] = func_pca(data, nbr_components)
    
    n = size(data, 1);
    m = size(data, 2);

    if nbr_components <= 0 ; error("func_pca: input 'nbr_components' must be greater than zero."); end
    if nbr_components > m; error("func_pca: input 'nbr_components' must be smaller than dimensionallity of data."); end
    
    data_mean = mean(data, 1);
    
    U = data - data_mean; % Or just data?
    
    % Eigen Decomposition (Trick)
    [Phi, Lambda] = eig(U'*U/m);
    V = U * Phi * diag( 1./sqrt(m .* diag(Lambda)) ); % norm

    % Flip
    V = flip(V, 2);
    Lambda = flip(diag(Lambda));
    
    % Truncate 
    V_hat = V(:, 1:nbr_components);
    Lambda_hat = Lambda(1:nbr_components);
    
    y = V' * U;
    
    x_hat = V_hat * y(1:nbr_components, :) + data_mean; 
    
end