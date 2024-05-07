function [y, x_hat, U, D, U_hat] = func_pca(data, nbr_components)
    
    ndim = size(data,2);
    if nbr_components <= 0 ; error("func_pca: input 'nbr_components' must be greater than zero."); end
    if nbr_components > ndim; error("func_pca: input 'nbr_components' must be smaller than dimensionallity of data."); end
    
    
    x_mean = mean(data);
    x_cov = cov(data);

    [U, D] = eig(x_cov);
    U = flip(U);
    %D = flip(flip(D, 1), 2);

    %U_hat = zeros(size(U));
    %U_hat(1:nbr_comp,:) = U(1:nbr_comp, :);
    U_hat = U(1:nbr_components, :);

    y = ( U_hat * (data - x_mean)')';
    x_hat = x_mean + y*U_hat;
end