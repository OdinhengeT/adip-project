function [q, mean_projections, Sw, Sb, labels_mean] = func_lda(data, labels) 
    n = size(data,1);   % Number of values per sample
   %m = size(data, 2);  % Number of samples
    
    labels_unique = unique(labels);
    
    labels_mean = zeros(length(labels_unique), n);
    for i=1:length(labels_unique)
        labels_mean(i,:) = mean( data(:, labels==labels_unique(i)), 2 );
    end 
    
    % Pre-calculate mean of all samples
    y_mean = mean(data,2);
    
    % Initialize Sample matrices
    Sw = zeros(n);
    Sb = zeros(n);

    % Add contributions to the sample matrices from each sample per label category
    for i=1:length(labels_unique)
        Sw = Sw + ( data(:, labels==labels_unique(i)) - labels_mean(i) ) * (data(:, labels==labels_unique(i))-labels_mean(i))';
        Sb = Sb + (labels_mean(i)-y_mean) * (labels_mean(i)-y_mean)';
    end
    
    % Generalized eigen-decomposition
    [V, D] = eig(Sb, Sw);
    
    % This method requires inv(Sw) to exist, could be done differently if this were to become a problem
    %Q = Sw \ Sb;
    
    % Eigen-decomposition of Q
    %[V, D] = eig(Q);
    D = diag(D);
    
    % Isolate the nbr_unique_labels - 1 largest eigenvalues (These are the only ones which 'are' non-zero) 
    [~, idxs] = maxk(D, length(labels_unique)-1);
    
    % Eigenvectors corresponding to the above eigenvalues yield the Fisher discriminant transform
    q = V(:, idxs); 
    
    % Calculate the projections of the means for each label
    mean_projections = q' * labels_mean';
end 