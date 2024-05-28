function [q, labels_mean, Sw, Sb, mean_projections] = func_lda(y, labels)
    
    ndim = size(y,1);
    
    labels_unique = unique(labels);
    
    labels_mean = zeros(length(labels_unique), ndim);
    for i=1:length(labels_unique)
        labels_mean(i,:) = mean( y(:, labels==labels_unique(i)), 2 );
    end 
    
    y_mean = mean(y,2);
    
    Sw = zeros(ndim);
    Sb = zeros(ndim);

    for i=1:length(labels_unique)
        Sw = Sw + ( y(:, labels==labels_unique(i)) - labels_mean(i) ) * (y(:, labels==labels_unique(i))-labels_mean(i))';
        Sb = Sb + (labels_mean(i)-y_mean) * (labels_mean(i)-y_mean)';
    end
    
    q = Sw \ Sb;
    
    mean_projections= q' * labels_mean';
end 