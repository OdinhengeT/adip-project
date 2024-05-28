function [labels] = func_nn_classifier(x_star, x_star_class_means, labels_unique)

    distances = zeros(size(x_star_class_means, 2), size(x_star,2));
    
    for i = 1:size(x_star_class_means, 2)
        distances(i,:) = vecnorm( (x_star - x_star_class_means(:,i)) , 2);
    end
    
    [~,labels] = min(distances, [], 1);
    
    for i = 1:length(labels)
        labels(i) = labels_unique(labels(i));
    end
    
    labels = labels';
end