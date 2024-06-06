function [labels] = func_nn_classifier(data, data_class_means, labels_unique)

    % Initialize matrix containing the distances to each class mean from each sample in data
    distances = zeros(size(data_class_means, 2), size(data,2));
    
    % Calculate distances to each class mean from each sample in data
    for i = 1:size(data_class_means, 2)
        distances(i,:) = vecnorm( (data - data_class_means(:,i)) , 2, 1);
    end
    
    % Find indicies corresponding to the closest class mean for each sample in data 
    [~,labels] = min(distances, [], 1);
    
    % Convert above index to appropriate label value.
    for i = 1:length(labels)
        labels(i) = labels_unique(labels(i));
    end
    
    labels = labels';
end