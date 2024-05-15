function [q, means, Sw, Sb, mean_projections] = func_lda(y,label)
    
    ndim = size(y,2);

    ulabels=unique(label);
    means=zeros(length(ulabels));
    for i=1:length(ulabels)
        means(i)=mean(y(label==ulabels(i)));
    end 
    y_mean = mean(y);
    Sw=zeros(ndim);
    Sb=zeros(ndim);

    for i=1:length(ulabels)
        Sw =Sw+ (y(label==ulabels(i))-means(i))' * (y(label==ulabels(i))-means(i));
        Sb =Sb+ (means(i)-y_mean)' * (means(i)-y_mean);
    end
    
    q = Sw \ Sb;
    
    mean_projections= q' * means';
end 