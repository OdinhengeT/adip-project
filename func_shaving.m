function [a_kept] = func_shaving(a, a_kept, smin)
    
    % Separate positive and negative values 
    pos = a > 0;
    neg = a <= 0;
    
    pos(~a_kept) = 0;
    neg(~a_kept) = 0;
    
    test = {pos,neg};
    
    % Preform connectivity check and omitt objects smaller than smin
    for i = 1:2
        PixelIdxList = regionprops3(test{i}, 'Volume');
        CC = bwconncomp(test{i});
        numPixels = cellfun(@numel,CC.PixelIdxList);
        idx = find(numPixels > smin);
        obj_im{i} = false(size(test{i}));
        for j=1:length (idx)
            obj_im{i}(CC.PixelIdxList{idx(j)}) = true;
        end 
    end 

    % Union
    V = or(obj_im{1},obj_im{2});

    % Find indices and lowest 25 % of values of absolute a and replace with 0 
    
    % Exclude voxels not in V for above
    temp = abs(a);
    temp(~logical(V)) = inf;

    % Reshape necessary form mink function
    temp = reshape(temp,[numel(a),1,1]);

    [~,I] = mink( temp, round( sum(V,'all')/4 ) );
    
    % Do not know a better way to index into corresponding 3D indecies in V
    [ind1, ind2, ind3] = ind2sub(size(a), I);
    for i = 1:length(ind1)
        V( ind1(i), ind2(i), ind3(i) ) = 0;
    end 
    
    a_kept = V;
end 
