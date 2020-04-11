% NEWEST VERSION ::: 1-March-2020
function outMask = maskRelabeling (inMask,sizeLimit)
% A function to take an input label mask and relabeling it with objects
% start with 1 index
outMask = uint8(zeros(size(inMask,1),size(inMask,2)));
uniqueLabels = unique(inMask);
uniqueLabels(1) = [];
i=1;

for t = 1:length(uniqueLabels)
    thisMask = inMask==uniqueLabels(t);
    % Check if two separate objects are labeled similarly
    cc = bwconncomp(thisMask);
    for c = 1:cc.NumObjects
        if length(cc.PixelIdxList{c}) >= sizeLimit
            outMask(cc.PixelIdxList{c}) = i;
            i=i+1;
        end
    end
end
end