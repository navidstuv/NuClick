function outMask = maskRelabeling (inMask)
% A function to take an input label mask and relabeling it with objects
% start with 1 index
outMask = uint8(zeros(size(inMask,1),size(inMask,2)));
uniqueLabels = unique(inMask);
uniqueLabels(1) = [];
i=1;
t=1;
while t <= length(uniqueLabels)
    thisMask = inMask==uniqueLabels(t);
    if bwarea(thisMask) >= 500
        outMask(thisMask) = i;
        i=i+1;
        t=t+1;
    else
        t=t+1;
    end
end
end