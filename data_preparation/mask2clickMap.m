function [clickMap,cx,cy] = mask2clickMap (mask)
% Accepts an instance segmetation map and creat a clickMap, which contains
% positive pixels at each nuclei (real) centroid and negative elsewhere.

%simple and fast approache: using centroid opption of regionprops func.
stats = regionprops(mask,'centroid');
centroids = round(reshape([stats.Centroid],[2,length(stats)])');
cx = centroids(:,1);
cy = centroids(:,2);
Indxs = sub2ind(size(mask), cy, cx);
clickMap = zeros(size(mask))>0;
clickMap(Indxs)=1;


