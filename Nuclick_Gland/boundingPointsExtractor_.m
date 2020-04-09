function boundingPoints = boundingPointsExtractor_(thisObject)
    %%
    [m,n] = size(thisObject);
    stats = regionprops(thisObject,'BoundingBox');
    bb = [stats.BoundingBox];
       
    centroids = [];
    
    rectImg = zeros(m,n)>0;
    leftLine = rectImg;
    leftLine (ceil(bb(2)):ceil(bb(2))+floor(bb(4))-1, ceil(bb(1))) = 1;
    leftLine = (leftLine.*thisObject)>0;
    leftLine = bwareafilt(leftLine, 1);
    pcStats = regionprops(leftLine,'Centroid');
    centroids = [centroids pcStats.Centroid];
    
    rightLine = rectImg;
    rightLine (ceil(bb(2)):ceil(bb(2))+floor(bb(4))-1, ceil(bb(1))+floor(bb(3))-1) = 1;
    rightLine = (rightLine.*thisObject)>0;
    rightLine = bwareafilt(rightLine, 1);
    pcStats = regionprops(rightLine,'Centroid');
    centroids = [centroids pcStats.Centroid];
    
    upLine = rectImg;
    upLine (ceil(bb(2)), ceil(bb(1)):ceil(bb(1))+floor(bb(3))-1) = 1;
    upLine = (upLine.*thisObject)>0;
    upLine = bwareafilt(upLine, 1);
    pcStats = regionprops(upLine,'Centroid');
    centroids = [centroids pcStats.Centroid];
    
    downLine = rectImg;
    downLine (ceil(bb(2))+floor(bb(4))-1, ceil(bb(1)):ceil(bb(1))+floor(bb(3))-1) = 1;
    downLine = (downLine.*thisObject)>0;
    downLine = bwareafilt(downLine, 1);
    pcStats = regionprops(downLine,'Centroid');
    centroids = [centroids pcStats.Centroid];
    
    centroids = round(reshape(centroids,[2,length(centroids)/2])');
    cx = centroids(:,1);
    cy = centroids(:,2);
    Indxs = sub2ind(size(thisObject), cy, cx);
    boundingPoints = zeros(size(thisObject))>0;
    boundingPoints(Indxs)=1;

end