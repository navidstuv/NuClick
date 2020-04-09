function boundingPoints = boundingPointsExtractor(thisObject)
    %%
    [m,n] = size(thisObject);
    stats = regionprops(thisObject,'BoundingBox');
    bb = [stats.BoundingBox];
    xoff = randi(10);
    yoff = randi(10);
    xStart = ceil(bb(1)) + xoff;
    yStart = ceil(bb(2)) + yoff;
    width = floor(bb(3)) - xoff - randi(5);
    height = floor(bb(4)) - yoff - randi(5);
    
    rectImg = zeros(m,n)>0;
    leftLine = rectImg;
    leftLine (yStart+2:yStart+height-2, xStart) = 1;
    leftLine = (leftLine.*thisObject)>0;
    leftLine = bwareafilt(leftLine, 1);
    
    rightLine = rectImg;
    rightLine (yStart+2:yStart+height-2, xStart+width-1) = 1;
    rightLine = (rightLine.*thisObject)>0;
    rightLine = bwareafilt(rightLine, 1);
    
    upLine = rectImg;
    upLine (yStart, xStart+2:xStart+width-2) = 1;
    upLine = (upLine.*thisObject)>0;
    upLine = bwareafilt(upLine, 1);
    
    downLine = rectImg;
    downLine (yStart+height-1, xStart+2:xStart+width-2) = 1;
    downLine = (downLine.*thisObject)>0;
    downLine = bwareafilt(downLine, 1);
    
    rectImg = (leftLine+rightLine+upLine+downLine)>0;
    pointCandidcates = rectImg .* thisObject;
    pcStats = regionprops(bwlabel(pointCandidcates),'Centroid');
    centroids = round(reshape([pcStats.Centroid],[2,length(pcStats)])');
    cx = centroids(:,1);
    cy = centroids(:,2);
    Indxs = sub2ind(size(thisObject), cy, cx);
    boundingPoints = zeros(size(thisObject))>0;
    boundingPoints(Indxs)=1;

end