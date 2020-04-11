%%
clear all
close all
path = '.\kumar-dataset\augmented_test\masks\';
imgPath = '.\kumar-dataset\augmented_test\images\';
infoPath = '.\kumar-dataset\augmented_test\infos\';
if ~exist(infoPath)
    mkdir(infoPath)
end
files = dir([path '*.png']);
for f = 1:length(files)
    maskName = files(f).name;
    masks = imread([path maskName]);
    masks = bwlabel(masks);
    masks = maskRelabeling (masks); % relabeling masks, avoiding redundant computations for crops
    img = imread([imgPath maskName(1:end-9) '.png']);
    for i=1:max(masks(:))
        thisMask = masks==i;
        allMasks(:,:,i) = thisMask;
    end
    if max(masks(:))==0
        w = ones(size(img(:,:,1)));
        margins = zeros(size(img(:,:,1)))>0;
        maskBW = zeros(size(img(:,:,1)))>0;
        separationBorderBW = zeros(size(img(:,:,1)))>0;
        allMaskSeparated = zeros(size(img(:,:,1)))>0;
        imagesc(w);colorbar,colormap jet;drawnow
        disp(['***' maskName '***'])
        save([infoPath maskName(1:end-9) '_info.mat'],'img','w','margins','maskBW','separationBorderBW','allMaskSeparated','luminance');
        continue;
    end
    se = ones(3);%strel('disk',1);
    allMasksDilated = imdilate(allMasks,se);
    allMargins = allMasksDilated - allMasks;
    margins2 = sum(allMargins,3);
    allMaskSeparated2 = ~margins2.*sum(allMasks,3);
    % figure,imshow(allMaskSeparated2)
    margins2 = margins2>0;
    maskBW = masks>0;
    separationBorderBW2 = (maskBW - allMaskSeparated2)>0;
    % figure,imshow(~separationBorder.*maskBW)
    
    %% constructing d1 and d2 togheter
    % calculating the distance of each pixel from each cell
    for i = 1:max(masks(:))
        thisPerimeter = bwperim(allMasks(:,:,i));
        thisDistance = bwdist(thisPerimeter,'euclidean');
        allDistance (:,:,i) = thisDistance;
        allPerimeters (:,:,i) = thisPerimeter;
    end
    margins = sum(allPerimeters,3);
    separationBorderBW = separationBorderBW2.* margins;
    allMaskSeparated = maskBW - separationBorderBW;
    
    allSortedDistance = sort(allDistance,3);
    if size(allSortedDistance,3) == 1
        backWeight = ~separationBorderBW.* (~allMaskSeparated*.5 + 1.5);
        w = allMaskSeparated+ (~allMaskSeparated).*(backWeight);
    else
        d1 = allSortedDistance(:,:,1);
        d2 = allSortedDistance(:,:,2);
        
        % Creating the final weight map
        if bwarea(separationBorderBW)>0
            sepWeight = bwarea(allMaskSeparated>0)/bwarea(separationBorderBW);%20; # separation borderWeights
        else
            sepWeight = 9;
        end
        if bwarea(margins)>0
            alpha = bwarea(allMaskSeparated>0)/bwarea((margins-separationBorderBW)>0);%20; # separation borderWeights
            marginWeight = alpha*(margins-separationBorderBW);
        else
            sepWeight = alpha*(margins-separationBorderBW);
        end
        sigma = 5;
        backWeight = ~separationBorderBW .* (~allMaskSeparated*.5 + .5);
        
        w = allMaskSeparated+ (~allMaskSeparated).*(backWeight+sepWeight*exp(-((d1+d2).^2)./(2*5^2))) + marginWeight;
    end
    imagesc(w);colorbar,colormap jet;drawnow
    disp(maskName)
%     pause()
    margins = margins>0;
    separationBorderBW = separationBorderBW>0;
    allMaskSeparated = allMaskSeparated>0;
    
    % obtaining the residual weight map for small objects
    residualMap = 9*(maskBW - bwareaopen(maskBW,90))+1;

    save([infoPath maskName(1:end-9) '_info.mat'],'img','w','margins','maskBW','separationBorderBW','allMaskSeparated','residualMap');
    clear allMasks, clear allDistance, clear allPerimeters
    % figure,imagesc((~allMaskSeparated2).*(~allMaskSeparated2+w0*exp(-((d1+d2).^2)./(2*4.5^2))))
    % colorbar,colormap jet
    
end