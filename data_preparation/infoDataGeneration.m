%%% NuClick : Semi-automatic Nuclei instance segmentation
%%%
% Generating info files to be fed into data.py function.
% This funciton accepts image and their relative masks and output a MAT
% file that encompasses all need information about that images.
%%%
%%% Training/Validation data generation
% set application
application = 'Gland'; % either 'Cell', 'Gland', 'Nucleus'
switch application
    case 'Gland'
        sizeLimit = 500;
        m = 512;
        n = 512;
    case 'Cell'
        sizeLimit = 300;
        bb = 256;
    case 'Nucleus'
        sizeLimit = 100;
        bb = 128;
end

%% image reading & definitions
% set the paths for image reading and info saving
set = 'testB';  %either: 'train', 'testA', or 'testB'
imgPath = ['.\Data\GlandSegmentation\data\' set '\'];
maskPath = ['.\Data\GlandSegmentation\data\' set '_mask\'];
infosSavePath = ['.\Data\' set '\infos\'];

imgExt = '*.bmp';
maskExt = '_anno.bmp';

% making the folders
if ~exist(infosSavePath)
    mkdir(infosSavePath);
end
%%
files = dir([imgPath imgExt]);
total=1;
for i = 1:length(files)
    disp (['Working on ' num2str(i)]);
    img = imread([imgPath files(i).name]);
    mask = imread([maskPath files(i).name(1:end-4) maskExt]);
    if strcmp(application,'Gland')
        img = imresize(img,[m,n]);
        thisImg = img;
        mask = imresize(mask,[m,n],'nearest');
    else
        [m,n,~] = size(mask);
    end
    
    mask = maskRelabeling (mask,sizeLimit);
    
    if strcmp(application,'Gland') % for Gland, the process include image resizing
        
        for j = 1:max(mask(:)) % do for all objects in the mask
            thisObject = mask==j;
            [thisPoint,~,~] = clickMapGenerator (mask);
            otherObjects = uint8(~thisObject).*mask;
            otherObjects = maskRelabeling (otherObjects);
            [otherPoints,~,~] = clickMapGenerator (otherObjects);
            
            thisBack = ~(mask>0);
            if bwarea(thisObject)>0
                w0 = (bwarea(otherObjects)+eps)/(bwarea(thisObject)+eps);
            else
                w0 = 1;
            end
            thisWeight = double(thisBack)+(2)*double(otherObjects>0)+(2+w0)*double(thisObject);
            
            imagesc(thisWeightBCE); drawnow
            % saving the information for the synthetised image
            saveName = sprintf('%s_%s_%.03d_%.03d',application,set,i,j);
            save([infosSavePath saveName '_info.mat'],'thisImg','thisObject','thisPoint','otherObjects','otherPoints','thisWeight');
            total=total+1;
        end
        
    else % for processing nucleus and cells, we crop a patch of [bb x bb]
        [clickMap,cx,cy]  = clickMapGenerator (mask);
        
        for j = 1:length(cx) % do for all bounding boxes
            thisCx = cx(j);
            thisCy = cy(j);
            xStart = max(thisCx-bb/2,1);
            yStart = max(thisCy-bb/2,1);
            xEnd = xStart+bb-1;
            yEnd = yStart+bb-1;
            if xEnd > n
                xEnd = n;
                xStart = n-bb+1;
            end
            if yEnd > m
                yEnd = m;
                yStart = m-bb+1;
            end
            
            % Cropping  the image & mask based on the bounding box
            maskVal = mask(thisCy,thisCx);
            if maskVal==0
                continue;
            end
            thisObject = mask==maskVal;
            otherObjects = ((mask>0)-thisObject)>0;
            thisObject = thisObject(yStart:yEnd, xStart:xEnd);
            otherObjects = otherObjects(yStart:yEnd, xStart:xEnd);
            thisImg = img(yStart:yEnd, xStart:xEnd,:);
            thisBack = (1-thisObject-otherObjects)>0;
            
            thisPoint = zeros(m,n)>0;
            thisPoint(thisCy,thisCx)=1;
            otherPoints = (clickMap-thisPoint)>0;
            thisPoint = thisPoint(yStart:yEnd, xStart:xEnd);
            otherPoints = otherPoints(yStart:yEnd, xStart:xEnd);
            
            if bwarea(thisObject)>0
                w0 = (bwarea(otherObjects)+eps)/(bwarea(thisObject)+eps);
            else
                w0 = 1;
            end
            thisWeight = double(thisBack)+(2)*double(otherObjects>0)+(2+w0)*double(thisObject);
            
            imagesc(thisWeightBCE);drawnow
            
            % saving the informaion for the synthetised image
            saveName = sprintf('%s_%s_%.03d_%.03d',application,set,i,j);
            save([infosSavePath saveName '_info.mat'],'thisImg','thisObject','thisPoint','otherObjects','otherPoints','thisWeight');
            total=total+1;
        end   
    end
end

