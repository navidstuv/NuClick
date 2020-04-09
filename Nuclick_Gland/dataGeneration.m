%%% NuClick : Semi-automatic Nuclei instance segmentation
%%% Training/Validation data generation

%% image reading & definitions
set = 'testB';  %either: 'train', 'testA', or 'testB'
imgPath = ['H:\Jahani\DEEP\GlandSegmentation\data\' set '\'];
maskPath = ['H:\Jahani\DEEP\GlandSegmentation\data\' set '_mask\'];
imgSavePath = ['.\\Data\\' set '\\images\\'];
maskSavePath = ['.\\Data\\' set '\\masks\\'];
infosSavePath = ['.\Data\' set '\infos\'];

% making the folders
if ~exist('.\\Data\\')
    mkdir('.\\Data\\');
end
if ~exist(['.\\Data\\' set '\\'])
    mkdir(['.\\Data\\' set '\\']);
end
if ~exist(imgSavePath)
    mkdir(imgSavePath);
end
if ~exist(maskSavePath)
    mkdir(maskSavePath);
end
if ~exist(infosSavePath)
    mkdir(infosSavePath);
end
%%
m = 320;
n = 512;
files = dir([imgPath '*.bmp']);
total=1;
for i = 1:length(files)
    disp (['Working on ' num2str(i)]);
    img = imread([imgPath files(i).name]);
    mask = imread([maskPath files(i).name(1:end-4) '_anno.bmp']);
    img = imresize(img,[m,n]);
    mask = imresize(mask,[m,n],'nearest');
    mask = maskRelabeling (mask);
%     [m,n,~] = size(mask);
       
    for j = 1:max(mask(:)) % do for all bounding boxes
        thisObject = mask==j;
        thisBoundingPoints = boundingPointsExtractor_(thisObject);
        
        otherObjects = uint8(~thisObject).*mask;
        otherObjects = maskRelabeling (otherObjects);
        pcStats = regionprops(otherObjects,'Centroid');
        centroids = round(reshape([pcStats.Centroid],[2,length(pcStats)])');
        cx = centroids(:,1);
        cy = centroids(:,2);
        Indxs = sub2ind(size(otherObjects), cy, cx);
        otherPoints = zeros(size(otherObjects))>0;
        otherPoints(Indxs)=1;

        thisBack = ~thisObject;
        if bwarea(thisObject)>0
            w0 = bwarea(otherObjects)/bwarea(thisObject);
        else
            w0 = 3;
        end
        thisWeightBCE = double(thisBack)+(2)*double(otherObjects>0)+(2+w0)*double(thisObject);
        thisWeightJacc = 4*double(thisObject) + 3*double(thisBack) + double(otherObjects>0);
        
        imagesc(thisWeightBCE); drawnow
        % saving the information for the synthetised image
        saveName = sprintf('gland_%s_%.05d',set,total);
        imwrite(img,[imgSavePath saveName '.png'],'png');
        imwrite(thisObject,[maskSavePath saveName '_mask.png'],'png');
        save([infosSavePath saveName '_info.mat'],'img','thisObject','thisBoundingPoints','otherPoints','thisWeightBCE','thisWeightJacc');
        total=total+1;
    end
end

