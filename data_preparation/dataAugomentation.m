% A code for synthetic image generation
% Full Synthtic deformation operations
% load('completingTemplatePap.mat','completingTemplate');
completingTemplate = zeros([256,256,3]);%imread('background.bmp');
augPath = 'D:\Experiments\Seg_TMI(continued)\MoNuSeg_v2\kumar-dataset\augmented_test\';
augSegPath = augPath;%'D:\imageToSegment\Augomented_maskAll\';
if ~exist(augPath)
    mkdir(augPath);
    mkdir([augPath '\images\']);
    mkdir([augPath '\masks\']);
end
sourcePath = 'D:\Experiments\Seg_TMI(continued)\MoNuSeg_v2\kumar-dataset\test\';
ds = 0.008; % deformation scale
total = 1;
m=256; n=256;
%%% Preparing deformation
[a,b] = meshgrid(linspace(0,1,n), linspace(0,1,m));
xSP = 16; % number of sampling points
ySP = 16;
x = linspace(0,1,xSP);
y = linspace(0,1,ySP);
[X,Y] = meshgrid(x,y);

files = dir([sourcePath 'masks\*.png']);
fileID = fopen([augPath 'fileProps.txt'],'w');
for i = 1:length(files)
    disp(sprintf('%d/%d',i,length(files)))
    imgName = files(i).name(1:end-4);
    
    
    try
    img = imread([sourcePath 'images\' imgName '.png']);
catch
    warning('format is tiff');
    img = imread([sourcePath 'images\' imgName '.tif']);
end
    
%     img = imread([sourcePath 'Tissue images\' imgName '.tif']);
    mask = imread([sourcePath 'masks\' imgName '.png']);
    [mm, nn, ~] = size(img);
    mT = 5;%floor((mm-1)/m)+1;
    nT = 5;%floor((nn-1)/n)+1;
    om = (mT*m-mm)/(mT-1);on = (nT*n-nn)/(nT-1); % calculating overlaps
    if mT==1
        om=0;
    end
    if nT==1
        on=0;
    end
    % Loop over crops!
    for mt = 1:mT
        for nt = 1:nT
            thisCrop = img((m-om)*(mt-1)+1:m*mt-om*(mt-1), (n-on)*(nt-1)+1:n*nt-on*(nt-1), :);
            thisCropMask = mask((m-om)*(mt-1)+1:m*mt-om*(mt-1), (n-on)*(nt-1)+1:n*nt-on*(nt-1), :);
            %%%% Saving the augomented image
            saveImg = thisCrop;
            saveMask = thisCropMask;
            saveName = [imgName '_' num2str(total)];
            fprintf(fileID,'%s\t',saveName);
            fprintf(fileID,'%s\r\n',num2str(i));
            imwrite(saveImg,[augPath 'images\' saveName '.png'],'png');
            imwrite(saveMask,[augPath 'masks\' saveName '_mask.png'],'png');
            total = total + 1;
            
            %                 R = zeros(size(saveMask));G = zeros(size(saveMask));B = zeros(size(saveMask));
            %                 rndColorPallet = .1 + .9.*rand([max(saveMask(:)) , 3]);
            %                 for j = 1: max(saveMask(:))
            %                     thisMask = saveMask==j;
            %                     R(thisMask) = rndColorPallet(j,1);
            %                     G(thisMask) = rndColorPallet(j,2);
            %                     B(thisMask) = rndColorPallet(j,2);
            %                 end
            %                 masksColor = cat(3,R,G,B);
            %                 imwrite(masksColor,[augPath 'images\' saveName '_color.png'],'png');
            
        end
    end
end
fclose(fileID);