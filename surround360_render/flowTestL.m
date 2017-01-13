% load optical flow and flow images
imageL = double(imread('overlap_0_L.png'));
imageR = double(imread('overlap_0_R.png'));

opt = 1;
if opt == 1
    novelViewWarpBuffer = warpL;
    invertT = 0;
    [fx,fy] = readFlowFromFile('../flow/flowRtoL_0.bin');
    [efx,efy] = readFlowFromFile('../flow/flowRtoL_1.bin');
    img = imageL;
    imgExtra = double(imread('overlap_1_L.png'));
elseif opt == 2
    novelViewWarpBuffer = warpL;
    invertT = 1;
    [fx,fy] = readFlowFromFile('../flow/flowLtoR_0.bin');
    [efx,efy] = readFlowFromFile('../flow/flowRtoL_1.bin');
    img = double(imread('overlap_1_R.png'));
    imgExtra = image1L;    
elseif opt == 3
    novelViewWarpBuffer = warpR;
    invertT = 0;
    [fx,fy] = readFlowFromFile('../flow/flowRtoL_0.bin');
    [efx,efy] = readFlowFromFile('../flow/flowLtoR_13.bin');
    img = imageL;
    imgExtra = double(imread('overlap_13_R.png'));
elseif opt == 4
    novelViewWarpBuffer = warpR;
    invertT = 1;
    [fx,fy] = readFlowFromFile('../flow/flowLtoR_0.bin');
    [efx,efy] = readFlowFromFile('../flow/flowLtoR_13.bin');
    img = imageR;
    imgExtra = double(imread('overlap_13_R.png'));
end

currChunkX = 1; 
numCams = 14;
camImageWidth = 1361;
camImageHeight = 1327;
numNovelViews = 450;
vergeAtInfinitySlabDisplacement = 279.554718;
warpL = [];
warpR = [];


% calculate where the chunks are in the image
for nvIdx = 0:numNovelViews-1
    shift = nvIdx / numNovelViews;
    slabShift =  camImageWidth * 0.5 - (numNovelViews - nvIdx);
           
    for v = 0:camImageHeight-1
      warpL(v+1,currChunkX,1) = slabShift + vergeAtInfinitySlabDisplacement;
      warpL(v+1,currChunkX,2) = v;
      warpL(v+1,currChunkX,3) = shift;
            
      warpR(v+1,currChunkX,1) = slabShift - vergeAtInfinitySlabDisplacement;
      warpR(v+1,currChunkX,2) = v;
      warpR(v+1,currChunkX,3) = shift;
    end
    currChunkX = currChunkX + 1;
end


width = numNovelViews;
height = camImageHeight;

startX = novelViewWarpBuffer(1,1,1);
if startX > 0
    limitX = size(imageL,2);
else
    limitX = 0;
end
endX = startX + width;
offset = endX - limitX + 1;
overrun_idx = floor(limitX-startX);


if (startX < 0)
    offset = -450;
    overrun_idx = ceil(limitX-startX)+1;
end


warpOpticalFlow = zeros(height, width,2);
extraWarpOpticalFlow = zeros(height,width,2);

% get the flow for each chunk location
for y = 0:height-1
   for x = 0:width-1
       lazyWarpX = novelViewWarpBuffer(y+1,x+1,1);
       lazyWarpY = novelViewWarpBuffer(y+1,x+1,2);
       warpOpticalFlow(y+1,x+1,:) = cat(3, lazyWarpX, lazyWarpY);
       extraWarpOpticalFlow(y+1,x+1,:) = cat(3, lazyWarpX - offset, lazyWarpY);
   end
end

remappedFlowX = interp2(fx,warpOpticalFlow(:,:,1),warpOpticalFlow(:,:,2),'cubic');
remappedFlowY = interp2(fy,warpOpticalFlow(:,:,1),warpOpticalFlow(:,:,2),'cubic');

extraRemappedFlowX = interp2(efx,extraWarpOpticalFlow(:,:,1),extraWarpOpticalFlow(:,:,2),'cubic');
extraRemappedFlowY = interp2(efy,extraWarpOpticalFlow(:,:,1),extraWarpOpticalFlow(:,:,2),'cubic');

% get use the warped flow to calcluate the texture chunks
warpCompositionX = zeros(height,width);
warpCompositionY = zeros(height,width);
extraWarpCompositionX = zeros(height,width);
extraWarpCompositionY = zeros(height,width);

for y = 0:height-1
   for x = 0:width-1
       lazyWarp = novelViewWarpBuffer(y+1,x+1,:);
       
       flowDirX = remappedFlowX(y+1,x+1);
       flowDirY = remappedFlowY(y+1,x+1);
       
       extraFlowDirX = extraRemappedFlowX(y+1,x+1);
       extraFlowDirY = extraRemappedFlowY(y+1,x+1);
       
       if invertT
           t = 1 - lazyWarp(3);
       else
           t = lazyWarp(3);
       end
       
       warpCompositionX(y+1,x+1) = lazyWarp(1) + flowDirX*t;
       warpCompositionY(y+1,x+1) = lazyWarp(2) + flowDirY*t;
       
       if startX > 0
        extraT = (-floor(offset)+x) / numNovelViews;
       else
        extraT = (-x) / numNovelViews;
       end
       
       extraWarpCompositionX(y+1,x+1) = lazyWarp(1) - offset + extraFlowDirX*extraT;
       extraWarpCompositionY(y+1,x+1) = lazyWarp(2) + extraFlowDirY*extraT;
   end
end

for ii = 1:3
    novelView(:,:,ii) = interp2(img(:,:,ii),warpCompositionX,warpCompositionY,'cubic');
end
for ii = 1:3
    novelViewExtra(:,:,ii) = interp2(imgExtra(:,:,ii),extraWarpCompositionX,extraWarpCompositionY,'cubic');
end

figure;
imagesc([novelView(:,1:overrun_idx,:)/200 novelViewExtra(:,1:(width - overrun_idx),:)/200])
% imagesc([novelViewExtra(:,1:overrun_idx,:)/200 novelView(:,overrun_idx+1:end,:)/200])


axis image

