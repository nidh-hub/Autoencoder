Files=dir('/Users/apple/Documents/MSCPROJ/images/images_0_5_5/crop_tr/*.jpg');
for k=1:34
   FileNames = Files(k).name;
   fpath = fullfile('/Users/apple/Documents/MSCPROJ/images/images_0_5_5/crop_tr/',FileNames);
   RGBImage = imread(fpath);
   g = rgb2gray(RGBImage);
   %imshow(g)
   %imshow(RGBImage);
   b = reshape(g, 1, []);
   %csvwrite('/Users/apple/Documents/MSCPROJ/csv/img055tri',b);
   dlmwrite('/Users/apple/Documents/MSCPROJ/csv/img055tri.csv',b,'delimiter',',','-append','newline','unix');
  
end
% a = dlmread('/Users/apple/Documents/MSCPROJ/csv/img005tri.csv');
data = csvread('/Users/apple/Documents/MSCPROJ/csv/img055tri.csv');
    
formatted_data = [reshape(data,7,422500)];
formatted_data(:,422501) = 055;
%formatted_data = transpose(formatted_data);
dblToCell = num2cell(formatted_data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%   0_5_10   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Files10=dir('/Users/apple/Documents/MSCPROJ/images/images_0_5_10/crop_0_5_10tr/*.jpg');
for k=1:34
   FileNames10 = Files10(k).name;
   fpath10 = fullfile('/Users/apple/Documents/MSCPROJ/images/images_0_5_10/crop_0_5_10tr/',FileNames10);
   RGBImage10 = imread(fpath10);
   g10 = rgb2gray(RGBImage10);
   %imshow(g)
   %imshow(RGBImage);
   b10 = reshape(g10, 1, []);
   %csvwrite('/Users/apple/Documents/MSCPROJ/csv/img055tri',b);
   dlmwrite('/Users/apple/Documents/MSCPROJ/csv/img0510tri.csv',b10,'delimiter',',','-append','newline','unix');
  
end
% a = dlmread('/Users/apple/Documents/MSCPROJ/csv/img005tri.csv');
data10 = csvread('/Users/apple/Documents/MSCPROJ/csv/img0510tri.csv');
    
formatted_data10 = [reshape(data10,7,422500)];
formatted_data10(:,422501) = 0510;
%formatted_data10 = transpose(formatted_data10);
dblToCell10 = num2cell(formatted_data10);


%%%%%%%%%%%%%%  AUTOENCODER CODE %%%%%%%%%%%%%%%%%%%%%%%

train_data = [formatted_data; formatted_data10];
xTrainImages = train_data(:,1:422500);
%xTrainImages = num2cell(xTrainImages);
tTrain = train_data(:,422501);

formattedxTrainImages = {};
for i=1:1:14
    oneTrainImage= xTrainImages(i,:);
    oneImageMatrix = {reshape(oneTrainImage,650,650)};
    formattedxTrainImages(i,:)=oneImageMatrix;
end  
% formattedtTrain = {};
% formattedtTrain = num2cell(tTrain);

rng('default')
hiddenSize1 = 8;


autoenc1 = trainAutoencoder(formattedxTrainImages,hiddenSize1, ...
    'MaxEpochs',4, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

view(autoenc1)
figure()
plotWeights(autoenc1);

feat1 = encode(autoenc1,formattedxTrainImages);
hiddenSize2 = 4;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',4, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

view(autoenc2)

feat2 = encode(autoenc2,feat1);

softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',5);

view(softnet)
