% Load the training data into memory
train_data = csvread('/Users/apple/Documents/MSCPROJ/mnist/mtrain.csv');

xTrainImagesCSV= train_data(5:788,:);
tTrainCSV = train_data(1:4,:);
%length(xTrainImages)
%length(tTrain)

formattedxTrainImagesCSV = {};
for i=1:1:7591
    oneTrainImage= xTrainImagesCSV(:,i);
    oneImageMatrix = {reshape(oneTrainImage,28,28)};
    formattedxTrainImagesCSV(:,i)=oneImageMatrix;
end  
formattedtTrainCSV = {};
%{
for i=1:1:7591
    oneTrainLabel = tTrainCSV (1:10,i);
    oneLabelMatrix = {reshape(oneTrainLabel,10,1)};
    formattedtTrainCSV(1,i) = oneTrainLabel;
end
%}
formattedtTrainCSV = num2cell(tTrainCSV);
%length(formattedxTrainImagesCSV)
%formattedtTrainCSV = {}
%for i=1:1:60000
%    oneTrainLabel= tTrainCSV(i,:);
%    onetrainLabelMatrix = {reshape(oneTrainLabel,1,1)};
%    formattedtTrainCSV(:,i)=oneTrainLabel;
%end  
%}
%disp(formattedtTrainCSV)

% We are using display_network from the autoencoder code

%display_network(xTrainImages(:,1:100)); % Show the first 100 images
%disp(tTrain(1:10));
% Display some of the training images
%clf
%for i = 1:20
%    subplot(4,5,i);
%    imshow(xTrainImages{i});
%end


rng('default')
hiddenSize1 = 100;


autoenc1 = trainAutoencoder(formattedxTrainImagesCSV,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

view(autoenc1)

plotWeights(autoenc1);

feat1 = encode(autoenc1,formattedxTrainImagesCSV);

hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

view(autoenc2)

feat2 = encode(autoenc2,feat1);

softnet = trainSoftmaxLayer(feat2,tTrainCSV,'MaxEpochs',100);

view(softnet)

view(autoenc1)
view(autoenc2)
view(softnet)

stackednet = stack(autoenc1,autoenc2,softnet);

view(stackednet)

% Get the number of pixels in each image
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;

% Load the test images
test_data = csvread('/Users/apple/Documents/MSCPROJ/mnist/mtest.csv');
xTestImagesCSV = test_data(5:788,:);
tTestCSV = test_data(1:4,:);

formattedxTestImagesCSV = {};
for i=1:1:401
    oneTestImage= xTestImagesCSV(:,i);
    oneTestImageMatrix = {reshape(oneTestImage,28,28)};
    formattedxTestImagesCSV(:,i)=oneTestImageMatrix;
end
formattedtTestCSV = num2cell(tTestCSV);

%{
formattedtTestCSV = {};
for i=1:1:401
    oneTestLabel = tTestCSV (1:10,i);
    oneTestLabelMatrix = {reshape(oneTestLabel,10,1)};
    formattedtTestCSV(10,i) = oneTestLabelMatrix;
end
%}
%formattedtTestCSV = {}
%for i=1:1:60000
%    oneTestLabel= tTestCSV(i,:);
%    oneLabelMatrix = {reshape(oneTestLabel,1,1)};
%    formattedtTestCSV(:,i)=oneLabelMatrix;
%end  
% clearvars xTrainImagesCSV  hiddenSize1 hiddenSize2 imageHeight imageWidth
% clearvars oneImageMatrix oneTestImage  oneTestImageMatrix oneTrainImage softnet feat1 feat2 i

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(formattedxTestImagesCSV));
for i = 1:numel(formattedxTestImagesCSV)
    xTest(:,i) = formattedxTestImagesCSV{i}(:);
end


y = stackednet(xTest);
plotconfusion(tTestCSV,y);

% Turn the training images into vectors and put them in a matrix
xTrain = zeros(inputSize,numel(formattedxTrainImagesCSV));
for i = 1:numel(formattedxTrainImagesCSV)
    xTrain(:,i) = formattedxTrainImagesCSV{i}(:);
end

% Perform fine tuning
stackednet = train(stackednet,xTrain,tTrainCSV);

y = stackednet(xTest);

plotconfusion(tTestCSV,y);

